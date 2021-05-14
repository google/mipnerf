# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different model implementation plus a general port for all the models."""
import functools
from typing import Any, Callable
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp

from internal import mip
from internal import utils


@gin.configurable
class MipNerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs."""
  num_samples: int = 128  # The number of samples per level.
  num_levels: int = 2  # The number of sampling levels.
  resample_padding: float = 0.01  # Dirichlet/alpha "padding" on the histogram.
  stop_level_grad: bool = True  # If True, don't backprop across levels')
  use_viewdirs: bool = True  # If True, use view directions as a condition.
  lindisp: bool = False  # If True, sample linearly in disparity, not in depth.
  ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 16  # Max degree of positional encoding for 3D points.
  deg_view: int = 4  # Degree of positional encoding for viewdirs.
  density_activation: Callable[..., Any] = nn.softplus  # Density activation.
  density_noise: float = 0.  # Standard deviation of noise added to raw density.
  density_bias: float = -1.  # The shift added to raw densities pre-activation.
  rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  disable_integration: bool = False  # If True, use PE instead of IPE.

  @nn.compact
  def __call__(self, rng, rays, randomized, white_bkgd):
    """The mip-NeRF Model.

    Args:
      rng: jnp.ndarray, random number generator.
      rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
      randomized: bool, use randomized stratified sampling.
      white_bkgd: bool, if True, use white as the background (black o.w.).

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
    # Construct the MLP.
    mlp = MLP()

    ret = []
    for i_level in range(self.num_levels):
      key, rng = random.split(rng)
      if i_level == 0:
        # Stratified sampling along rays
        t_vals, samples = mip.sample_along_rays(
            key,
            rays.origins,
            rays.directions,
            rays.radii,
            self.num_samples,
            rays.near,
            rays.far,
            randomized,
            self.lindisp,
            self.ray_shape,
        )
      else:
        t_vals, samples = mip.resample_along_rays(
            key,
            rays.origins,
            rays.directions,
            rays.radii,
            t_vals,
            weights,
            randomized,
            self.ray_shape,
            self.stop_level_grad,
            resample_padding=self.resample_padding,
        )
      if self.disable_integration:
        samples = (samples[0], jnp.zeros_like(samples[1]))
      samples_enc = mip.integrated_pos_enc(
          samples,
          self.min_deg_point,
          self.max_deg_point,
      )

      # Point attribute predictions
      if self.use_viewdirs:
        viewdirs_enc = mip.pos_enc(
            rays.viewdirs,
            min_deg=0,
            max_deg=self.deg_view,
            append_identity=True,
        )
        raw_rgb, raw_density = mlp(samples_enc, viewdirs_enc)
      else:
        raw_rgb, raw_density = mlp(samples_enc)

      # Add noise to regularize the density predictions if needed.
      if randomized and (self.density_noise > 0):
        key, rng = random.split(rng)
        raw_density += self.density_noise * random.normal(
            key, raw_density.shape, dtype=raw_density.dtype)

      # Volumetric rendering.
      rgb = self.rgb_activation(raw_rgb)
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
      density = self.density_activation(raw_density + self.density_bias)
      comp_rgb, distance, acc, weights = mip.volumetric_rendering(
          rgb,
          density,
          t_vals,
          rays.directions,
          white_bkgd=white_bkgd,
      )
      ret.append((comp_rgb, distance, acc))

    return ret


def construct_mipnerf(rng, example_batch):
  """Construct a Neural Radiance Field.

  Args:
    rng: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  model = MipNerfModel()
  key, rng = random.split(rng)
  init_variables = model.init(
      key,
      rng=rng,
      rays=utils.namedtuple_map(lambda x: x[0], example_batch['rays']),
      randomized=False,
      white_bkgd=False)
  return model, init_variables


@gin.configurable
class MLP(nn.Module):
  """A simple MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  net_activation: Callable[..., Any] = nn.relu  # The activation function.
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  num_rgb_channels: int = 3  # The number of RGB channels.
  num_density_channels: int = 1  # The number of density channels.

  @nn.compact
  def __call__(self, x, condition=None):
    """Evaluate the MLP.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.
      condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.

    Returns:
      raw_rgb: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_rgb_channels].
      raw_density: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_density_channels].
    """
    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
    inputs = x
    for i in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    raw_density = dense_layer(self.num_density_channels)(x).reshape(
        [-1, num_samples, self.num_density_channels])

    if condition is not None:
      # Output of the first part of MLP.
      bottleneck = dense_layer(self.net_width)(x)
      # Broadcast condition from [batch, feature] to
      # [batch, num_samples, feature] since all the samples along the same ray
      # have the same viewdir.
      condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
      # Collapse the [batch, num_samples, feature] tensor to
      # [batch * num_samples, feature] so that it can be fed into nn.Dense.
      condition = condition.reshape([-1, condition.shape[-1]])
      x = jnp.concatenate([bottleneck, condition], axis=-1)
      # Here use 1 extra layer to align with the original nerf model.
      for i in range(self.net_depth_condition):
        x = dense_layer(self.net_width_condition)(x)
        x = self.net_activation(x)
    raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape(
        [-1, num_samples, self.num_rgb_channels])
    return raw_rgb, raw_density


def render_image(render_fn, rays, rng, chunk=8192):
  """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function.
    rays: a `Rays` namedtuple, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    chunk: int, the size of chunks to render sequentially.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  height, width = rays[0].shape[:2]
  num_rays = height * width
  rays = utils.namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)

  host_id = jax.host_id()
  results = []
  for i in range(0, num_rays, chunk):
    # pylint: disable=cell-var-from-loop
    chunk_rays = utils.namedtuple_map(lambda r: r[i:i + chunk], rays)
    chunk_size = chunk_rays[0].shape[0]
    rays_remaining = chunk_size % jax.device_count()
    if rays_remaining != 0:
      padding = jax.device_count() - rays_remaining
      chunk_rays = utils.namedtuple_map(
          lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # host_count.
    rays_per_host = chunk_rays[0].shape[0] // jax.host_count()
    start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
    chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]),
                                      chunk_rays)
    chunk_results = render_fn(rng, chunk_rays)[-1]
    results.append([utils.unshard(x[0], padding) for x in chunk_results])
    # pylint: enable=cell-var-from-loop
  rgb, distance, acc = [jnp.concatenate(r, axis=0) for r in zip(*results)]
  rgb = rgb.reshape((height, width, -1))
  distance = distance.reshape((height, width))
  acc = acc.reshape((height, width))
  return (rgb, distance, acc)
