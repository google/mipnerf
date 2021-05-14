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
"""Helper functions for visualizing things."""
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.cm as cm


def sinebow(h):
  """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
  f = lambda x: jnp.sin(jnp.pi * x)**2
  return jnp.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def convolve2d(z, f):
  return jsp.signal.convolve2d(
      z, f, mode='same', precision=jax.lax.Precision.HIGHEST)


def depth_to_normals(depth):
  """Assuming `depth` is orthographic, linearize it to a set of normals."""
  f_blur = jnp.array([1, 2, 1]) / 4
  f_edge = jnp.array([-1, 0, 1]) / 2
  dy = convolve2d(depth, f_blur[None, :] * f_edge[:, None])
  dx = convolve2d(depth, f_blur[:, None] * f_edge[None, :])
  inv_denom = 1 / jnp.sqrt(1 + dx**2 + dy**2)
  normals = jnp.stack([dx * inv_denom, dy * inv_denom, inv_denom], -1)
  return normals


def visualize_depth(depth,
                    acc=None,
                    near=None,
                    far=None,
                    curve_fn=lambda x: jnp.log(x + jnp.finfo(jnp.float32).eps),
                    modulus=0,
                    colormap=None):
  """Visualize a depth map.

  Args:
    depth: A depth map.
    acc: An accumulation map, in [0, 1].
    near: The depth of the near plane, if None then just use the min().
    far: The depth of the far plane, if None then just use the max().
    curve_fn: A curve function that gets applied to `depth`, `near`, and `far`
      before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
    modulus: If > 0, mod the normalized depth by `modulus`. Use (0, 1].
    colormap: A colormap function. If None (default), will be set to
      matplotlib's viridis if modulus==0, sinebow otherwise.

  Returns:
    An RGB visualization of `depth`.
  """
  # If `near` or `far` are None, identify the min/max non-NaN values.
  eps = jnp.finfo(jnp.float32).eps
  near = near or jnp.min(jnp.nan_to_num(depth, jnp.inf)) - eps
  far = far or jnp.max(jnp.nan_to_num(depth, -jnp.inf)) + eps

  # Curve all values.
  depth, near, far = [curve_fn(x) for x in [depth, near, far]]

  # Wrap the values around if requested.
  if modulus > 0:
    value = jnp.mod(depth, modulus) / modulus
    colormap = colormap or sinebow
  else:
    # Scale to [0, 1].
    value = jnp.nan_to_num(jnp.clip((depth - near) / (far - near), 0, 1))
    colormap = colormap or cm.get_cmap('viridis')

  vis = colormap(value)[:, :, :3]

  # Set non-accumulated pixels to white.
  if acc is not None:
    vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

  return vis


def visualize_normals(depth, acc, scaling=None):
  """Visualize fake normals of `depth` (optionally scaled to be isotropic)."""
  if scaling is None:
    mask = ~jnp.isnan(depth)
    x, y = jnp.meshgrid(
        jnp.arange(depth.shape[1]), jnp.arange(depth.shape[0]), indexing='xy')
    xy_var = (jnp.var(x[mask]) + jnp.var(y[mask])) / 2
    z_var = jnp.var(depth[mask])
    scaling = jnp.sqrt(xy_var / z_var)

  scaled_depth = scaling * depth
  normals = depth_to_normals(scaled_depth)
  vis = jnp.isnan(normals) + jnp.nan_to_num((normals + 1) / 2, 0)

  # Set non-accumulated pixels to white.
  if acc is not None:
    vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

  return vis


def visualize_suite(depth, acc):
  """A wrapper around other visualizations for easy integration."""
  vis = {
      'depth': visualize_depth(depth, acc),
      'depth_mod': visualize_depth(depth, acc, modulus=0.1),
      'depth_normals': visualize_normals(depth, acc)
  }
  return vis
