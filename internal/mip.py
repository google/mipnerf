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
"""Helper functions for mip-NeRF."""

from jax import lax
from jax import random
import jax.numpy as jnp

from internal import math


def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                   list(x.shape[:-1]) + [-1])
  four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  if append_identity:
    return jnp.concatenate([x] + [four_feat], axis=-1)
  else:
    return four_feat


def expected_sin(x, x_var):
  """Estimates mean and variance of sin(z), z ~ N(x, var)."""
  # When the variance is wide, shrink sin towards zero.
  y = jnp.exp(-0.5 * x_var) * math.safe_sin(x)
  y_var = jnp.maximum(
      0, 0.5 * (1 - jnp.exp(-2 * x_var) * math.safe_cos(2 * x)) - y**2)
  return y, y_var


def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[..., None, :] * t_mean[..., None]

  d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = jnp.eye(d.shape[-1])
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
  """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: jnp.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
  if stable:
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                      (3 * mu**2 + hw**2)**2)
    r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                              (hw**4) / (3 * mu**2 + hw**2))
  else:
    t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
    r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
    t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_var = t_mosq - t_mean**2
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
  """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: jnp.float32 3-vector, the axis of the cylinder
    t0: float, the starting distance of the cylinder.
    t1: float, the ending distance of the cylinder.
    radius: float, the radius of the cylinder
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t_mean = (t0 + t1) / 2
  r_var = radius**2 / 4
  t_var = (t1 - t0)**2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
  """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    t_vals: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
  t0 = t_vals[..., :-1]
  t1 = t_vals[..., 1:]
  if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
  elif ray_shape == 'cylinder':
    gaussian_fn = cylinder_to_gaussian
  else:
    assert False
  means, covs = gaussian_fn(directions, t0, t1, radii, diag)
  means = means + origins[..., None, :]
  return means, covs


def integrated_pos_enc(x_coord, min_deg, max_deg, diag=True):
  """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

  Args:
    x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
      be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
    min_deg: int, the min degree of the encoding.
    max_deg: int, the max degree of the encoding.
    diag: bool, if true, expects input covariances to be diagonal (full
      otherwise).

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if diag:
    x, x_cov_diag = x_coord
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    shape = list(x.shape[:-1]) + [-1]
    y = jnp.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = jnp.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
  else:
    x, x_cov = x_coord
    num_dims = x.shape[-1]
    basis = jnp.concatenate(
        [2**i * jnp.eye(num_dims) for i in range(min_deg, max_deg)], 1)
    y = math.matmul(x, basis)
    # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
    # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
    y_var = jnp.sum((math.matmul(x_cov, basis)) * basis, -2)

  return expected_sin(
      jnp.concatenate([y, y + 0.5 * jnp.pi], axis=-1),
      jnp.concatenate([y_var] * 2, axis=-1))[0]


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
  """Volumetric Rendering Function.

  Args:
    rgb: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
    density: jnp.ndarray(float32), density, [batch_size, num_samples, 1].
    t_vals: jnp.ndarray(float32), [batch_size, num_samples].
    dirs: jnp.ndarray(float32), [batch_size, 3].
    white_bkgd: bool.

  Returns:
    comp_rgb: jnp.ndarray(float32), [batch_size, 3].
    disp: jnp.ndarray(float32), [batch_size].
    acc: jnp.ndarray(float32), [batch_size].
    weights: jnp.ndarray(float32), [batch_size, num_samples]
  """
  t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
  t_dists = t_vals[..., 1:] - t_vals[..., :-1]
  delta = t_dists * jnp.linalg.norm(dirs[..., None, :], axis=-1)
  # Note that we're quietly turning density from [..., 0] to [...].
  density_delta = density[..., 0] * delta

  alpha = 1 - jnp.exp(-density_delta)
  trans = jnp.exp(-jnp.concatenate([
      jnp.zeros_like(density_delta[..., :1]),
      jnp.cumsum(density_delta[..., :-1], axis=-1)
  ],
                                   axis=-1))
  weights = alpha * trans

  comp_rgb = (weights[..., None] * rgb).sum(axis=-2)
  acc = weights.sum(axis=-1)
  distance = (weights * t_mids).sum(axis=-1) / acc
  distance = jnp.clip(
      jnp.nan_to_num(distance, jnp.inf), t_vals[:, 0], t_vals[:, -1])
  if white_bkgd:
    comp_rgb = comp_rgb + (1. - acc[..., None])
  return comp_rgb, distance, acc, weights


def sample_along_rays(key, origins, directions, radii, num_samples, near, far,
                      randomized, lindisp, ray_shape):
  """Stratified sampling along the rays.

  Args:
    key: jnp.ndarray, random generator key.
    origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
    directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
    radii: jnp.ndarray(float32), [batch_size, 3], ray radii.
    num_samples: int.
    near: jnp.ndarray, [batch_size, 1], near clip.
    far: jnp.ndarray, [batch_size, 1], far clip.
    randomized: bool, use randomized stratified sampling.
    lindisp: bool, sampling linearly in disparity rather than depth.
    ray_shape: string, which shape ray to assume.

  Returns:
    t_vals: jnp.ndarray, [batch_size, num_samples], sampled z values.
    means: jnp.ndarray, [batch_size, num_samples, 3], sampled means.
    covs: jnp.ndarray, [batch_size, num_samples, 3, 3], sampled covariances.
  """
  batch_size = origins.shape[0]

  t_vals = jnp.linspace(0., 1., num_samples + 1)
  if lindisp:
    t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
  else:
    t_vals = near * (1. - t_vals) + far * t_vals

  if randomized:
    mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    upper = jnp.concatenate([mids, t_vals[..., -1:]], -1)
    lower = jnp.concatenate([t_vals[..., :1], mids], -1)
    t_rand = random.uniform(key, [batch_size, num_samples + 1])
    t_vals = lower + (upper - lower) * t_rand
  else:
    # Broadcast t_vals to make the returned shape consistent.
    t_vals = jnp.broadcast_to(t_vals, [batch_size, num_samples + 1])
  means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
  return t_vals, (means, covs)


def resample_along_rays(key, origins, directions, radii, t_vals, weights,
                        randomized, ray_shape, stop_grad, resample_padding):
  """Resampling.

  Args:
    key: jnp.ndarray(float32), [2,], random number generator.
    origins: jnp.ndarray(float32), [batch_size, 3], ray origins.
    directions: jnp.ndarray(float32), [batch_size, 3], ray directions.
    radii: jnp.ndarray(float32), [batch_size, 3], ray radii.
    t_vals: jnp.ndarray(float32), [batch_size, num_samples+1].
    weights: jnp.array(float32), weights for t_vals
    randomized: bool, use randomized samples.
    ray_shape: string, which kind of shape to assume for the ray.
    stop_grad: bool, whether or not to backprop through sampling.
    resample_padding: float, added to the weights before normalizing.

  Returns:
    t_vals: jnp.ndarray(float32), [batch_size, num_samples+1].
    points: jnp.ndarray(float32), [batch_size, num_samples, 3].
  """
  # Do a blurpool.
  weights_pad = jnp.concatenate([
      weights[..., :1],
      weights,
      weights[..., -1:],
  ],
                                axis=-1)
  weights_max = jnp.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
  weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

  # Add in a constant (the sampling function will renormalize the PDF).
  weights = weights_blur + resample_padding

  new_t_vals = math.sorted_piecewise_constant_pdf(
      key,
      t_vals,
      weights,
      t_vals.shape[-1],
      randomized,
  )
  if stop_grad:
    new_t_vals = lax.stop_gradient(new_t_vals)
  means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
  return new_t_vals, (means, covs)
