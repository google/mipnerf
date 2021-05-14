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
"""Unit tests for mip."""
from absl.testing import absltest
import jax
from jax import random
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as np

from internal import math
from internal import mip


def surface_stats(points):
  """Get the sample mean and covariance matrix of a set of matrices [..., d]."""
  means = jnp.mean(points, -1)
  centered = points - means[..., None]
  covs = jnp.mean(centered[..., None, :, :] * centered[..., :, None, :], -1)
  return means, covs


def sqrtm(mat):
  """Take the matrix square root of a PSD matrix [..., d, d]."""
  eigval, eigvec = jax.scipy.linalg.eigh(mat)
  scaling = jnp.sqrt(jnp.maximum(0., eigval))[..., None, :]
  return math.matmul(eigvec * scaling, jnp.moveaxis(eigvec, -2, -1))


def control_points(mean, cov):
  """Construct "sigma points" using a matrix sqrt (Cholesky or SVD are fine)."""
  sqrtm_cov = sqrtm(cov)  # or could be jax.scipy.linalg.cholesky(cov)
  offsets = jnp.sqrt(mean.shape[-1] + 0.5) * jnp.concatenate(
      [jnp.zeros_like(mean[..., None]), sqrtm_cov, -sqrtm_cov], -1)
  return mean[..., None] + offsets


def inside_conical_frustum(x, d, t0, t1, r, ttol=1e-6, rtol=1e-6):
  """Test if `x` is inside the conical frustum specified by the other inputs."""
  d_normsq = jnp.sum(d**2)
  d_norm = jnp.sqrt(d_normsq)
  x_normsq = jnp.sum(x**2, -1)
  x_norm = jnp.sqrt(x_normsq)
  xd = math.matmul(x, d)
  is_inside = (
      (t0 - ttol) <= xd / d_normsq) & (xd / d_normsq <= (t1 + ttol)) & (
          (xd / (d_norm * x_norm)) >=
          (1 / jnp.sqrt(1 + r**2 / d_normsq) - rtol))
  return is_inside


def stable_pos_enc(x, n):
  """A stable posenc for very high degrees, courtesy of Sameer Agrawal."""
  sin_x = np.sin(x)
  cos_x = np.cos(x)
  output = []
  rotmat = np.array([[cos_x, -sin_x], [sin_x, cos_x]], dtype='double')
  for _ in range(n):
    output.append(rotmat[::-1, 0, :])
    rotmat = np.einsum('ijn,jkn->ikn', rotmat, rotmat)
  return np.reshape(np.transpose(np.stack(output, 0), [2, 1, 0]), [-1, 2 * n])


def sample_conical_frustum(rng, num_samples, d, t0, t1, base_radius):
  """Draw random samples from a conical frustum.

  Args:
    rng: The RNG seed.
    num_samples: int, the number of samples to draw.
    d: jnp.float32 3-vector, the axis of the cone.
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.

  Returns:
    A matrix of samples.
  """
  key, rng = random.split(rng)
  u = random.uniform(key, shape=[num_samples])
  t = (t0**3 * (1 - u) + t1**3 * u)**(1 / 3)
  key, rng = random.split(rng)
  theta = random.uniform(key, shape=[num_samples], minval=0, maxval=jnp.pi * 2)
  key, rng = random.split(rng)
  r = base_radius * t * jnp.sqrt(random.uniform(key, shape=[num_samples]))

  d_norm = d / jnp.linalg.norm(d)
  null = jnp.eye(3) - d_norm[:, None] * d_norm[None, :]
  basis = jnp.linalg.svd(null)[0][:, :2]
  rot_samples = ((basis[:, 0:1] * r * jnp.cos(theta)) +
                 (basis[:, 1:2] * r * jnp.sin(theta)) + d[:, None] * t).T
  return rot_samples


def generate_random_cylinder(rng, num_zs=4):
  t0, t1 = [], []
  for _ in range(num_zs):
    rng, key = random.split(rng)
    z_mean = random.uniform(key, minval=1.5, maxval=3)
    rng, key = random.split(rng)
    z_delta = random.uniform(key, minval=0.1, maxval=.3)
    t0.append(z_mean - z_delta)
    t1.append(z_mean + z_delta)
  t0 = jnp.array(t0)
  t1 = jnp.array(t1)

  rng, key = random.split(rng)
  radius = random.uniform(key, minval=0.1, maxval=.2)

  rng, key = random.split(rng)
  raydir = random.normal(key, [3])
  raydir = raydir / jnp.sqrt(jnp.sum(raydir**2, -1))

  rng, key = random.split(rng)
  scale = random.uniform(key, minval=0.4, maxval=1.2)
  raydir = scale * raydir

  return raydir, t0, t1, radius


def generate_random_conical_frustum(rng, num_zs=4):
  t0, t1 = [], []
  for _ in range(num_zs):
    rng, key = random.split(rng)
    z_mean = random.uniform(key, minval=1.5, maxval=3)
    rng, key = random.split(rng)
    z_delta = random.uniform(key, minval=0.1, maxval=.3)
    t0.append(z_mean - z_delta)
    t1.append(z_mean + z_delta)
  t0 = jnp.array(t0)
  t1 = jnp.array(t1)

  rng, key = random.split(rng)
  r = random.uniform(key, minval=0.01, maxval=.05)

  rng, key = random.split(rng)
  raydir = random.normal(key, [3])
  raydir = raydir / jnp.sqrt(jnp.sum(raydir**2, -1))

  rng, key = random.split(rng)
  scale = random.uniform(key, minval=0.8, maxval=1.2)
  raydir = scale * raydir

  return raydir, t0, t1, r


def cylinder_to_gaussian_sample(key,
                                raydir,
                                t0,
                                t1,
                                radius,
                                padding=1,
                                num_samples=1000000):
  # Sample uniformly from a cube that surrounds the entire conical frustom.
  z_max = max(t0, t1)
  samples = random.uniform(
      key, [num_samples, 3],
      minval=jnp.min(raydir) * z_max - padding,
      maxval=jnp.max(raydir) * z_max + padding)

  # Grab only the points within the cylinder.
  raydir_magsq = jnp.sum(raydir**2, -1, keepdims=True)
  proj = (raydir * (samples @ raydir)[:, None]) / raydir_magsq
  dist = samples @ raydir
  mask = (dist >= raydir_magsq * t0) & (dist <= raydir_magsq * t1) & (
      jnp.sum((proj - samples)**2, -1) < radius**2)
  samples = samples[mask, :]

  # Compute their mean and covariance.
  mean = jnp.mean(samples, 0)
  cov = jnp.cov(samples.T, bias=False)
  return mean, cov


def conical_frustum_to_gaussian_sample(key, raydir, t0, t1, r):
  """A brute-force numerical approximation to conical_frustum_to_gaussian()."""
  # Sample uniformly from a cube that surrounds the entire conical frustum.
  samples = sample_conical_frustum(key, 100000, raydir, t0, t1, r)
  # Compute their mean and covariance.
  return surface_stats(samples.T)


class MipUtilsTest(jtu.JaxTestCase):

  def test_posenc(self):
    n = 10
    x = np.linspace(-1, 1, 100)
    z = mip.pos_enc(x[:, None], 0, n, append_identity=False)
    z_stable = stable_pos_enc(x, n)
    self.assertLess(np.max(np.abs(z - z_stable)), 1e-4)

  def test_cylinder_scaling(self):
    d = jnp.array([0., 0., 1.])
    t0 = jnp.array([0.3])
    t1 = jnp.array([0.7])
    radius = jnp.array([0.4])
    mean, cov = mip.cylinder_to_gaussian(
        d,
        t0,
        t1,
        radius,
        False,
    )
    scale = 2.7
    scaled_mean, scaled_cov = mip.cylinder_to_gaussian(
        scale * d,
        t0,
        t1,
        radius,
        False,
    )
    self.assertAllClose(scale * mean, scaled_mean)
    self.assertAllClose(scale**2 * cov[2, 2], scaled_cov[2, 2])
    control = control_points(mean, cov)[0]
    control_scaled = control_points(scaled_mean, scaled_cov)[0]
    self.assertAllClose(control[:2, :], control_scaled[:2, :])
    self.assertAllClose(control[2, :] * scale, control_scaled[2, :])

  def test_conical_frustum_scaling(self):
    d = jnp.array([0., 0., 1.])
    t0 = jnp.array([0.3])
    t1 = jnp.array([0.7])
    radius = jnp.array([0.4])
    mean, cov = mip.conical_frustum_to_gaussian(
        d,
        t0,
        t1,
        radius,
        False,
    )
    scale = 2.7
    scaled_mean, scaled_cov = mip.conical_frustum_to_gaussian(
        scale * d,
        t0,
        t1,
        radius,
        False,
    )
    self.assertAllClose(scale * mean, scaled_mean)
    self.assertAllClose(scale**2 * cov[2, 2], scaled_cov[2, 2])
    control = control_points(mean, cov)[0]
    control_scaled = control_points(scaled_mean, scaled_cov)[0]
    self.assertAllClose(control[:2, :], control_scaled[:2, :])
    self.assertAllClose(control[2, :] * scale, control_scaled[2, :])

  def test_expected_sin(self):
    normal_samples = random.normal(random.PRNGKey(0), (10000,))
    for mu, var in [(0, 1), (1, 3), (-2, .2), (10, 10)]:
      sin_mu, sin_var = mip.expected_sin(mu, var)
      x = jnp.sin(jnp.sqrt(var) * normal_samples + mu)
      self.assertAllClose(sin_mu, jnp.mean(x), atol=1e-2)
      self.assertAllClose(sin_var, jnp.var(x), atol=1e-2)

  def test_control_points(self):
    rng = random.PRNGKey(0)
    batch_size = 10
    for num_dims in [1, 2, 3]:
      key, rng = random.split(rng)
      mean = jax.random.normal(key, [batch_size, num_dims])
      key, rng = random.split(rng)
      half_cov = jax.random.normal(key, [batch_size] + [num_dims] * 2)
      cov = half_cov @ jnp.moveaxis(half_cov, -1, -2)

      sqrtm_cov = sqrtm(cov)
      self.assertArraysAllClose(sqrtm_cov @ sqrtm_cov, cov, atol=1e-5)

      points = control_points(mean, cov)
      mean_recon, cov_recon = surface_stats(points)
      self.assertArraysAllClose(mean, mean_recon)
      self.assertArraysAllClose(cov, cov_recon, atol=1e-5)

  def test_conical_frustum(self):
    rng = random.PRNGKey(0)
    data = []
    for _ in range(10):
      key, rng = random.split(rng)
      raydir, t0, t1, r = generate_random_conical_frustum(key)
      i_results = []
      for i_t0, i_t1 in zip(t0, t1):
        key, rng = random.split(rng)
        i_results.append(
            conical_frustum_to_gaussian_sample(key, raydir, i_t0, i_t1, r))
      mean_gt, cov_gt = [jnp.stack(x, 0) for x in zip(*i_results)]
      data.append((raydir, t0, t1, r, mean_gt, cov_gt))
    raydir, t0, t1, r, mean_gt, cov_gt = [jnp.stack(x, 0) for x in zip(*data)]
    diag_cov_gt = jax.vmap(jax.vmap(jnp.diag))(cov_gt)
    for diag in [False, True]:
      for stable in [False, True]:
        mean, cov = mip.conical_frustum_to_gaussian(
            raydir, t0, t1, r[..., None], diag, stable=stable)
        self.assertAllClose(mean, mean_gt, atol=0.001)
        if diag:
          self.assertAllClose(cov, diag_cov_gt, atol=0.0002)
        else:
          self.assertAllClose(cov, cov_gt, atol=0.0002)

  def test_inside_conical_frustum(self):
    """This test only tests helper functions used by other tests."""
    rng = random.PRNGKey(0)
    for _ in range(20):
      key, rng = random.split(rng)
      d, t0, t1, r = generate_random_conical_frustum(key, num_zs=1)
      key, rng = random.split(rng)
      # Sample some points.
      samples = sample_conical_frustum(key, 1000000, d, t0, t1, r)
      # Check that they're all inside.
      check = lambda x: inside_conical_frustum(x, d, t0, t1, r)
      self.assertTrue(jnp.all(check(samples)))
      # Check that wiggling them a little puts some outside (potentially flaky).
      self.assertFalse(jnp.all(check(samples + 1e-3)))
      self.assertFalse(jnp.all(check(samples - 1e-3)))

  def test_conical_frustum_stable(self):
    rng = random.PRNGKey(0)
    for _ in range(10):
      key, rng = random.split(rng)
      d, t0, t1, r = generate_random_conical_frustum(key)
      for diag in [False, True]:
        mean, cov = mip.conical_frustum_to_gaussian(
            d, t0, t1, r, diag, stable=False)
        mean_stable, cov_stable = mip.conical_frustum_to_gaussian(
            d, t0, t1, r, diag, stable=True)
        self.assertAllClose(mean, mean_stable, atol=1e-7)
        self.assertAllClose(cov, cov_stable, atol=1e-5)

  def test_cylinder(self):
    rng = random.PRNGKey(0)
    data = []
    for _ in range(10):
      key, rng = random.split(rng)
      raydir, t0, t1, radius = generate_random_cylinder(rng)
      key, rng = random.split(rng)
      i_results = []
      for i_t0, i_t1 in zip(t0, t1):
        i_results.append(
            cylinder_to_gaussian_sample(key, raydir, i_t0, i_t1, radius))
      mean_gt, cov_gt = [jnp.stack(x, 0) for x in zip(*i_results)]
      data.append((raydir, t0, t1, radius, mean_gt, cov_gt))
    raydir, t0, t1, radius, mean_gt, cov_gt = [
        jnp.stack(x, 0) for x in zip(*data)
    ]
    mean, cov = mip.cylinder_to_gaussian(raydir, t0, t1, radius[..., None],
                                         False)
    self.assertAllClose(mean, mean_gt, atol=0.1)
    self.assertAllClose(cov, cov_gt, atol=0.01)

  def test_integrated_pos_enc(self):
    num_dims = 2  # The number of input dimensions.
    min_deg = 0
    max_deg = 4
    num_samples = 100000
    rng = random.PRNGKey(0)
    for _ in range(5):
      # Generate a coordinate's mean and covariance matrix.
      key, rng = random.split(rng)
      mean = random.normal(key, (2,))
      key, rng = random.split(rng)
      half_cov = jax.random.normal(key, [num_dims] * 2)
      cov = half_cov @ half_cov.T
      for diag in [False, True]:
        # Generate an IPE.
        enc = mip.integrated_pos_enc(
            (mean, jnp.diag(cov) if diag else cov),
            min_deg,
            max_deg,
            diag,
        )

        # Draw samples, encode them, and take their mean.
        key, rng = random.split(rng)
        samples = random.multivariate_normal(key, mean, cov, [num_samples])
        enc_samples = mip.pos_enc(
            samples, min_deg, max_deg, append_identity=False)
        enc_gt = jnp.mean(enc_samples, 0)
        self.assertAllClose(enc, enc_gt, rtol=1e-2, atol=1e-2)

  def test_lift_gaussian_diag(self):
    dims, n, m = 3, 10, 4
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    d = random.normal(key, [n, dims])
    key, rng = random.split(rng)
    z_mean = random.normal(key, [n, m])
    key, rng = random.split(rng)
    z_var = jnp.exp(random.normal(key, [n, m]))
    key, rng = random.split(rng)
    xy_var = jnp.exp(random.normal(key, [n, m]))
    mean, cov = mip.lift_gaussian(d, z_mean, z_var, xy_var, diag=False)
    mean_diag, cov_diag = mip.lift_gaussian(d, z_mean, z_var, xy_var, diag=True)
    self.assertAllClose(mean, mean_diag)
    self.assertAllClose(jax.vmap(jax.vmap(jnp.diag))(cov), cov_diag)

  def test_rotated_conic_frustums(self):
    # Test that conic frustum Gaussians are closed under rotation.
    diag = False
    rng = random.PRNGKey(0)
    for _ in range(10):
      rng, key = random.split(rng)
      z_mean = random.uniform(key, minval=1.5, maxval=3)
      rng, key = random.split(rng)
      z_delta = random.uniform(key, minval=0.1, maxval=.3)
      t0 = jnp.array(z_mean - z_delta)
      t1 = jnp.array(z_mean + z_delta)

      rng, key = random.split(rng)
      r = random.uniform(key, minval=0.1, maxval=.2)

      rng, key = random.split(rng)
      d = random.normal(key, [3])

      mean, cov = mip.conical_frustum_to_gaussian(d, t0, t1, r, diag)

      # Make a random rotation matrix.
      rng, key = random.split(rng)
      x = random.normal(key, [10, 3])
      rot_mat = x.T @ x
      u, _, v = jnp.linalg.svd(rot_mat)
      rot_mat = u @ v.T

      mean, cov = mip.conical_frustum_to_gaussian(d, t0, t1, r, diag)
      rot_mean, rot_cov = mip.conical_frustum_to_gaussian(
          rot_mat @ d, t0, t1, r, diag)
      gt_rot_mean, gt_rot_cov = surface_stats(
          rot_mat @ control_points(mean, cov))

      self.assertAllClose(rot_mean, gt_rot_mean)
      self.assertAllClose(rot_cov, gt_rot_cov)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
