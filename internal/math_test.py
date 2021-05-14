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
"""Unit tests for math."""
import functools

from absl.testing import absltest
import jax
from jax import random
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as np
import scipy as sp
import tensorflow as tf

from internal import math


def safe_trig_harness(fn, max_exp):
  x = 10**np.linspace(-30, max_exp, 10000)
  x = np.concatenate([-x[::-1], np.array([0]), x])
  y_true = getattr(np, fn)(x)
  y = getattr(math, 'safe_' + fn)(x)
  return y_true, y


class MathUtilsTest(jtu.JaxTestCase):

  def test_sin(self):
    """In [-1e10, 1e10] safe_sin and safe_cos are accurate."""
    for fn in ['sin', 'cos']:
      y_true, y = safe_trig_harness(fn, 10)
      self.assertLess(np.max(np.abs(y - y_true)), 1e-4)
      self.assertFalse(jnp.any(jnp.isnan(y)))
    # Beyond that range it's less accurate but we just don't want it to be NaN.
    for fn in ['sin', 'cos']:
      y_true, y = safe_trig_harness(fn, 60)
      self.assertFalse(jnp.any(jnp.isnan(y)))

  def test_psnr_round_trip(self):
    """MSE -> PSNR -> MSE is a no-op."""
    mse = 0.07
    self.assertAllClose(math.psnr_to_mse(math.mse_to_psnr(mse)), mse)

  def test_learning_rate_decay(self):
    np.random.seed(0)
    for _ in range(10):
      lr_init = np.exp(np.random.normal() - 3)
      lr_final = lr_init * np.exp(np.random.normal() - 5)
      max_steps = int(np.ceil(100 + 100 * np.exp(np.random.normal())))

      lr_fn = functools.partial(
          math.learning_rate_decay,
          lr_init=lr_init,
          lr_final=lr_final,
          max_steps=max_steps)

      # Test that the rate at the beginning is the initial rate.
      self.assertAllClose(lr_fn(0), lr_init)

      # Test that the rate at the end is the final rate.
      self.assertAllClose(lr_fn(max_steps), lr_final)

      # Test that the rate at the middle is the geometric mean of the two rates.
      self.assertAllClose(lr_fn(max_steps / 2), np.sqrt(lr_init * lr_final))

      # Test that the rate past the end is the final rate
      self.assertAllClose(lr_fn(max_steps + 100), lr_final)

  def test_delayed_learning_rate_decay(self):
    np.random.seed(0)
    for _ in range(10):
      lr_init = np.exp(np.random.normal() - 3)
      lr_final = lr_init * np.exp(np.random.normal() - 5)
      max_steps = int(np.ceil(100 + 100 * np.exp(np.random.normal())))
      lr_delay_steps = int(np.random.uniform(low=0.1, high=0.4) * max_steps)
      lr_delay_mult = np.exp(np.random.normal() - 3)

      lr_fn = functools.partial(
          math.learning_rate_decay,
          lr_init=lr_init,
          lr_final=lr_final,
          max_steps=max_steps,
          lr_delay_steps=lr_delay_steps,
          lr_delay_mult=lr_delay_mult)

      # Test that the rate at the beginning is the delayed initial rate.
      self.assertAllClose(lr_fn(0), lr_delay_mult * lr_init)

      # Test that the rate at the end is the final rate.
      self.assertAllClose(lr_fn(max_steps), lr_final)

      # Test that the rate at after the delay is over is the usual rate.
      self.assertAllClose(
          lr_fn(lr_delay_steps),
          math.learning_rate_decay(lr_delay_steps, lr_init, lr_final,
                                   max_steps))

      # Test that the rate at the middle is the geometric mean of the two rates.
      self.assertAllClose(lr_fn(max_steps / 2), np.sqrt(lr_init * lr_final))

      # Test that the rate past the end is the final rate
      self.assertAllClose(lr_fn(max_steps + 100), lr_final)

  def test_ssim_golden(self):
    """Test our SSIM implementation against the Tensorflow version."""
    rng = random.PRNGKey(0)
    shape = (2, 12, 12, 3)
    for _ in range(4):
      rng, key = random.split(rng)
      max_val = random.uniform(key, minval=0.1, maxval=3.)
      rng, key = random.split(rng)
      img0 = max_val * random.uniform(key, shape=shape, minval=-1, maxval=1)
      rng, key = random.split(rng)
      img1 = max_val * random.uniform(key, shape=shape, minval=-1, maxval=1)
      rng, key = random.split(rng)
      filter_size = random.randint(key, shape=(), minval=1, maxval=10)
      rng, key = random.split(rng)
      filter_sigma = random.uniform(key, shape=(), minval=0.1, maxval=10.)
      rng, key = random.split(rng)
      k1 = random.uniform(key, shape=(), minval=0.001, maxval=0.1)
      rng, key = random.split(rng)
      k2 = random.uniform(key, shape=(), minval=0.001, maxval=0.1)

      ssim_gt = tf.image.ssim(
          img0,
          img1,
          max_val,
          filter_size=filter_size,
          filter_sigma=filter_sigma,
          k1=k1,
          k2=k2).numpy()
      for return_map in [False, True]:
        ssim_fn = jax.jit(
            functools.partial(
                math.compute_ssim,
                max_val=max_val,
                filter_size=filter_size,
                filter_sigma=filter_sigma,
                k1=k1,
                k2=k2,
                return_map=return_map))
        ssim = ssim_fn(img0, img1)
        if not return_map:
          self.assertAllClose(ssim, ssim_gt)
        else:
          self.assertAllClose(np.mean(ssim, [1, 2, 3]), ssim_gt)
        self.assertLessEqual(np.max(ssim), 1.)
        self.assertGreaterEqual(np.min(ssim), -1.)

  def test_ssim_lowerbound(self):
    """Test the unusual corner case where SSIM is -1."""
    sz = 11
    img = np.meshgrid(*([np.linspace(-1, 1, sz)] * 2))[0][None, ..., None]
    eps = 1e-5
    ssim = math.compute_ssim(
        img, -img, 1., filter_size=sz, filter_sigma=1.5, k1=eps, k2=eps)
    self.assertAllClose(ssim, -np.ones_like(ssim))

  def test_srgb_linearize(self):
    x = np.linspace(-1, 3, 10000)  # Nobody should call this <0 but it works.
    # Check that the round-trip transformation is a no-op.
    self.assertAllClose(math.linear_to_srgb(math.srgb_to_linear(x)), x)
    self.assertAllClose(math.srgb_to_linear(math.linear_to_srgb(x)), x)
    # Check that gradients are finite.
    self.assertTrue(
        np.all(np.isfinite(jax.vmap(jax.grad(math.linear_to_srgb))(x))))
    self.assertTrue(
        np.all(np.isfinite(jax.vmap(jax.grad(math.srgb_to_linear))(x))))

  def test_sorted_piecewise_constant_pdf_train_mode(self):
    """Test that piecewise-constant sampling reproduces its distribution."""
    batch_size = 4
    num_bins = 16
    num_samples = 1000000
    precision = 1e5
    rng = random.PRNGKey(20202020)

    # Generate a series of random PDFs to sample from.
    data = []
    for _ in range(batch_size):
      rng, key = random.split(rng)
      # Randomly initialize the distances between bins.
      # We're rolling our own fixed precision here to make cumsum exact.
      bins_delta = jnp.round(precision * jnp.exp(
          random.uniform(key, shape=(num_bins + 1,), minval=-3, maxval=3)))

      # Set some of the bin distances to 0.
      rng, key = random.split(rng)
      bins_delta *= random.uniform(key, shape=bins_delta.shape) < 0.9

      # Integrate the bins.
      bins = jnp.cumsum(bins_delta) / precision
      rng, key = random.split(rng)
      bins += random.normal(key) * num_bins / 2
      rng, key = random.split(rng)

      # Randomly generate weights, allowing some to be zero.
      weights = jnp.maximum(
          0, random.uniform(key, shape=(num_bins,), minval=-0.5, maxval=1.))
      gt_hist = weights / weights.sum()
      data.append((bins, weights, gt_hist))

    # Tack on an "all zeros" weight matrix, which is a common cause of NaNs.
    weights = jnp.zeros_like(weights)
    gt_hist = jnp.ones_like(gt_hist) / num_bins
    data.append((bins, weights, gt_hist))

    bins, weights, gt_hist = [jnp.stack(x) for x in zip(*data)]

    for randomized in [True, False]:
      rng, key = random.split(rng)
      # Draw samples from the batch of PDFs.
      samples = math.sorted_piecewise_constant_pdf(
          key,
          bins,
          weights,
          num_samples,
          randomized,
      )
      self.assertEqual(samples.shape[-1], num_samples)

      # Check that samples are sorted.
      self.assertTrue(jnp.all(samples[..., 1:] >= samples[..., :-1]))

      # Verify that each set of samples resembles the target distribution.
      for i_samples, i_bins, i_gt_hist in zip(samples, bins, gt_hist):
        i_hist = jnp.float32(jnp.histogram(i_samples, i_bins)[0]) / num_samples
        i_gt_hist = jnp.array(i_gt_hist)

        # Merge any of the zero-span bins until there aren't any left.
        while jnp.any(i_bins[:-1] == i_bins[1:]):
          j = int(jnp.where(i_bins[:-1] == i_bins[1:])[0][0])
          i_hist = jnp.concatenate([
              i_hist[:j],
              jnp.array([i_hist[j] + i_hist[j + 1]]), i_hist[j + 2:]
          ])
          i_gt_hist = jnp.concatenate([
              i_gt_hist[:j],
              jnp.array([i_gt_hist[j] + i_gt_hist[j + 1]]), i_gt_hist[j + 2:]
          ])
          i_bins = jnp.concatenate([i_bins[:j], i_bins[j + 1:]])

        # Angle between the two histograms in degrees.
        angle = 180 / jnp.pi * jnp.arccos(
            jnp.minimum(
                1.,
                jnp.mean(
                    (i_hist * i_gt_hist) /
                    jnp.sqrt(jnp.mean(i_hist**2) * jnp.mean(i_gt_hist**2)))))
        # Jensen-Shannon divergence.
        m = (i_hist + i_gt_hist) / 2
        js_div = jnp.sum(
            sp.special.kl_div(i_hist, m) + sp.special.kl_div(i_gt_hist, m)) / 2
        self.assertLessEqual(angle, 0.5)
        self.assertLessEqual(js_div, 1e-5)

  def test_sorted_piecewise_constant_pdf_large_flat(self):
    """Test sampling when given a large flat distribution."""
    num_samples = 100
    num_bins = 100000
    key = random.PRNGKey(0)
    bins = jnp.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    samples = math.sorted_piecewise_constant_pdf(
        key,
        bins[None],
        weights[None],
        num_samples,
        True,
    )[0]
    # All samples should be within the range of the bins.
    self.assertTrue(jnp.all(samples >= bins[0]))
    self.assertTrue(jnp.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = jnp.mod(samples, 1)
    self.assertLessEqual(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic, 0.2)

    # All samples should collectively resemble a uniform distribution.
    self.assertLessEqual(
        sp.stats.kstest(samples, 'uniform', (bins[0], bins[-1])).statistic, 0.2)

  def test_sorted_piecewise_constant_pdf_sparse_delta(self):
    """Test sampling when given a large distribution with a big delta in it."""
    num_samples = 100
    num_bins = 100000
    key = random.PRNGKey(0)
    bins = jnp.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    delta_idx = len(weights) // 2
    weights[delta_idx] = len(weights) - 1
    samples = math.sorted_piecewise_constant_pdf(
        key,
        bins[None],
        weights[None],
        num_samples,
        True,
    )[0]

    # All samples should be within the range of the bins.
    self.assertTrue(jnp.all(samples >= bins[0]))
    self.assertTrue(jnp.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = jnp.mod(samples, 1)
    self.assertLessEqual(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic, 0.2)

    # The delta function bin should contain ~half of the samples.
    in_delta = (samples >= bins[delta_idx]) & (samples <= bins[delta_idx + 1])
    self.assertAllClose(jnp.mean(in_delta), 0.5, atol=0.05)

  def test_sorted_piecewise_constant_pdf_single_bin(self):
    """Test sampling when given a small `one hot' distribution."""
    num_samples = 625
    key = random.PRNGKey(0)
    bins = jnp.array([0, 1, 3, 6, 10], jnp.float32)
    for randomized in [False, True]:
      for i in range(len(bins) - 1):
        weights = np.zeros(len(bins) - 1, jnp.float32)
        weights[i] = 1.
        samples = math.sorted_piecewise_constant_pdf(
            key,
            bins[None],
            weights[None],
            num_samples,
            randomized,
        )[0]

        # All samples should be within [bins[i], bins[i+1]].
        self.assertTrue(jnp.all(samples >= bins[i]))
        self.assertTrue(jnp.all(samples <= bins[i + 1]))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
