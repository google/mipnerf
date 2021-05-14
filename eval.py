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
"""Evaluation script for mip-NeRF."""
import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np

from internal import datasets
from internal import math
from internal import models
from internal import utils
from internal import vis

FLAGS = flags.FLAGS
utils.define_common_flags()
flags.DEFINE_bool(
    'eval_once', True,
    'If True, evaluate the model only once, otherwise keeping evaluating new'
    'checkpoints if any exist.')
flags.DEFINE_bool('save_output', True,
                  'If True, save predicted images to disk.')


def main(unused_argv):
  config = utils.load_config()

  dataset = datasets.get_dataset('test', FLAGS.data_dir, config)
  model, init_variables = models.construct_mipnerf(
      random.PRNGKey(20200823), dataset.peek())
  optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
  state = utils.TrainState(optimizer=optimizer)
  del optimizer, init_variables

  # Rendering is forced to be deterministic even if training was randomized, as
  # this eliminates 'speckle' artifacts.
  def render_eval_fn(variables, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            random.PRNGKey(0),  # Unused.
            rays,
            randomized=False,
            white_bkgd=config.white_bkgd),
        axis_name='batch')

  # pmap over only the data input.
  render_eval_pfn = jax.pmap(
      render_eval_fn,
      in_axes=(None, None, 0),
      donate_argnums=2,
      axis_name='batch',
  )

  ssim_fn = jax.jit(functools.partial(math.compute_ssim, max_val=1.))

  last_step = 0
  out_dir = path.join(FLAGS.train_dir,
                      'path_renders' if config.render_path else 'test_preds')
  if not FLAGS.eval_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(FLAGS.train_dir, 'eval'))
  while True:
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    step = int(state.optimizer.state.step)
    if step <= last_step:
      continue
    if FLAGS.save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)
    psnr_values = []
    ssim_values = []
    avg_values = []
    if not FLAGS.eval_once:
      showcase_index = random.randint(random.PRNGKey(step), (), 0, dataset.size)
    for idx in range(dataset.size):
      print(f'Evaluating {idx+1}/{dataset.size}')
      batch = next(dataset)
      pred_color, pred_distance, pred_acc = models.render_image(
          functools.partial(render_eval_pfn, state.optimizer.target),
          batch['rays'],
          None,
          chunk=FLAGS.chunk)

      vis_suite = vis.visualize_suite(pred_distance, pred_acc)

      if jax.host_id() != 0:  # Only record via host 0.
        continue
      if not FLAGS.eval_once and idx == showcase_index:
        showcase_color = pred_color
        showcase_acc = pred_acc
        showcase_vis_suite = vis_suite
        if not config.render_path:
          showcase_gt = batch['pixels']
      if not config.render_path:
        psnr = float(
            math.mse_to_psnr(((pred_color - batch['pixels'])**2).mean()))
        ssim = float(ssim_fn(pred_color, batch['pixels']))
        print(f'PSNR={psnr:.4f} SSIM={ssim:.4f}')
        psnr_values.append(psnr)
        ssim_values.append(ssim)
      if FLAGS.save_output and (config.test_render_interval > 0):
        if (idx % config.test_render_interval) == 0:
          utils.save_img_uint8(
              pred_color, path.join(out_dir, 'color_{:03d}.png'.format(idx)))
          utils.save_img_float32(
              pred_distance,
              path.join(out_dir, 'distance_{:03d}.tiff'.format(idx)))
          utils.save_img_float32(
              pred_acc, path.join(out_dir, 'acc_{:03d}.tiff'.format(idx)))
          for k, v in vis_suite.items():
            utils.save_img_uint8(
                v, path.join(out_dir, k + '_{:03d}.png'.format(idx)))
    if (not FLAGS.eval_once) and (jax.host_id() == 0):
      summary_writer.image('pred_color', showcase_color, step)
      summary_writer.image('pred_acc', showcase_acc, step)
      for k, v in showcase_vis_suite.items():
        summary_writer.image('pred_' + k, v, step)
      if not config.render_path:
        summary_writer.scalar('psnr', np.mean(np.array(psnr_values)), step)
        summary_writer.scalar('ssim', np.mean(np.array(ssim_values)), step)
        summary_writer.image('target', showcase_gt, step)
    if FLAGS.save_output and (not config.render_path) and (jax.host_id() == 0):
      with utils.open_file(path.join(out_dir, f'psnrs_{step}.txt'), 'w') as f:
        f.write(' '.join([str(v) for v in psnr_values]))
      with utils.open_file(path.join(out_dir, f'ssims_{step}.txt'), 'w') as f:
        f.write(' '.join([str(v) for v in ssim_values]))
    if FLAGS.eval_once:
      break
    if int(step) >= config.max_steps:
      break
    last_step = step


if __name__ == '__main__':
  app.run(main)
