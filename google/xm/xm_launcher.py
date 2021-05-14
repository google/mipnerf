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
"""Launcher script for train/test jobs."""
import os

from absl import app
from absl import flags

from google3.learning.deepmind.python.adhoc_import import binary_import
import google3.learning.deepmind.xmanager2.client.google as xm
# pylint: disable=g-import-not-at-top
with binary_import.AutoGoogle3():
  from google3.learning.brain.research.jax.xmanager import jax_xm

FLAGS = flags.FLAGS
flags.DEFINE_string('cell', 'el', 'Cell on which to launch the jobs.')
flags.DEFINE_string('cell_eval', 'lu',
                    'Cell on which to launch the evaluation jobs.')
flags.DEFINE_integer('priority', 200, 'Priority to which launch the job.')
flags.DEFINE_integer('n_gpus', 1, 'Number of gpus per train worker.')
flags.DEFINE_integer('n_gpus_eval', 1, 'Number of gpus per eval worker.')
flags.DEFINE_string(
    'train_dir',
    None,  # Like: '/cns/{cell}-d/home/barron/data/',
    'Experiment path.')
flags.mark_flag_as_required('train_dir')
flags.DEFINE_string(
    'data_dir',
    None,  # Like: '/cns/{cell}-d/home/barron/data/',
    'Data path.')
flags.mark_flag_as_required('data_dir')
flags.DEFINE_string('gin_file', None, 'Gin config file.')
flags.DEFINE_bool('is_train', True, 'The job is in the training mode.')
flags.DEFINE_bool('use_tpu', True, 'Whether to use tpu for training.')
flags.DEFINE_enum('tpu_topology', '2x2', ['1x1', '2x2', '4x2', '4x4', '8x8'],
                  'TPU topology for training.')
flags.DEFINE_enum('tpu_platform', 'jf', ['jf', 'df'],
                  'TPU platform for training.')
flags.DEFINE_bool('use_tpu_eval', False, 'Whether to use tpu for evaluation.')
flags.DEFINE_enum('tpu_topology_eval', '2x2',
                  ['1x1', '2x2', '4x2', '4x4', '8x8'],
                  'TPU topology for evaluation.')
flags.DEFINE_enum('tpu_platform_eval', 'jf', ['jf', 'df'],
                  'TPU platform for evaluation.')
flags.DEFINE_integer(
    'chunk', None, 'the size of chunks for evaluation inferences, set to'
    'the value that fits your GPU/TPU memory.')
flags.DEFINE_string('nametag', 'mipnerf', 'A name to prepend to the xman name')


def main(argv):
  del argv  # Unused.
  if FLAGS.is_train:
    FLAGS.train_dir = FLAGS.train_dir.format(cell=FLAGS.cell)

  train_params = {
      'gin_file': FLAGS.gin_file,
      'data_dir': FLAGS.data_dir.format(cell=FLAGS.cell),
      'train_dir': FLAGS.train_dir,
      'render_every': 0,
      'jax_tpu_async': 1,
  }

  test_params = {
      'gin_file': FLAGS.gin_file,
      'data_dir': FLAGS.data_dir.format(cell=FLAGS.cell_eval),
      'train_dir': FLAGS.train_dir,
  }
  if FLAGS.is_train:
    test_params['eval_once'] = False
    test_params['save_output'] = True
  if FLAGS.chunk is not None:
    test_params['chunk'] = FLAGS.chunk

  # Job: train
  executables = []

  # Construct training job executables
  if FLAGS.is_train:
    if FLAGS.use_tpu:
      platform = xm.Platform.from_str(FLAGS.tpu_platform)
      topology = xm.TpuTopology(FLAGS.tpu_topology)
      overrides, imports, build_target_args = jax_xm.tpu_configuration(
          platform, topology)
      requirements = jax_xm.tpu_default_requirements(platform, topology)
      exec_train = xm.BuildTarget(
          '//experimental/users/barron/mipnerf_rc:train',
          runtime=xm.Borg(
              cell=FLAGS.cell,
              priority=FLAGS.priority,
              # Uncomment this line if you want others to be able to debug your
              # trainer by looking at your logs:
              # logs_read_access_roles=['all'],
              overrides=overrides,
              requirements=requirements,
              imports=imports),
          name='train_worker',
          args={
              **build_target_args,
              **train_params
          },
          platform=platform,
          build_flags=list(xm.get_platform_build_flags(platform)) +
          ['--experimental_deps_ok'])
    else:
      train_require = xm.Requirements(
          gpu=FLAGS.n_gpus,
          gpu_types=[xm.GpuType.V100],
      )
      train_worker = xm.Borg(
          cell=FLAGS.cell,
          priority=FLAGS.priority,
          # Uncomment this line if you want others to be able to debug your
          # trainer by looking at your logs:
          # logs_read_access_roles=['all'],
          requirements=train_require)
      exec_train = xm.BuildTarget(
          '//experimental/users/barron/mipnerf_rc:train',
          name='train_worker',
          args=train_params,
          platform=xm.Platform.GPU,
          runtime=train_worker,
          build_flags=list(xm.get_platform_build_flags(platform)) +
          ['--experimental_deps_ok'])
    executables.append(exec_train)

  # Construct evaluation job executables
  if FLAGS.use_tpu_eval:
    platform = xm.Platform.from_str(FLAGS.tpu_platform_eval)
    topology = xm.TpuTopology(FLAGS.tpu_topology_eval)
    overrides, imports, build_target_args = jax_xm.tpu_configuration(
        platform, topology)
    requirements = jax_xm.tpu_default_requirements(platform, topology)
    exec_eval = xm.BuildTarget(
        '//experimental/users/barron/mipnerf_rc:eval',
        runtime=xm.Borg(
            cell=FLAGS.cell_eval,
            priority=FLAGS.priority,
            # Uncomment this line if you want others to be able to debug your
            # trainer by looking at your logs:
            # logs_read_access_roles=['all'],
            overrides=overrides,
            requirements=requirements,
            imports=imports),
        name='eval_worker',
        args={
            **build_target_args,
            **test_params
        },
        platform=platform,
        build_flags=list(xm.get_platform_build_flags(platform)) +
        ['--experimental_deps_ok'])
  else:
    eval_require = xm.Requirements(
        gpu=FLAGS.n_gpus_eval,
        gpu_types=[xm.GpuType.V100],
    )
    eval_worker = xm.Borg(
        cell=FLAGS.cell_eval,
        priority=FLAGS.priority,
        # Uncomment this line if you want others to be able to debug your
        # trainer by looking at your logs:
        # logs_read_access_roles=['all'],
        requirements=eval_require)
    exec_eval = xm.BuildTarget(
        '//experimental/users/barron/mipnerf_rc:eval',
        name='eval_worker',
        args=test_params,
        platform=xm.Platform.GPU,
        runtime=eval_worker,
        build_flags=list(xm.get_platform_build_flags(platform)) +
        ['--experimental_deps_ok'])
  executables.append(exec_eval)

  nametag = FLAGS.nametag.replace(os.sep, '_').lower()
  # Combine train and eval
  experiment = xm.ParallelExecutable(executables, name=nametag + '_service')

  experiment = xm.ParameterSweep(experiment, [{}])
  if FLAGS.is_train:
    experiment = xm.WithTensorBoard(experiment, FLAGS.train_dir)

  # Launch experiments
  description = xm.ExperimentDescription(nametag)
  xm.launch_experiment(description, experiment)


if __name__ == '__main__':
  app.run(main)
