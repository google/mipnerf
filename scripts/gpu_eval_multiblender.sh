#!/bin/bash
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

# Script for evaluating on the multiscale Blender dataset.
export CUDA_VISIBLE_DEVICES=0

SCENE=lego
EXPERIMENT=debug
TRAIN_DIR=/usr/local/google/home/barron/tmp/nerf_results/$EXPERIMENT/$SCENE
DATA_DIR=/usr/local/google/home/barron/tmp/nerf_data/down4/$SCENE

blaze run -c opt --config=cuda eval --experimental_deps_ok -- \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --chunk=3076 \
  --gin_file=configs/multiblender.gin \
  --gin_param="Config.test_render_interval = 1" \
  --logtostderr
