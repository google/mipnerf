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

# Helper script for launching a truncated training job, useful for debugging.
export CUDA_VISIBLE_DEVICES=0

SCENE=trex
EXPERIMENT=debug
TRAIN_DIR=/usr/local/google/home/barron/tmp/nerf_results/$EXPERIMENT/$SCENE

rm $TRAIN_DIR/*
blaze run -c opt --config=cuda --experimental_deps_ok train -- \
  --data_dir=/usr/local/google/home/barron/tmp/nerf_data/nerf_llff_data/$SCENE \
  --train_dir=$TRAIN_DIR \
  --render_every=100 \
  --gin_file=configs/llff.gin \
  --gin_param="Config.save_every = 100" \
  --gin_param="Config.print_every = 10" \
  --gin_param="Config.batch_size = 2048" \
  --gin_param="Config.lr_delay_mult = 1." \
  --gin_param="Config.lr_delay_steps = 0" \
  --logtostderr
