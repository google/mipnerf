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


NAME=mipnerf

for CONFIG in blender blender_noextras blender_noipe
do
for SCENE in chair drums ficus hotdog lego materials mic ship
do
  TAG=${NAME}/${CONFIG}/${SCENE}
  echo "***********************************************************************"
  echo ${TAG}
  echo "***********************************************************************"
  /google/bin/releases/xmanager/cli/xmanager.par launch \
    google/xm/xm_launcher.py -- \
    --xm_deployment_env=alphabet \
    --xm_resource_alloc=group:peace/gcam \
    --xm_resource_pool=peace \
    --xm_skip_launch_confirmation \
    --noxm_monitor_on_launch \
    --cell=lu \
    --cell_eval=lu \
    --use_tpu \
    --tpu_topology=4x4 \
    --tpu_platform=jf \
    --use_tpu_eval \
    --tpu_topology_eval=4x4 \
    --tpu_platform_eval=jf \
    --data_dir=/cns/lu-d/home/barron/nerf_data/nerf_synthetic/${SCENE} \
    --train_dir=/cns/{cell}-d/home/${USER}/nerf/${TAG}  \
    --priority=100 \
    --gin_file=configs/${CONFIG}.gin \
    --nametag=${TAG}
done
done
