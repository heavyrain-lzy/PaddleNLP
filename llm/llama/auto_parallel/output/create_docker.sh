# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUDA_SO="$(\ls /usr/lib64/libcuda* | grep -v : | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | grep -v : | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(find /dev/nvidia* -maxdepth 1 -not -type d | xargs -I{} echo '--device {}:{}')
nvidia-modprobe -u -c=0
set -x
PWD=`pwd`

docker_home=lzy_paddle
docker run -d\
   ${CUDA_SO} \
   ${DEVICES} \
   -v /dev/shm:/dev/shm \
   -it --name $docker_home \
   --net=host \
   --privileged  \
   --entrypoint=/bin/bash \
   -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi \
   -v $PWD:/$docker_home/ \
   -p 9905:9905 -p 9906:9906 -p 9803:9803 \
   iregistry.baidu-int.com/paddlecloud/base-images:paddlecloud-ubuntu18.04-gcc8.2-cuda11.8-cudnn8.6-nccl2.15.5
