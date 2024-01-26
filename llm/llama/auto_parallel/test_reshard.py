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

import paddle
import paddle.distributed as dist

mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
mesh_2 = dist.ProcessMesh([2, 3], dim_names=["y"])
# dense tensor
a = paddle.ones([2, 4])

# distributed tensor
d_tensor = dist.shard_tensor(a, mesh, [dist.Partial()])

out_d_tensor = dist.reshard(d_tensor, mesh_2, [dist.Partial()])

print(out_d_tensor)
