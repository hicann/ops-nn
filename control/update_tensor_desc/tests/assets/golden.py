#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""update_tensor_desc — TestSpec（golden + ThirdPartyImpl + pre_compare）。

语义：输出 y 是 Device 侧 128×int64 描述缓冲的 RMW 视图：y[3] = N = len(shape)、
y[4+i] = shape[i]，其余元素保留执行前原值；numel > 128 时仅前 128 元素被 RMW，
可观察效果等价于「只覆盖 [3, 3+N)」。x 为占位输入，kernel 不读取其数据；
输出 shape = attr(shape)，dtype 恒为 int64。

期望口径（跨通路，走默认 binary_equal 整块比对）：RMW 窗口 [3, 4+N) 是算子唯一
确定性输出；窗口外是「保留区」，其值 = 执行前缓冲初值，而初值由执行通路决定
（kernel：TTK 预填 1；GEIR：GE RunGraph 内部分配，实测 0 且无契约保证），
不属于算子契约。golden 保留区填 0 占位；pre_compare 在默认比对前把 NPU 输出
的保留区清零后整体返回（GEIR 输出是 frombuffer 只读数组，须改副本返回）。
写入区必有值 ≠ 0（y[3] = N ≥ 1、shape[i] ≥ 1），漏写 / 清零 / 错位必被检出。

无对标竞品（PyTorch / TF 均无同名算子），ThirdPartyImpl 用 torch 小算子拼，
与 golden 同口径（保留区 0 + 窗口真值）。
纯 int64 元数据搬运，无浮点 / 累加 → binary_equal。
"""

__spec__ = {"update_tensor_desc": "UpdateTensorDescSpec"}

import math

import numpy
import torch


class UpdateTensorDescSpec:
    def golden(x, *, shape=None, **kwargs):
        """golden：numpy 进 numpy 出。窗口 [3, 4+N) 为真值，保留区 0 占位（pre_compare 会抹平）。"""
        attr_shape = [int(d) for d in shape]
        n = len(attr_shape)
        flat = numpy.zeros(math.prod(attr_shape), dtype=numpy.int64)
        flat[3] = n
        flat[4 : 4 + n] = attr_shape
        try:
            return [flat.reshape(attr_shape)]
        except ValueError:
            return [flat]

    def pre_compare(npu_out, golden_out):
        """默认比对前抹平通路差异：保留区初值由通路决定（kernel 预填 1 / GE 分配），
        不属于算子契约 → 清零 NPU 输出的保留区后整体返回（返回值模式）；窗口
        [3, 4+N) 不动，交给默认 binary_equal 判定。N 取自 golden[3]。"""
        n = int(golden_out.reshape(-1)[3])
        a = numpy.array(npu_out)  # copy：GEIR 输出为只读数组
        flat = a.reshape(-1)
        flat[:3] = 0
        flat[4 + n :] = 0
        return [a, golden_out]

    class ThirdPartyImpl:
        """third_party：torch 进 torch 出，无竞品 API，用 torch 小算子拼；与 golden 同口径。"""

        def __init__(self, *, shape=None, **kwargs):
            self.shape = [int(d) for d in shape]

        def __call__(self, x, *, device=None, **kwargs):
            n = len(self.shape)
            flat = torch.zeros(math.prod(self.shape), dtype=torch.int64, device=device)
            flat[3] = n
            flat[4 : 4 + n] = torch.tensor(self.shape, dtype=torch.int64, device=device)
            try:
                return [flat.reshape(self.shape)]
            except RuntimeError:
                return [flat]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {"int64": {"standard": "binary_equal"}}
