#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import os
import random

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

@register("function_selu")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs['x']

        if input_x.dtype == 'int8':
            # int8 参考实现：对齐 kernel ComputeInt8（op_kernel/arch35/selu.h）的定点语义。
            #   正分支 max(x,0)*SCALE：SCALE 取其 fp16 表示(1.05078125)，fp32 精确乘后向零截断(RTZ)回 fp16
            #                          （A5 硬件 fp16 Muls 为 RNE，直乘会在 x=59/98/118 多 1，故走 RTZ）。
            #   负分支 (exp(min(x,0))-1)*SCALE_ALPHA：half 精度计算后 ceil（向 +inf 取整）。
            #   合并后 half->int8：溢出回绕（numpy astype(int8) 语义）t-256*(t>=128)，而非饱和 clip。
            SCALE_ALPHA = 1.75809934085          # SCALE * ALPHA 预乘常量
            SCALE_F16 = 1.05078125               # = np.float16(1.05070098736)，SCALE 的 fp16 表示
            x_np = input_x.cpu().numpy().astype(np.float16)

            # 负分支：half 精度计算 + ceil
            neg_res = np.minimum(x_np, np.float16(0))
            sub_res = (np.exp(neg_res).astype(np.float16) - np.float16(1)).astype(np.float16)
            neg_muls = (sub_res * np.float16(SCALE_ALPHA)).astype(np.float16)
            neg_muls = np.ceil(neg_muls).astype(np.float16)

            # 正分支：fp32 精确乘 + 向零截断(RTZ)回 fp16，对齐 kernel 的 CAST_TRUNC
            pos_res = np.maximum(x_np, np.float16(0))
            pos_f32 = pos_res.astype(np.float32) * np.float32(SCALE_F16)
            pos_muls = pos_f32.astype(np.float16)                              # RNE 舍入
            overshoot = np.abs(pos_muls.astype(np.float32)) > np.abs(pos_f32)  # RNE 向外舍入处
            pos_muls = np.where(overshoot, np.nextafter(pos_muls, np.float16(0)), pos_muls).astype(np.float16)

            # 合并 + 溢出回绕（与 kernel 一致：先 trunc 取整，再 t-256*(t>=128)）
            merged = (neg_muls + pos_muls).astype(np.float16)
            ti32 = np.trunc(merged.astype(np.float32)).astype(np.int32)
            wrapped = ti32 - np.int32(256) * (ti32 >= np.int32(128))
            output_tensor = torch.from_numpy(wrapped.astype(np.int8))
        elif input_x.dtype == 'int32':
            input_x = input_x.float()
            output_tensor = torch.selu(input_x).to(torch.int32)
        else:
            output_tensor = torch.selu(input_x)
        return output_tensor

