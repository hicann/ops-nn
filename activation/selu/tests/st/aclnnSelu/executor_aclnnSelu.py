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
            # int8 路径：int8 -> float16 -> compute -> ceil negative -> int8
            import numpy as np
            SCALE_ALPHA = 1.75809934085
            SCALE = 1.05070098736
            x_np = input_x.cpu().numpy().astype(np.float16)
            neg_res = np.minimum(x_np, np.float16(0))
            pos_res = np.maximum(x_np, np.float16(0))
            exp_res = np.exp(neg_res)
            sub_res = exp_res - np.float16(1)
            neg_muls = (sub_res * np.float16(SCALE_ALPHA)).astype(np.float16)
            neg_muls = np.ceil(neg_muls).astype(np.float16)
            pos_muls = (pos_res * np.float16(SCALE)).astype(np.float16)
            res = (neg_muls + pos_muls).astype(np.float16)
            res = np.clip(res, np.float16(-128), np.float16(127))
            output_tensor = torch.from_numpy(res.astype(np.int8))
        elif input_x.dtype == 'int32':
            input_x = input_x.float()
            output_tensor = torch.selu(input_x).to(torch.int32)
        else:
            output_tensor = torch.selu(input_x)
        return output_tensor

