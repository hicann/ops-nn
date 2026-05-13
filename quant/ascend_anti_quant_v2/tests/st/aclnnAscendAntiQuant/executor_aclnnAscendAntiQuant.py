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
#

import torch
import numpy as np
from ml_dtypes import int4
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

@register("aclnnAscendAntiQuant")
class AscendAntiQuant(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # cast_f16_ub = tbe.cast_to(x, "float16")
        x = input_data.kwargs["x"]
        scale = input_data.kwargs["scale"]
        offset = input_data.kwargs["offset"]
        dstType = input_data.kwargs["dstType"]
        sqrt_mode = input_data.kwargs["sqrt_mode"]
        if x.dtype == torch.int32:
            dst_shape = x.numpy().shape
            if len(dst_shape) == 0:
                dst_shape = (8, )
            else:
                dst_shape = (*(dst_shape[:-1]), dst_shape[-1] * 8)

            sa1 = x.numpy().astype(np.int32)
            sa2 = sa1.tobytes()
            sa3 = np.frombuffer(sa2, dtype=np.uint8)
            shift = np.array([0, 4], dtype=np.uint8)
            sa4 = np.bitwise_and(sa3.reshape([-1, 1]) >> shift, 0b00001111).astype(int4).astype(np.int8).reshape(dst_shape)
            x = torch.from_numpy(sa4)

            x_f16 = x.to(torch.float16)
            x_f32 = x_f16.to(torch.float32)
            scale_f32 = scale.to(torch.float32)
            offset_f32 = offset.to(torch.float32)
            x_offset_fp32 = x_f32 + offset_f32
            res = x_offset_fp32 * scale_f32
            if sqrt_mode:
                res = x_offset_fp32 * scale_f32 * scale_f32
        else:
            x_offset = x + offset
            res = x_offset * scale
            if sqrt_mode:
                res = res * scale
        if dstType ==1:
            res = res.to(torch.float16)
        else:
            res = res.to(torch.bfloat16)
        return res

