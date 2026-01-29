#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

@register("function_aclnn_avgpool3d_backward")
class Functionavgpool3dApi(BaseApi):
    def __call__(self, input_data, InputDataset, with_output: bool = False):
        x1 = input_data.kwargs["gradOutput"]
        x2 = input_data.kwargs["self"]
        kernel_size = input_data.kwargs["kernelSize"]
        stride = input_data.kwargs["stride"]
        padding = input_data.kwargs["padding"]
        ceil_mode = input_data.kwargs["ceilMode"]
        count_include_pad = input_data.kwargs["countIncludePad"]
        divisor_override = input_data.kwargs["divisorOverride"]

        if x1.dtype == torch.bfloat16 or x1.dtype == torch.float16:
            in_dtype = x1.dtype
            is_16 = True
            x1 = x1.to(torch.float32)
            x2 = x2.to(torch.float32)
        else:
            is_16 = False
        
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        
        model_result = torch.ops.aten.avg_pool3d_backward(
            x1,
            x2,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override)
        y = model_result

        if is_16:
            y = y.to(in_dtype)
        return y
