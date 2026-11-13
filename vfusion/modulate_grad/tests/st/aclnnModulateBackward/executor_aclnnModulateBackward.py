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


@register("ascend_modulate_backward")
class ModulateBackward(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad_output = input_data.kwargs["grad_output"]
        input_tensor = input_data.kwargs["input"]
        scale = input_data.kwargs["scale"]
        dtype = grad_output.dtype

        # Match the API's FP32 arithmetic while preserving the double benchmark.
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        grad_output = grad_output.to(compute_dtype)
        input_tensor = input_tensor.to(compute_dtype)
        scale = scale.to(compute_dtype)

        grad_input = grad_output * scale.unsqueeze(1)
        grad_scale = (grad_output * input_tensor).sum(dim=1)
        grad_shift = grad_output.sum(dim=1)
        return tuple(
            output.to(dtype) for output in (grad_input, grad_scale, grad_shift)
        )
