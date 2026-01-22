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

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import numpy as np
import logging


@register("function_ascend_max_unpool2d_backward")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):       
        gradOutput = input_data.kwargs["gradOutput"]
        # self_tensor = input_data.kwargs["self"]
        indices = input_data.kwargs["indices"]
        # outputSize = input_data.kwargs["outputSize"]
        
        if gradOutput.dim() == 3:
            gradOutput = gradOutput.unsqueeze(0)
            indices = indices.unsqueeze(0)
            is_3D = True
        else:
            is_3D = False
        
        N, C, H, W = gradOutput.shape

        gradOutput_flat = gradOutput.reshape(N, C, -1)
        indices_flat = indices.reshape(N, C, -1).long()

        output_flat = torch.gather(input=gradOutput_flat, dim=2, index=indices_flat)
        output = output_flat.reshape(N, C, H, W)

        if is_3D:
            output = output.squeeze(0)

        return output      
