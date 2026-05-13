#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import logging
import torch
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute.base_api import BaseApi
import numpy as np
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import torch.nn as nn

@register("function_aclnnAdaLayerNorm")
class FunctionaclnnAdaLayerNorm(BaseApi):
    
    # 精度对比标杆
    def AdaLayerNormGolden(self, x, weightOptional, biasOptional, scale, shift, epsilon):
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(torch.float32)
            weightOptional = None if weightOptional is None else weightOptional.to(torch.float32)
            biasOptional = None if biasOptional is None else biasOptional.to(torch.float32)
            scale = scale.to(torch.float32)
            shift = shift.to(torch.float32)
        factor = torch.tensor((1.0 / x.shape[-1]), dtype=torch.float32)

        mean = torch.sum(x * factor, dim=-1)
        x_mean = x - mean.unsqueeze(-1)
        var = torch.sum(x_mean * x_mean * factor, dim=-1)
        rstd = 1.0 / torch.sqrt(var + epsilon)
        y = x_mean * rstd.unsqueeze(-1)
        if weightOptional is not None:
            y = y * weightOptional
        if biasOptional is not None:
            y = y + biasOptional

        if len(scale.shape) != len(x.shape):
          scale = scale.unsqueeze(-2)
        if len(shift.shape) != len(x.shape):
          shift = shift.unsqueeze(-2)
        out = y * (1 + scale) + shift
        return out

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == 'pyaclnn':
            input_data.kwargs["epsilon"] = ctypes.c_double(input_data.kwargs["epsilon"])

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        scale = input_data.kwargs["scale"]
        shift = input_data.kwargs["shift"]
        weightOptional = input_data.kwargs["weightOptional"]
        biasOptional = input_data.kwargs["biasOptional"]
        epsilon = input_data.kwargs["epsilon"]
        if weightOptional.shape == torch.Size([]):
            weightOptional = None
        if biasOptional.shape == torch.Size([]):
            biasOptional = None

        if self.output is None:
            out = self.AdaLayerNormGolden(x, weightOptional, biasOptional, scale, shift, epsilon)
        else:
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        return out


@register("function_aclnnAdaLayerNorm_pgacl")
class FunctionaclnnAdaLayerNorm_pgacl(AclnnBaseApi):

    def init_by_input_data(self, input_data: InputDataset):
        x = input_data.kwargs["x"]
        for index, output_data in enumerate(self.task_result.output_info_list):
            if x.dtype == torch.bfloat16:
                output_data.dtype = "torch.bfloat16"
            elif x.dtype == torch.float16:
                output_data.dtype = "torch.float16"
        
        input_args, output_packages = super().init_by_input_data(input_data)
        
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
        weightOptional = input_data.kwargs["weightOptional"]
        if weightOptional.shape == torch.Size([]):
            input_args[3] = TensorPtr()
        biasOptional = input_data.kwargs["biasOptional"]
        if biasOptional.shape == torch.Size([]):
            input_args[4] = TensorPtr()

        return input_args, output_packages
