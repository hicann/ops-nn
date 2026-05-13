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

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("gdd_function_rms_norm_910b")
class FunctionApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "npu":
            import torch_npu
            torch_npu.npu.set_compile_mode(jit_compile=False)
        input_x = input_data.kwargs["x"]
        gamma = input_data.kwargs["gamma"]
        eps = 1e-6

        if input_x.shape == torch.Size([]) or (0 in input_x.shape) :
             output = input_x
             output1 = input_x
             output1 = output1.type(torch.float32)
        else:

            if self.output is None:
                if self.device == "cpu" or self.device == "gpu":
                    output1 = torch.rsqrt(input_x.pow(2).mean(-1, keepdim=True) + eps)
                    out = input_x * output1
                    gamma_add = gamma.add(1)
                    output = out * gamma_add
                    output1 = output1.type(torch.float32)
                elif self.device == "npu":
                    import torch_npu
                    output, output1 = torch_npu.npu_gemma_rms_norm(input_x, gamma, eps)

            else:
                if "pro" in self.name:
                    input_x = input_x.type(torch.float32)
                    gamma = gamma.type(torch.float32)
                    output1 = torch.rsqrt(input_x.pow(2).mean(-1, keepdim=True) + eps)
                    out = input_x * output1
                    gamma_add = gamma.add(1)
                    out * gamma_add
                else:
                    import torch_npu
                    torch_npu.npu_gemma_rms_norm(input_x, gamma, eps)
                if isinstance(self.output, int):
                    output = input_data.args[self.output]
                elif isinstance(self.output, str):
                    output = input_data.kwargs[self.output]
                else:
                    raise ValueError(
                        f"self.output {self.output} value is " f"error"
                    )
        return output, output1
    def init_by_input_data(self, input_data: InputDataset):
        input_x = input_data.kwargs["x"]

        if self.device == "pyaclnn":
            input_data.kwargs['eps'] = 1e-6
            if input_x.shape == torch.Size([]) or (0 in input_x.shape) :
                pass
            
    
