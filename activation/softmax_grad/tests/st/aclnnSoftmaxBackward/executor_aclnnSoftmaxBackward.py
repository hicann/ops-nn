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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("function_softmax_backward")
class MethodTorchNnSoftmaxApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        origin_dtype = input_data.kwargs["gradOutput"].dtype
        input_gradOutput = input_data.kwargs["gradOutput"]
        input_Output = input_data.kwargs["Output"]
        input_dim = input_data.kwargs["dim"]
        # print(input_data)

        if (origin_dtype == torch.float16) or (origin_dtype == torch.bfloat16):
            # output = torch.nn.functional.softmax(input_d.to(torch.float32), dim).to(origin_dtype)
            output_tensor = torch.ops.aten._softmax_backward_data(input_gradOutput.to(torch.float32),
                                                                  input_Output.to(torch.float32), input_dim, 1).to(
                origin_dtype)
        else:
            output_tensor = torch.ops.aten._softmax_backward_data(input_gradOutput, input_Output, input_dim, 1)

        # print(output_tensor[0])

        return output_tensor