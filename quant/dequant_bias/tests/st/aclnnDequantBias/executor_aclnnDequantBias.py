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

@register("aclnn_dequant_bias")
class TorchGluBackward(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x_data = input_data.kwargs["x"]
        weight_scale_data = input_data.kwargs["weightScale"]
        activate_scale_data = input_data.kwargs["activateScaleOptional"]
        bias_data = input_data.kwargs["biasOptional"]
        output_dtype = input_data.kwargs["outputDtype"]
        # 如果 bias_data 不为 None 且类型是 int32，则忽略偏置，y = (x_data + bias_data) * weight_scale_data * activate_scale_data
        if bias_data is not None:
            if bias_data.dtype == torch.int32:
                # bias_data 是 int32 类型，忽略偏置
                if activate_scale_data is None:
                    # 如果 activate_scale_data 是 None，则忽略乘以 activate_scale_data
                    y = (x_data + bias_data) * weight_scale_data
                else:
                    y = (x_data + bias_data) * weight_scale_data * activate_scale_data[:, None]
            elif bias_data.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                # bias_data 是 fp16 或 bf16 或 fp32 类型，使用 y = x_data * weight_scale_data * activate_scale_data + bias_data
                if activate_scale_data is None:
                    # 如果 activate_scale_data 是 None，则忽略乘以 activate_scale_data
                    y = x_data * weight_scale_data + bias_data
                else:
                    y = x_data * weight_scale_data * activate_scale_data[:, None] + bias_data
            else:
                raise ValueError(f"Unsupported bias_data type: {bias_data.dtype}")
        else:
            # 如果没有 bias_data，则使用 y = x_data * weight_scale_data * activate_scale_data
            if activate_scale_data is None:
                # 如果 activate_scale_data 是 None，则忽略乘以 activate_scale_data
                y = x_data * weight_scale_data
            else:
                y = x_data * weight_scale_data * activate_scale_data[:, None]

        # 根据 output_dtype 设置输出数据类型
        if output_dtype == 1:
            y = y.to(torch.float16)  # fp16
        elif output_dtype == 27:
            y = y.to(torch.bfloat16)  # bf16
        else:
            raise ValueError("Unsupported output_dtype value. Use 1 for fp16 or 27 for bf16.")

        return y