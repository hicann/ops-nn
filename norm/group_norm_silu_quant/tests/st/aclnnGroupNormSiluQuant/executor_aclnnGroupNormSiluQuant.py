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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import numpy as np


@register("ascend_function_group_norm_silu_quant")
class MethodTorchGroupNormSiluQuantApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupNormSiluQuantApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        dtype = x.dtype
        gamma = input_data.kwargs["gamma"]
        beta = input_data.kwargs["beta"]
        group = input_data.kwargs["group"]
        eps = input_data.kwargs["eps"]
        activateSilu = input_data.kwargs["activateSilu"]
        # 新增：获取量化尺度参数
        quantScale = input_data.kwargs["quantScale"]
    
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        N = x.shape[0]
        C = x.shape[1]
        remaining_dims = x.shape[2:]
        HW = 1
        for size in remaining_dims:
            HW *= size

        gamma_golden = gamma.to(torch.float32).reshape(1, C, 1)
        beta_golden = beta.to(torch.float32).reshape(1, C, 1)
        x_golden = x.to(torch.float32).reshape(N, group, -1)

        mean_x = torch.mean(x_golden, dim=-1, keepdim=True)
        var_x = torch.var(x_golden, dim=-1, unbiased=False, keepdim=True)
        rstd_x = 1 / torch.sqrt(var_x + eps)
        group_norm_x = (x_golden - mean_x) * rstd_x
        group_norm_x = group_norm_x.reshape(N, C, HW)*gamma_golden + beta_golden

        out = group_norm_x
        if activateSilu == True :
            sigmoid_out = 1 / (1 + torch.exp(-1 * out))
            out = out * sigmoid_out
        
        # 确保张量在CPU上，因为量化操作通常需要在CPU上执行
        out_cpu = out.cpu().to(torch.float32)
        quantScale_cpu = torch.from_numpy(quantScale.astype(np.float32)) if isinstance(quantScale, np.ndarray) else quantScale.cpu().to(torch.float32)
        
        # 根据quantScale的形状确定量化参数
        if quantScale_cpu.shape[0] == C:
            # 逐通道量化：每个通道有独立的量化参数
            zero_points = torch.zeros([C], dtype=torch.int32)
            axis = 1  # 通道维度
            x_int8 = torch.quantize_per_channel(out_cpu, quantScale_cpu, zero_points=zero_points,
                                                axis=axis, dtype=torch.qint8).int_repr()
        else:
           # 情况2：进行标量量化 (Per-Tensor)
            scale_value = quantScale_cpu.item() if isinstance(quantScale_cpu, torch.Tensor) else quantScale_cpu
            zero_point_value = 0  # 标量量化通常将 zero_point 设为 0，尤其是对称量化时
            # 使用标量量化接口
            x_int8 = torch.quantize_per_tensor(out_cpu, scale=scale_value, zero_point=zero_point_value,
                                               dtype=torch.qint8).int_repr()

        # 重塑为原始形状 (N, C, H, W)
        x_int8 = x_int8.reshape(N, C, *remaining_dims)
        # 需要调整mean_x和rstd_x的形状以符合预期输出
        mean_out = mean_x.squeeze(-1)  # 从(N, group, 1)变为(N, group)
        rstd_out = rstd_x.squeeze(-1)  # 从(N, group, 1)变为(N, group)

        return x_int8, mean_out.to(dtype), rstd_out.to(dtype)