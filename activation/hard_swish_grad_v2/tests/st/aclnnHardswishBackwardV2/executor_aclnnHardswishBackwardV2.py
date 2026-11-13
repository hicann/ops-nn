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


@register("ascend_aclnn_hardswish_backward_v2")
class AclnnHardswishBackwardV2Api(BaseApi):
    """HardswishBackwardV2标杆执行器: cpu小算子拼接实现梯度公式.

    计算公式(与aclnnHardswishBackwardV2文档一致):
        gradSelf = 0                , self <= -3
        gradSelf = self / 3 + 0.5   , -3 < self < 3
        gradSelf = 1                , self >= 3
        out = gradOutput * gradSelf
    非有限值语义(与kernel实现对齐): self为NaN时两次比较结果均为false,
    gradSelf取1; gradOutput为正负Inf且gradSelf为0时, 输出遵循IEEE 754乘法
    语义, 结果为NaN。
    """

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad_output = input_data.kwargs["gradOutput"]
        self_x = input_data.kwargs["self"]
        ori_dtype = self_x.dtype

        # 与kernel实现对齐: fp32中间计算, 逐段select后cast回原dtype
        g = grad_output.to(torch.float32)
        x = self_x.to(torch.float32)
        one_third = torch.tensor(1.0 / 3.0, dtype=torch.float32)
        one_half = torch.tensor(0.5, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        one = torch.tensor(1.0, dtype=torch.float32)

        val = x * one_third + one_half
        val = torch.where(x > -3.0, val, zero)
        grad_self = torch.where(x < 3.0, val, one)

        output = g * grad_self
        return output.to(ori_dtype)
