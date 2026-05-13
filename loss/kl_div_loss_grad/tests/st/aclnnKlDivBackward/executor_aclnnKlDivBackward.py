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
import random
import torch
from torch.nn import functional as F

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("aclnn_kl_div_backward")
class AclnnKlDivBackwardApi(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        reduction_map = {
            0: "none",
            1: "mean",
            2: "sum",
            3: "batchmean"
        }

        grad_output = input_data.kwargs['grad_output']
        input = input_data.kwargs['input']
        target = input_data.kwargs['target']
        log_target = input_data.kwargs['log_target']
        reduction = reduction_map[input_data.kwargs['reduction']]

        ori_grad_dtype = grad_output.dtype
        not_fp32 = ori_grad_dtype != torch.float32
        if self.device == 'cpu' and not_fp32:
            grad_output = grad_output.to(torch.float32)
            input = input.to(torch.float32)
            target = target.to(torch.float32)

        input = input.detach().requires_grad_()

        loss = F.kl_div(input, target, reduction=reduction, log_target=log_target)

        # 适配 grad_output shape
        if reduction == "none":
            grad = grad_output.expand_as(loss)
        else:
            grad = grad_output.reshape(loss.shape)

        loss.backward(grad)
        output = input.grad

        if self.device == 'cpu' and not_fp32:
            output = output.to(ori_grad_dtype)

        return output

