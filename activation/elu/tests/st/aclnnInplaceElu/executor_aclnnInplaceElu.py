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
import torch.nn as nn
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import logging

class ExtendedELU(nn.Module):
    def __init__(self, alpha=1.0, scale=1.0, input_scale=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.scale = scale
        self.input_scale = input_scale
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            mask = x <= 0
            x[mask] = self.alpha * self.scale * (torch.exp(self.input_scale * x[mask]) - 1)
            x[~mask] *= self.scale
            return x
        else:
            return torch.where(
                x > 0,
                self.scale * x,
                self.alpha * self.scale * (torch.exp(self.input_scale * x) - 1)
            )

@register("function_inplace_elu")
class MethodAclnnInplaceEluApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnInplaceEluApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_a = input_data.kwargs["self"]    
        input_b = input_data.kwargs["alpha"]
        input_c = input_data.kwargs["scale"]
        input_d = input_data.kwargs["inputScale"]

        ext_elu = ExtendedELU(alpha=input_b, scale=input_c, input_scale=input_d, inplace=True)
        output = ext_elu(input_a)

        return output

@register("aclnn_inplace_elu")
class AddAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output