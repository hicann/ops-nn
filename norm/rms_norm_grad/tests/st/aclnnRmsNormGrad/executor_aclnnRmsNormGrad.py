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

from abc import ABC

import torch

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.function_api import FunctionApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.configs.results_config import OutputData

logging = Logger().get_logger()

try:
    import torch_npu
except Exception as e:
    logging.warning("import torch_npu failed!!!")

@register("aclnn_rms_norm_grad")
class FunctionSegsumApi(FunctionApi):

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_dy = input_data.kwargs["dy"]
        input_x = input_data.kwargs["input"]
        rstd = input_data.kwargs["rstd"]
        gamma = input_data.kwargs["gamma"]
        s_dim = input_dy.shape[-1]
        if self.device in ["cpu", "gpu"]:
            input_x = input_x.float()
            x_rstd = (input_x * rstd).to(input_dy.dtype)
            dy_rstd = (input_dy * x_rstd).to(torch.float)
            output1 = torch.sum(dy_rstd.reshape([-1, s_dim]), dim=0, keepdim=True)
            output = ((gamma * input_dy).float() * rstd - torch.mean(
                (gamma * input_dy).float() * rstd.pow(3) * input_x, dim=-1, keepdim=True) * input_x).to(
                input_dy.dtype)
        if self.device == "npu":
            output, output1 = torch_npu.npu_rms_norm_backward(input_dy, input_x, gamma, rstd)
        return output, output1.float().view(-1)