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
from atk.tasks.api_execute.function_api import FunctionApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
import numpy as np
import torch


@register("batch_norm_elemt_cpu")
class AclnnBatchNormElemt(FunctionApi):
    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "input" or index == 0:
            input = input_data.kwargs["input"]
            dims = input.dim()
            if dims == 2:
                return AclFormat.ACL_FORMAT_NC
            elif dims == 3:
                return AclFormat.ACL_FORMAT_NCL
            elif dims == 4:
                return AclFormat.ACL_FORMAT_NCHW
            elif dims == 5:
                return AclFormat.ACL_FORMAT_NCDHW
        return AclFormat.ACL_FORMAT_ND


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        oritype = input_data.kwargs["input"].dtype
        input = input_data.kwargs["input"]
        weight = input_data.kwargs["weight"]
        bias = input_data.kwargs["bias"]
        mean = input_data.kwargs["mean"]
        invstd = input_data.kwargs["invstd"]
        eps = input_data.kwargs["eps"]

        if self.device == 'cpu' :
            input = input.to(torch.float32)
            bias = bias.to(torch.float32)
            mean = mean.to(torch.float32)
            weight = weight.to(torch.float32)
            invstd = invstd.to(torch.float32)
            
            output= torch.nn.functional.batch_norm(input,mean,invstd,weight,bias,False,1,eps)
            return output.to(oritype)
