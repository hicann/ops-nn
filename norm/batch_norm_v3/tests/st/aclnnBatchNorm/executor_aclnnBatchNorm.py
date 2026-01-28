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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import numpy as np
import torch


@register("aclnn_batch_norm_test")
class AclnnBatchNorm(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    istrain = False

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        self.istrain = input_args[5].value
        return input_args, output_packages

    # 默认调用，可省略
    def after_call(self, output_packages):
        output1,output2,output3 = super().after_call(output_packages)
        if not self.istrain :
            output2.zero_()
            output3.zero_()
            self.istrain = False

        return output1, output2,output3

    # 默认调用，可省略
    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND


@register("batch_norm_cpu")
class BatchNorm(BaseApi):
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
        runningMean = input_data.kwargs["runningMean"]
        runningVar = input_data.kwargs["runningVar"]
        training = input_data.kwargs["training"]
        momentum = input_data.kwargs["momentum"]
        eps = input_data.kwargs["eps"]

        if self.device == 'cpu' :
            input = input.to(torch.float32)
            bias = bias.to(torch.float32)
            runningMean = runningMean.to(torch.float32)
            weight = weight.to(torch.float32)
            runningVar = runningVar.to(torch.float32)
            output= torch.nn.functional.batch_norm(input,runningMean,runningVar,weight,bias,training,momentum,eps)
            for i in range(runningMean.shape[0]):
                runningMean[i] = torch.mean(input[:,i,])
                runningVar[i] = torch.var(input[:,i,])
            if training :
                return output.to(oritype),runningMean.to(oritype),runningVar.to(oritype)
            else:
                return output.to(oritype),runningMean.zero_(),runningVar.zero_()
