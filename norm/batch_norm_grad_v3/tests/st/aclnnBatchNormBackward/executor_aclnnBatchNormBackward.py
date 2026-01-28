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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
import numpy as np


@register("function_aclnnBatchNormBackward")
class aclnnBatchNormBackwardExecutor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradOut = input_data.kwargs["input0"]
        input_x = input_data.kwargs["input1"]
        weight = input_data.kwargs["input2"]
        runningMean = input_data.kwargs["input3"]
        runningVar = input_data.kwargs["input4"]
        saveMean = input_data.kwargs["input5"]
        saveInvstd = input_data.kwargs["input6"]
        training = input_data.kwargs["training"]
        eps = input_data.kwargs["eps"]
        outputMask = input_data.kwargs["outputmask"]

        shape_list = [gradOut.shape, weight.shape, weight.shape]
        dtype_list = [gradOut.dtype, weight.dtype, weight.dtype]

        gradOut_t = None
        input_x_t = None

        if gradOut.dtype == torch.float16:
            gradOut_t = gradOut.to(torch.float32)
            input_x_t = input_x.to(torch.float32)
        else:
            gradOut_t = gradOut
            input_x_t = input_x

        res = torch.ops.aten.native_batch_norm_backward(gradOut_t, input_x_t, weight, runningMean, runningVar, saveMean, saveInvstd, training, eps, outputMask)

        res_list = []
        for i in range(3):
            if res[i] != None:
                res_list.append(res[i])
            else:
                res_list.append(torch.zeros(shape_list[i]).to(dtype_list[i]))
        if gradOut.dtype == torch.float16:
            res_list[0] = res_list[0].to(torch.float16)

        output = tuple(res_list)

        return output


@register("aclnnBatchNormBackward_aclnn_api")
class aclnnBatchNormBackwardAclnnApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
        self.outputMask = []
        self.shape_list = []
        self.dtype_list = []

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "input0" or name == "input1" or index == 0:
            gradOut = input_data.kwargs["input0"]
            weight = input_data.kwargs["input2"]
            self.outputMask = input_data.kwargs["outputmask"]
            self.shape_list = [gradOut.shape, weight.shape, weight.shape]
            self.dtype_list = [gradOut.dtype, weight.dtype, weight.dtype]
            dims = gradOut.dim()
            if dims == 2:
                return AclFormat.ACL_FORMAT_NC
            elif dims == 3:
                return AclFormat.ACL_FORMAT_NCL
            elif dims == 4:
                return AclFormat.ACL_FORMAT_NCHW
            elif dims == 5:
                return AclFormat.ACL_FORMAT_NCDHW
        return AclFormat.ACL_FORMAT_ND

    def after_call(self, output_packages):
        output = []
        for i in range(3):
            if self.outputMask[i]:
                output.append(self.acl_tensor_to_torch(output_packages[i]))
            else:
                output.append(torch.zeros(self.shape_list[i]).to(self.dtype_list[i]))
        return output
        