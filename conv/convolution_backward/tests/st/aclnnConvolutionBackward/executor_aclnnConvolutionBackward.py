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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("golden_conv_backward")
class AscendConvolutionBackwardInputApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AscendConvolutionBackwardInputApi, self).__init__(task_result)
        self.format = None

    def init_by_input_data(self, input_data: InputDataset):
        self.format = input_data.kwargs["format"]

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
            input_data.kwargs["gradOutput"].to(torch.float32),
            input_data.kwargs["input"].to(torch.float32),
            input_data.kwargs["weight"].to(torch.float32),
            bias_sizes = input_data.kwargs["biasSizes"],
            stride = input_data.kwargs["stride"],
            padding = input_data.kwargs["padding"],
            dilation = input_data.kwargs["dilation"],
            transposed = input_data.kwargs["transposed"],
            output_padding = input_data.kwargs["outputPadding"],
            groups = input_data.kwargs["groups"],
            output_mask = input_data.kwargs["outputMask"]
        )
        if input_data.kwargs["outputMask"][0] == False:
            grad_input = torch.zeros(input_data.kwargs["input"].shape).to(input_data.kwargs["input"].dtype)
        if input_data.kwargs["outputMask"][1] == False:
            grad_weight = torch.zeros(input_data.kwargs["weight"].shape).to(input_data.kwargs["weight"].dtype)
        if input_data.kwargs["outputMask"][2] == False:
            if self.format == "NDHWC" or self.format == "NHWC":
                channelIndex = len(input_data.kwargs["gradOutput"].shape) - 1
            else:
                channelIndex = 1
            grad_bias = torch.zeros(input_data.kwargs["gradOutput"].shape[channelIndex]).to(input_data.kwargs["gradOutput"].dtype)
        if grad_input != None:
            grad_input = grad_input.to(input_data.kwargs["gradOutput"].dtype)
        if grad_weight != None:
            grad_weight = grad_weight.to(input_data.kwargs["gradOutput"].dtype)
        if grad_bias != None:
            grad_bias = grad_bias.to(input_data.kwargs["gradOutput"].dtype)
        
        return grad_input, grad_weight, grad_bias

    def get_format(self, input_data: InputDataset, index=None, name=None):
            """
            :param input_data: 参数列表
            :param index: 参数位置
            :param name: 参数名字
            :return:
            format at this index or name
            """
            if index == 2:
                return AclFormat.ACL_FORMAT_ND
            if input_data.kwargs["format"] == "NCHW":
                return AclFormat.ACL_FORMAT_NCHW
            elif input_data.kwargs["format"] == "NCL":
                return AclFormat.ACL_FORMAT_NCL
            elif input_data.kwargs["format"] == "ND":
                return AclFormat.ACL_FORMAT_ND
            elif input_data.kwargs["format"] == "NCDHW":
                return AclFormat.ACL_FORMAT_NCDHW

@register("aclnn_conv_backward")
class aclnnAscendConvolutionBackwardInputApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnAscendConvolutionBackwardInputApi, self).__init__(task_result, backend)
        self.input = None

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # # 输出在输入前
        # output_packages[:] = [input_args[2]]
        # input_args.pop()
        # atk不支持int8 json输入，在此转换
        import ctypes
        input_args[11] = ctypes.c_int8(input_data.kwargs["cubeMathType"])
        input_args[9] = ctypes.c_int(input_data.kwargs["groups"])
        self.input = input_data
        return input_args, output_packages
    
    def after_call(self, output_packages):
        grad_input, grad_weight, grad_bias = super().after_call(output_packages)
        
        if self.input.kwargs["outputMask"][0] == False:
            grad_input = torch.zeros(self.input.kwargs["input"].shape).to(self.input.kwargs["input"].dtype)
        if self.input.kwargs["outputMask"][1] == False:
            grad_weight = torch.zeros(self.input.kwargs["weight"].shape).to(self.input.kwargs["weight"].dtype)
        if self.input.kwargs["outputMask"][2] == False:
            if self.input.kwargs["format"] == "NDHWC" or self.input.kwargs["format"] == "NHWC":
                channelIndex = len(self.input.kwargs["gradOutput"].shape) - 1
            else:
                channelIndex = 1
            grad_bias = torch.zeros(self.input.kwargs["gradOutput"].shape[channelIndex]).to(self.input.kwargs["gradOutput"].dtype)
        
        return grad_input, grad_weight, grad_bias

    def get_cpp_func_signature_type(self):
        # 校验函数
        return "aclnnStatus aclnnAddmvGetWorkspaceSize(const aclTensor *gradOutput, \
            const aclTensor *input, const aclTensor *weight, const aclIntArray *biasSizes, \
            const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, \
            bool transposed, const aclIntArray *outputPadding, int groups, \
            const aclBoolArray *outputMask, int8_t cubeMathType, \
            aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, \
            uint64_t* workspaceSize, aclOpExecutor** executor)"