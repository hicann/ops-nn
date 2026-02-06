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
import math
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("ascend_method_torch_nn_MaxPool2dWithMask")
class MethodTorchNnMaxPool2DWithMaskApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradIn = None
        if self.device == 'cpu':
            self_tensor = input_data.kwargs["self"]
            kernelSize = input_data.kwargs["kernelSize"]
            stride = input_data.kwargs["stride"]
            padding = input_data.kwargs["padding"]
            dilation = input_data.kwargs["dilation"]
            ceilMode = input_data.kwargs["ceilMode"]
            m = torch.nn.MaxPool2d(kernel_size=kernelSize, stride=stride, padding=padding,
                                   dilation=dilation, return_indices=True, ceil_mode=ceilMode)

            ori_type = self_tensor.dtype
            out, indices = m(self_tensor.to(torch.float32))
            out = out.to(ori_type)
            indices = indices.to(torch.int8)

        return out

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if len(input_data.kwargs['self'].shape) == 3:
            return AclFormat.ACL_FORMAT_ND
        return AclFormat.ACL_FORMAT_NCHW

@register("aclnn_max_pool_2d_with_mask")
class MaxPool2dWithMaskAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args = [] # 算子入参列表
        output_packages = [] # 算子出参数据包列表
        is_ND = 0

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        if self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        if self.device == "pyaclnn":
            device = f"npu:{self.device_id}"
        else:
            device = "cpu"

        # === 处理输入参数 ===
        # 将输入数据转换为aclnn所需C++格式
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        
        for name, kwarg in input_data.kwargs.items():
            if name == "self":
                if len(kwarg.shape) == 3:
                    C, H, W = kwarg.shape
                    N = 1
                    is_ND = 1
                else:
                    N, C, H, W = kwarg.shape
            if name == "kernelSize":
                k_h = kwarg[0]
                k_w = kwarg[1]
                H_indices = k_h * k_w
            
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        # === 处理算子输出 ===
        for index, output_data in enumerate(self.task_result.output_info_list):
            if index == 0:
                if len(output_data.shape) == 3:
                    C_out, H_out, W_out = output_data.shape
                    N_out = 1
                else:
                    N_out, C_out, H_out, W_out = output_data.shape
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        
        # 将算子输出Tensor追加到输入参数
        input_args.extend(output_packages)

        W_indices = (math.ceil(H_out * W_out / 16) + 1) * 2 * 16
        indices = torch.zeros((N, C, H_indices, W_indices), dtype=torch.int8).to(device)
        if is_ND == 1:
            indices = indices.squeeze(0)
        value_to_insert = self.backend.convert_input_data(indices)
        input_args.extend(value_to_insert)

        return input_args, output_packages
    
    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output