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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("ascend_method_torch_nn_MaxPool2dWithIndicesBackward")
class MethodTorchNnMaxPool2DWithIndicesBackwardApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradIn = None
        if self.device == 'cpu':
            gradOut = input_data.kwargs["gradOutput"]
            self_tensor = input_data.kwargs["self"]
            indices = input_data.kwargs["indices"]
            kernelSize = input_data.kwargs["kernelSize"]
            stride = input_data.kwargs["stride"]
            padding = input_data.kwargs["padding"]
            dilation = input_data.kwargs["dilation"]
            ceilMode = input_data.kwargs["ceilMode"]

            ori_type = self_tensor.dtype
            gradOut = gradOut.to(torch.float32)
            self_tensor = self_tensor.to(torch.float32)
            indices = indices.to(torch.int64)
            gradIn = torch.ops.aten.max_pool2d_with_indices_backward(gradOut, self_tensor, kernelSize, stride,
                                                                     padding, dilation, ceilMode, indices.to(torch.int64))

            gradIn = gradIn.to(ori_type)

        return gradIn
    
    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_NCHW
    
    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        gradOut = input_data.kwargs["gradOutput"]
        self_tensor = input_data.kwargs["self"]
        kernelSize = input_data.kwargs["kernelSize"]
        stride = input_data.kwargs["stride"]
        padding = input_data.kwargs["padding"]
        dilation = input_data.kwargs["dilation"]
        ceilMode = input_data.kwargs["ceilMode"]

        self_tensor = self_tensor.to(torch.float32)
        m = torch.nn.MaxPool2d(kernelSize, stride, padding, dilation, return_indices=True, ceil_mode=ceilMode).cpu()
        out, indices = m(self_tensor.cpu())

        input_data.kwargs["indices"] = indices.to(input_data.kwargs["indices"].dtype)

        if self.device == 'pyaclnn':
            input_data.kwargs["indices"] = input_data.kwargs["indices"].to(torch.device("npu"))
