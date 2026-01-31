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
import torch_npu
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import numpy as np

@register("ascend_method_torch_nn_MaxPool3dWithArgmaxBackward")
class MethodTorchNnMaxPool3DWithArgmaxBackwardApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradIn = None
        if self.device == 'cpu':
            gradOut = input_data.kwargs["gradOutput"]
            x = input_data.kwargs["self"]
            kernelSize = input_data.kwargs["kernelSize"]
            stride = input_data.kwargs["stride"]
            padding = input_data.kwargs["padding"]
            dilation = input_data.kwargs["dilation"]
            ceilMode = input_data.kwargs["ceilMode"]
            m = torch.nn.MaxPool3d(kernel_size=kernelSize, stride=stride, padding=padding,
                                   dilation=dilation, return_indices=True, ceil_mode=ceilMode)
            ori_type = x.dtype
            x_float = x.to(torch.float)
            gradOut_float = gradOut.to(torch.float)
            out, indices = m(x_float)
            gradIn = torch.ops.aten.max_pool3d_with_indices_backward(gradOut_float, x_float, kernelSize, stride,
                                                                     padding, dilation, ceilMode, indices.to(torch.int64))
        return gradIn.to(ori_type) if ori_type != torch.float else gradIn
    
    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'pyaclnn':
            x = input_data.kwargs["self"]
            kernelSize = input_data.kwargs["kernelSize"]
            stride = input_data.kwargs["stride"]
            padding = input_data.kwargs["padding"]
            dilation = input_data.kwargs["dilation"]
            ceilMode = input_data.kwargs["ceilMode"]
            m = torch.nn.MaxPool3d(kernel_size=kernelSize, stride=stride, padding=padding, dilation=dilation,
                                   return_indices=True, ceil_mode=ceilMode)
            out, indices = m(x.cpu().float())
            input_data.kwargs['indices'] = indices.to(torch.int32).npu()
