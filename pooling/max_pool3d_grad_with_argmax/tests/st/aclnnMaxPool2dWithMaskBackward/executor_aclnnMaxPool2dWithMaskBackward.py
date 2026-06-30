# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved. 
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
import numpy as np
import torch_npu
import math

@register("ascend_method_torch_nn_MaxPool2dWithMaskBackward")
class MethodTorchNnMaxPool2DWithMaskBackwardApi(BaseApi):
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
            m = torch.nn.MaxPool2d(kernel_size=kernelSize, stride=stride, padding=padding,
                                   dilation=dilation, return_indices=True, ceil_mode=ceilMode)
            ori_type = x.dtype
            x_float = x.to(torch.float)
            gradOut_float = gradOut.to(torch.float)
            out, indices = m(x_float)
            
            gradIn = torch.ops.aten.max_pool2d_with_indices_backward(gradOut_float, x_float, kernelSize, stride,
                                                                     padding, dilation, ceilMode, indices.to(torch.int64))
            return gradIn.to(ori_type)
        return gradIn

    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'pyaclnn':
            x = input_data.kwargs["self"]
            kernelSize = input_data.kwargs["kernelSize"]
            stride = input_data.kwargs["stride"]
            padding = input_data.kwargs["padding"]
            dilation = input_data.kwargs["dilation"]
            ceilMode = input_data.kwargs["ceilMode"]
            x_cpu = x.cpu()
            m = torch.nn.MaxPool2d(kernel_size=kernelSize, stride=stride, padding=padding, dilation=dilation,
                                        return_indices=True, ceil_mode=ceilMode)
            out_cpu, indices_cpu = m(x_cpu.float())
            n = indices_cpu.shape[0]
            c = indices_cpu.shape[1]
            ho = indices_cpu.shape[2]
            wo = indices_cpu.shape[3]
            kh = kernelSize[0]
            kw = kernelSize[1]
            indices_npu_shape = [n, c, kh * kw, (math.ceil(ho * wo / 16) + 1) * 32]
            pad_size = n * c * kh * kw * (math.ceil(ho * wo / 16) + 1) * 8 - indices_cpu.numel()
            flatten_indices_cpu = indices_cpu.to(torch.int32).flatten()
            zero_padding = torch.zeros(int(pad_size), dtype=torch.int32)
            indices_npu_cat = torch.cat((flatten_indices_cpu, zero_padding)).view(torch.int8).reshape(indices_npu_shape).npu()
            input_data.kwargs['indices'] = indices_npu_cat