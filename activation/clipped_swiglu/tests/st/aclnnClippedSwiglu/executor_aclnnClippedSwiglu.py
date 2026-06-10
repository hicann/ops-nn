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
import ctypes
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("aclnn_clipped_swiglu")
class FunctionAclnnClippedSwiglu(AclnnBaseApi):

    def init_by_input_data(self, input_data):
        import random
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

        input_args, output_packages = super().init_by_input_data(input_data)
        input_x = self.backend.acl_tensor_to_torch(input_args[0])
        if type(input_args[1]).__name__ != "c_void_p":
            group_index = self.backend.acl_tensor_to_torch(input_args[1])
        dim = input_args[2].value
        
        self.dim = dim if dim >= 0 else dim + input_x.dim() 
        shape = input_x.shape   # 合轴前的shape
        dim1 = 1
            
        # 计算合轴后的shape
        if input_x.ndim > 1 :
            if dim != 0:
                dim1 = torch.prod(torch.tensor(shape[:dim])).item()
                dim2 = torch.prod(torch.tensor(shape[dim:])).item()
                self.after_shape = [int(dim1), int(dim2) // 2]
            else:
                self.after_shape = [int(1), int(torch.prod(torch.tensor(shape[dim:])).item()) // 2]
        else:
            self.after_shape = [int(1), shape[0] // 2]
        if type(input_args[1]).__name__ == "c_void_p":
            AclTensorPtr = ctypes.POINTER(AclTensor)
            null_void_ptr = ctypes.c_void_p(None)
            null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)
            input_args[1] = null_tensor_ptr
            self.groupsum = None
        else:
            self.groupsum = int(min(torch.sum(group_index).item(), dim1))
        self.res_shape = list(shape)
        self.res_shape[dim] = self.res_shape[dim] // 2

        return input_args, output_packages

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        torch_tensor = output[0]
        res_shape = torch_tensor.shape
        if self.groupsum is not None:    
            torch_tensor = torch_tensor.reshape(self.after_shape) 
            torch_tensor[self.groupsum:] = 0
            # 消除合轴
            torch_tensor = torch_tensor.reshape(res_shape) 
            
        return output

@register("function_clipped_swiglu")
class FunctionClippedSwiglu(BaseApi):
    
    def clipped_swiglu_golden(self, input_x, input_group_index, dim=-1, alpha=1.702, limit=7.0, 
                        bias=1.0, interleaved=True):
        dim = dim if dim >= 0 else dim + input_x.dim()
        dtype = input_x.dtype
        if input_x.dtype in [torch.bfloat16, torch.float16]:
            input_x = input_x.to(torch.float32)
        shape = input_x.shape    
        if input_x.ndim > 1 :
            if dim != 0:
                dim1 = torch.prod(torch.tensor(shape[:dim])).item()
                dim2 = torch.prod(torch.tensor(shape[dim:])).item()
                x = input_x.reshape(dim1, dim2).clone()
            else:
                x = input_x.reshape(1, torch.prod(torch.tensor(shape[dim:])).item()).clone()
        else:
            x = input_x.reshape(1, shape[0]).clone()
        group = x.shape[0] 
        if input_group_index is not None:
            group = min(torch.sum(input_group_index).item(), x.shape[0])
        x_tensor = x[:group]
        remain_tensor = torch.zeros_like(x[group:, :x.shape[1]//2])
        if interleaved:
            x_glu = x_tensor[..., ::2]
            x_linear = x_tensor[..., 1::2] 
        else:
            out = torch.chunk(x_tensor, 2, dim=-1)
            x_glu = out[0]
            x_linear = out[1]
        x_glu = x_glu.clamp(min=None, max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        sigmoid_part = torch.sigmoid(alpha * x_glu)
        result = x_glu * sigmoid_part * (x_linear + bias)
        result = torch.cat((result, remain_tensor), dim=0)
        res_shape = list(input_x.shape)
        res_shape[dim] = res_shape[dim] // 2

        if result.numel() != 0:
            result = result.reshape(res_shape)
        if dtype == torch.bfloat16:
            result = result.to(torch.bfloat16)
        elif dtype == torch.float16:
            result = result.to(torch.float16)
        return result

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs['x']
        input_group_index = input_data.kwargs['group_index']
        dim = input_data.kwargs['dim']
        alpha = input_data.kwargs['alpha']
        limit = input_data.kwargs['limit']
        bias = input_data.kwargs['bias']
        interleaved = input_data.kwargs['interleaved']
        y = self.clipped_swiglu_golden(input_x, input_group_index, dim, alpha, limit, bias, interleaved)
        return y