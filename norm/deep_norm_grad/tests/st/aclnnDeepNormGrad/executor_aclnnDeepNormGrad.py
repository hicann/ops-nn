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

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

try:
    import torch_npu
except Exception as e:
    logging.warning("import torch_npu failed!!!")

alpha = 0.3
eps = 1e-6

def deep_norm_foward(x1, x2, gamma, beta):
    d_dimension = gamma.shape
    reduce_axis = tuple([_*-1-1 for _ in range(len(d_dimension))])
    calc_x = x1
    calc_gx = x2
    calc_beta = beta
    calc_gamma = gamma

    x = calc_x * alpha + calc_gx
    mean = torch.mean(x, reduce_axis, keepdims=True)
    diff = x - mean
    variance = torch.mean(torch.pow(diff, 2), reduce_axis, keepdims=True)
    rstd = 1 / torch.sqrt(variance + eps)
    res = calc_gamma * (diff) * rstd + calc_beta

    if "bfloat16" in str(x1.dtype):
        y = res.type(torch.float32).to(torch.bfloat16)
    elif "float16" in str(x1.dtype):
        y = res.type(torch.float32).to(torch.float16)
    else:
        y = res.type(torch.float32)
    mean = mean.type(torch.float32)
    rstd = rstd.type(torch.float32)

    return y, mean, rstd

def deep_norm_backward(grad_y, x1, x2, mean, rstd, gamma):
    input_shape = x1.shape
    d_dimension = gamma.shape
    reduce_axis = tuple([_*-1-1 for _ in range(len(d_dimension))])
    axis = tuple([_ for _ in range(len(input_shape) - len(d_dimension))])

    dy_hp = grad_y
    rstd_hp = rstd
    mean_hp = mean
    gamma_hp = gamma    

    x_hp = alpha * x1.type(torch.float32) + x2.type(torch.float32)
    shape = x_hp.shape
    if len(shape) == 2:      # (n, d)
        n, d = shape
    elif len(shape) == 3:    # (n, m, d)
        n, _, d = shape
    elif len(shape) > 3:     # (n, m, k, ..., d)
        *batch_shape, d = shape
    # n, _, d = x_hp.shape

    pd_xl = dy_hp * gamma_hp
    x2_tensor = x_hp - mean_hp

    pd_var_first_part = (-0.5) * pd_xl * x2_tensor * torch.pow(rstd_hp, 3)
    pd_var = torch.sum(pd_var_first_part, reduce_axis, keepdims=True)

    pd_mean_first_part = torch.sum(((-1.0) * pd_xl * rstd_hp), reduce_axis, keepdims=True)
    pd_mean_second_part = torch.sum(((-2.0) * x2_tensor), reduce_axis, keepdims=True)
    pd_mean = pd_mean_first_part + pd_var * (1.0 / d) * pd_mean_second_part
    pd_x_first_part = pd_xl * rstd_hp
    pd_x_second_part = pd_var * (2.0 / d) * x2_tensor + pd_mean * (1.0 / d)
    pd_gx = pd_x_first_part + pd_x_second_part

    pd_x = alpha * pd_gx

    dgamma = torch.sum(dy_hp * x2_tensor * rstd_hp, axis=axis, keepdims=True)
    dbeta = torch.sum(dy_hp, axis=axis, keepdims=True)

    if "bfloat16" in str(x1.dtype):
        pd_x = pd_x.type(torch.float32).to(torch.bfloat16)
        pd_gx = pd_gx.type(torch.float32).to(torch.bfloat16)
    elif "float16" in str(x1.dtype):
        pd_x = pd_x.type(torch.float32).to(torch.float16)
        pd_gx = pd_gx.type(torch.float32).to(torch.float16)
    else:
        pd_x = pd_x.type(torch.float32)
        pd_gx = pd_gx.type(torch.float32)
    dgamma = dgamma.type(torch.float32)
    dbeta = dbeta.type(torch.float32)

    return pd_x, pd_gx, dgamma, dbeta
    
class DeepNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, gamma, beta):
        y, mean, rstd = deep_norm_foward(x1, x2, gamma, beta)
        ctx.save_for_backward(x1, x2, mean, rstd, gamma)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x1, x2, mean, rstd, gamma = ctx.saved_variables
        dx1, dx2, dgamma, dbeta = deep_norm_backward(grad_y, x1, x2, mean, rstd, gamma)
        return dx1, dx2, dgamma, dbeta
    
def compute_torch_golden(dy, x1, x2, beta, gamma):
    input_dtype = x1.dtype 
    if 'fp16' in str(input_dtype) or 'bf16' in str(input_dtype):
        tensor_x1 = x1.to(torch.float)
        tensor_x2 = x2.to(torch.float)
        tensor_gamma = gamma.to(torch.float)
        tensor_beta = beta.to(torch.float)
    else:
        tensor_x1 = x1
        tensor_x2 = x2
        tensor_gamma = gamma
        tensor_beta = beta

    tensor_x1.requires_grad = True
    tensor_x2.requires_grad = True
    tensor_gamma.requires_grad = True
    tensor_beta.requires_grad = True

    out = DeepNormFunction.apply(tensor_x1, tensor_x2, tensor_gamma, tensor_beta)
    grad_y = dy
    out.backward(grad_y)

    x1_cpu = tensor_x1.grad
    x2_cpu = tensor_x2.grad
    gamma_cpu = tensor_gamma.grad
    beta_cpu = tensor_beta.grad

    if 'bf16' in str(input_dtype) or 'fp16' in str(input_dtype):
        pd_x = x1_cpu.detach().clone().type(input_dtype)
        pd_gx = x2_cpu.detach().clone().type(input_dtype)
    else:
        pd_x = x1_cpu.detach().clone()
        pd_gx = x2_cpu.detach().clone()
    pd_gamma = gamma_cpu.detach().clone()
    pd_beta = beta_cpu.detach().clone()

    return pd_x, pd_gx, pd_gamma.type(torch.float).zero_(), pd_beta.type(torch.float).zero_()

    
@register("aclnn_deep_norm_grad")
# 没有pro的判断 也没有with_output
class DeepNormGradFunctionApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "npu":
           import torch_npu
           torch_npu.npu.set_compile_mode(jit_compile=False)

        dy = input_data.kwargs["dy"]
        calc_x = input_data.kwargs["x"]
        calc_gx = input_data.kwargs["gx"]
        calc_gamma = input_data.kwargs["gamma"]
        calc_beta = input_data.kwargs["beta"]

        # if self.output is None:
        if self.device == "cpu" or self.device == "gpu":
            return compute_torch_golden(dy, calc_x, calc_gx, calc_beta, calc_gamma)

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            input_x1 = input_data.kwargs["x"]
            input_x2 = input_data.kwargs["gx"]
            gamma = input_data.kwargs["gamma"]
            beta = input_data.kwargs["beta"]
            d_dimension = gamma.shape
            reduce_axis = tuple([_*-1-1 for _ in range(len(d_dimension))])
            eps = 1e-6
            x = input_x1 * alpha + input_x2
            mean = torch.mean(x, reduce_axis, keepdims=True)
            diff = x - mean
            variance = torch.mean(torch.pow(diff, 2), reduce_axis, keepdims=True)
            rstd = 1 / torch.sqrt(variance + eps)
            
            input_data.kwargs['mean'] = mean.type(torch.float)
            input_data.kwargs['rstd'] = rstd.type(torch.float)
            

@register("pyaclnn_deep_norm_grad")
class SampleAclnnApi(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def init_by_input_data(self, input_data: InputDataset):             
        input_args, output_packages = super().init_by_input_data(input_data)        
        # 把最后一个入参去除（默认处理会把标杆的输出拼到入参后，但实际上算子并无此参数）
        del input_args[3]
        return input_args, output_packages            
    
    def after_call(self, output_packages):
        output1, output2, output3, output4 = super().after_call(output_packages)
        output3.zero_()
        output4.zero_()
        return output1, output2, output3, output4