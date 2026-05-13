#!/usr/bin/env python3
# -- coding: utf-8 --
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
from dataclasses import dataclass
import numpy as np
import math
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@dataclass
class AdamWParams:
    param: any
    exp_avg: any
    exp_avg_sq: any
    grad: any
    step_t: any
    max_exp_avg_sq: any
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    amsgrad: bool
    maximize: bool

def single_tensor_adamw(adamw_params: AdamWParams):
    param, exp_avg, exp_avg_sq, grad, step_t, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, amsgrad, maximize =\
        adamw_params.param, adamw_params.exp_avg, adamw_params.exp_avg_sq, adamw_params.grad, adamw_params.step_t,\
        adamw_params.max_exp_avg_sq, adamw_params.lr, adamw_params.beta1, adamw_params.beta2, adamw_params.eps,\
        adamw_params.weight_decay, adamw_params.amsgrad, adamw_params.maximize
    dtype1 = param.dtype
    dtype2 = grad.dtype

    lr = (np.float32)(lr)
    beta1 = (np.float32)(beta1)
    beta2 = (np.float32)(beta2)
    weight_decay = (np.float32)(weight_decay)
    eps = (np.float32)(eps)

    if dtype1 != dtype2:
        grad = grad.to(dtype1)
        max_exp_avg_sq = max_exp_avg_sq.to(dtype1)
    if maximize:
        grad = -grad

    # update step, 接口内部step+1了, 所以pta接口调aclnn前step-1, 如过看护pta的话这里不+1
    step = step_t
    step = step.item() + 1
    
    # Perform stepweight decay
    param = param * (1 - lr * weight_decay)

    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = lr / bias_correction1
    bias_correction2_sqrt = math.sqrt(bias_correction2)
    if amsgrad:
        torch.maximum(max_exp_avg_sq, exp_avg_sq, out = max_exp_avg_sq)
        denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
    else:
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

    param.addcdiv_(exp_avg, denom, value = -step_size)

    if dtype1 != dtype2:
        max_exp_avg_sq = max_exp_avg_sq.to(dtype2)
    return param, exp_avg, exp_avg_sq, max_exp_avg_sq

@dataclass
class AdamWParameters:
    var: any
    m: any
    v: any
    max_grad_norm: any
    grad: any
    step: any
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    amsgrad: bool
    maximize: bool

def apply_adam_w_v2(params: AdamWParameters):
    var_dtype, m_dtype, v_dtype, grad_dtype, step_dtype, max_grad_norm_dtype =\
        params.var.dtype, params.m.dtype, params.v.dtype, params.grad.dtype,\
        params.step.dtype, params.max_grad_norm.dtype
    var = params.var
    m =params.m
    v = params.v
    grad = params.grad
    step = params.step
    max_grad_norm = params.max_grad_norm
    lr = params.lr
    beta1 = params.beta1
    beta2 = params.beta2
    weight_decay = params.weight_decay
    eps = params.eps
    amsgrad = params.amsgrad
    maximize = params.maximize

    # 混合精度运算, var f16/bf16 ->全升f32 ->否则判断gradf16/bf16 -> 仅grad和max_grad_norm升f32
    if "bfloat16" in str(var_dtype) or "float16" in str(var_dtype):
        var, m, v, grad, max_grad_norm = var.to(torch.float32), m.to(torch.float32),\
            v.to(torch.float32), grad.to(torch.float32), max_grad_norm.to(torch.float32)
    elif "bfloat16" in str(grad_dtype) or "float16" in str(grad_dtype):
        grad, max_grad_norm = grad.to(torch.float32), max_grad_norm.to(torch.float32)
    
    if "int64" in str(step_dtype):
        step =step.to(torch.float32)
    
    adamw_params = AdamWParams(
        param = var,
        exp_avg = m,
        exp_avg_sq = v,
        grad = grad,
        step_t = step,
        max_exp_avg_sq = max_grad_norm,
        lr = lr,
        beta1 = beta1,
        beta2 = beta2,
        weight_decay = weight_decay,
        eps = eps,
        amsgrad = amsgrad,
        maximize = maximize
    )

    res_var, res_m, res_v, res_max_grad_norm = single_tensor_adamw(adamw_params)

    # 混合精度计算,cast回目标精度
    if "bfloat16" in str(var_dtype) or "float16" in str(var_dtype):
        res_var, res_m, res_v, res_max_grad_norm = res_var.to(var_dtype), res_m.to(var_dtype),\
            res_v.to(var_dtype), res_max_grad_norm.to(var_dtype)
    elif "bfloat16" in str(grad_dtype) or "float16" in str(grad_dtype):
        res_max_grad_norm = res_max_grad_norm.to(grad_dtype)
    return res_var, res_m, res_v,res_max_grad_norm

@register("executor_apply_adam_w_v2")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
            params = AdamWParameters(
                var = input_data.kwargs["varRef"],
                m = input_data.kwargs["mRef"], 
                v = input_data.kwargs["vRef"],
                max_grad_norm = input_data.kwargs["maxGradNormOptionalRef"],
                grad = input_data.kwargs["grad"],
                step = input_data.kwargs["step"],
                lr = input_data.kwargs["lr"],
                beta1 = input_data.kwargs["beta1"],
                beta2 = input_data.kwargs["beta2"],
                weight_decay = input_data.kwargs["weightDecay"],
                eps = input_data.kwargs["eps"],
                amsgrad = input_data.kwargs["amsgrad"],
                maximize = input_data.kwargs["maximize"]
            )
            var_ref, m_ref, v_ref, max_grad_norm_optional_ref = apply_adam_w_v2(params)
            return var_ref, m_ref, v_ref, max_grad_norm_optional_ref

@register("aclnn_apply_adam_w_v2")
class AdamWV2AclnnApi(AclnnBaseApi):
    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        input_args.pop()
        input_args.pop()
        input_args.pop()
        output_packages[:] = [input_args[0], input_args[1], input_args[2], input_args[3]]
        return input_args, output_packages