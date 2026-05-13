#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from dataclasses import dataclass
import numpy as np
import math
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@dataclass
class AdamWQuantParams:
    varRef: any
    grad: any
    mRef: int
    vRef: int
    qmapM: float
    qmapV: float
    absmaxMRef: float
    absmaxVRef: float
    step_t: int

    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    gnorm_scale: float
    quant_mode: str
    block_size: int

def quantize_state(state, qmap, absmax):
    normalized = state / absmax
    distances = torch.abs(normalized[:, None] - qmap[None, :])
    indices = torch.argmin(distances, axis=1)

    if torch.any(qmap < 0):
        quantiled = qmap[indices]
        sign_diff = torch.not_equal(torch.sign(state), torch.sign(quantiled))
        adjusted_indices = indices.clone()
        adjusted_indices[sign_diff & (state > 0)] += 1
        adjusted_indices[sign_diff & (state < 0)] -= 1
        indices = torch.clip(adjusted_indices, 0, len(qmap) - 1)

    return indices.to(torch.uint8)

def single_tensor_adamwquant(adamwquant_params: AdamWQuantParams):
    var_ref, grad, state_m, state_v, absmax_m, absmax_v, qmap_m, qmap_v,\
        step, lr, beta1, beta2, eps, weight_decay, gnorm_scale, block_size = adamwquant_params.varRef,\
        adamwquant_params.grad, adamwquant_params.mRef, adamwquant_params.vRef, adamwquant_params.absmaxMRef,\
        adamwquant_params.absmaxVRef, adamwquant_params.qmapM, adamwquant_params.qmapV, adamwquant_params.step_t,\
        adamwquant_params.lr, adamwquant_params.beta1, adamwquant_params.beta2, adamwquant_params.eps,\
        adamwquant_params.weight_decay, adamwquant_params.gnorm_scale, adamwquant_params.block_size
    # print(f"jisuan:{qmap_m}")

    step += 1

    bias_correction1 = 1 - beta1 ** step
    bias_correction2_sqrt = torch.sqrt(1 - beta2 ** step)
    step_size = -lr * bias_correction2_sqrt / bias_correction1

    params_flat = var_ref.flatten()
    grads_flat = grad.flatten()
    state_m_flat = state_m.flatten()
    state_v_flat = state_v.flatten()

    num_blocks = (params_flat.numel() + block_size - 1) // block_size

    new_absmax_m = torch.zeros_like(absmax_m)
    new_absmax_v = torch.zeros_like(absmax_v)

    for block_idx in range(num_blocks):
        start_idx = block_idx * block_size
        end_idx = min((block_idx + 1) * block_size, params_flat.numel())

        p_block = params_flat[start_idx:end_idx]
        g_block = grads_flat[start_idx:end_idx]
        s1_block = state_m_flat[start_idx:end_idx]
        s2_block = state_v_flat[start_idx:end_idx]

        s1_dequant = qmap_m[s1_block.to(torch.int32)] * absmax_m[block_idx]
        s2_dequant = qmap_v[s2_block.to(torch.int32)] * absmax_v[block_idx]

        g_block_scaled = g_block * gnorm_scale

        s1_update = s1_dequant * beta1 + (1 - beta1) * g_block_scaled

        s2_update = s2_dequant * beta2 + (1 - beta2) * (g_block_scaled ** 2)

        denom = torch.sqrt(s2_update) + eps * bias_correction2_sqrt
        p_block += step_size * s1_update / denom

        if weight_decay > 0:
            p_block *= (1 - lr * weight_decay)

        new_absmax1 = torch.max(torch.abs(s1_update))
        new_absmax2 = torch.max(s2_update)

        s1_quantized = quantize_state(s1_update, qmap_m, new_absmax1)
        s2_quantized = quantize_state(s2_update, qmap_v, new_absmax2)

        params_flat[start_idx:end_idx] = p_block
        state_m_flat[start_idx:end_idx] = s1_quantized
        state_v_flat[start_idx:end_idx] = s2_quantized
        new_absmax_m[block_idx] = new_absmax1
        new_absmax_v[block_idx] = new_absmax2

    updated_params = params_flat.reshape(var_ref.shape)
    updated_m = state_m_flat.reshape(state_m.shape)
    updated_v = state_v_flat.reshape(state_v.shape)

    return updated_params, updated_m, updated_v, new_absmax_m, new_absmax_v

@dataclass
class AdamWQuantParameters:
    varRef: any
    grad: any
    mRef: any
    vRef: any
    qmapM: any
    qmapV: any
    absmaxMRef: any
    absmaxVRef: any
    step_t: any
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    eps: float
    gnorm_scale: float
    quant_mode: str
    block_size: int

def adamw_8bit_blockwise_step(params: AdamWQuantParameters):
    varRef_dtype, grad_dtype, mRef_dtype, vRef_dtype, qmapM_dtype, qmapV_dtype,\
        absmaxMRef_dtype, absmaxVRef_dtype, step_dtype = params.varRef.dtype, params.grad.dtype,\
        params.mRef.dtype, params.vRef.dtype, params.qmapM.dtype, params.qmapV.dtype,\
        params.absmaxMRef.dtype, params.absmaxVRef.dtype, params.step_t.dtype
    varRef = params.varRef
    grad = params.grad
    mRef = params.mRef
    vRef = params.vRef
    qmapM = params.qmapM
    qmapV = params.qmapV
    absmaxMRef = params.absmaxMRef
    absmaxVRef = params.absmaxVRef
    step_t = params.step_t
    lr = params.lr
    beta1 = params.beta1
    beta2 = params.beta2
    weight_decay = params.weight_decay
    eps = params.eps
    gnorm_scale = params.gnorm_scale
    quant_mode = params.quant_mode
    block_size = params.block_size
    # 混合精度运算, varRef 和grad f16/bf16 ->全升f32
    if "bfloat16" in str(varRef_dtype) or "float16" in str(varRef_dtype):
        varRef, grad = varRef.to(torch.float32), grad.to(torch.float32)

    if "int64" in str(step_dtype):
        step_t =step_t.to(torch.float32)

    adamwquant_params = AdamWQuantParams(
        varRef = varRef,
        grad = grad,
        mRef = mRef,
        vRef = vRef,
        qmapM = qmapM,
        qmapV = qmapV,
        absmaxMRef = absmaxMRef,
        absmaxVRef = absmaxVRef,
        step_t = step_t,
        lr = lr,
        beta1 = beta1,
        beta2 = beta2,
        weight_decay = weight_decay,
        eps = eps,
        gnorm_scale = gnorm_scale,
        quant_mode = quant_mode,
        block_size = block_size
    )

    updated_params, updated_m, updated_v, new_absmax_m, new_absmax_v = single_tensor_adamwquant(adamwquant_params)

    if "bfloat16" in str(varRef_dtype) or "float16" in str(varRef_dtype):
        updated_params = updated_params.to(varRef_dtype)
    
    return updated_params, updated_m, updated_v, new_absmax_m, new_absmax_v

@register("executor_apply_adam_w_quant")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

            params = AdamWQuantParameters(
                varRef = input_data.kwargs["varRef"],
                grad = input_data.kwargs["grad"],
                mRef = input_data.kwargs["mRef"], 
                vRef = input_data.kwargs["vRef"],
                qmapM = input_data.kwargs["qmapM"],
                qmapV = input_data.kwargs["qmapV"],
                absmaxMRef = input_data.kwargs["absmaxMRef"],
                absmaxVRef = input_data.kwargs["absmaxVRef"],
                step_t = input_data.kwargs["step"],
                lr = input_data.kwargs["lr"],
                beta1 = input_data.kwargs["beta1"],
                beta2 = input_data.kwargs["beta2"],
                weight_decay = input_data.kwargs["weightDecay"],
                eps = input_data.kwargs["eps"],
                gnorm_scale = input_data.kwargs["gnormScale"],
                quant_mode = input_data.kwargs["quantMode"],
                block_size = input_data.kwargs["blockSize"]
            )
            
            updated_params, updated_m, updated_v, new_absmax_m, new_absmax_v = adamw_8bit_blockwise_step(params)

            return updated_params

    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        # print("before----------------------------------------------------")
        # print(input_data.kwargs['qmapM'])
        # print("before----------------------------------------------------")
        qmap_m_customize = torch.linspace(-1, 1, steps=256, dtype=torch.float32)
        qmap_v_customize = torch.linspace(0, 1, steps=256, dtype=torch.float32)
        if self.device == 'pyaclnn':
            input_data.kwargs['qmapM'] = qmap_m_customize.npu()
            input_data.kwargs['qmapV'] = qmap_v_customize.npu()
        else:
            input_data.kwargs['qmapM'] = qmap_m_customize.cpu()
            input_data.kwargs['qmapV'] = qmap_v_customize.cpu()
        # print("----------------------------------------------------")
        # print(input_data.kwargs['qmapM'])
        # print("----------------------------------------------------")

@register("aclnn_apply_adam_w_quant")
class ApplyAdamWQuantAclnnApi(AclnnBaseApi):
    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages