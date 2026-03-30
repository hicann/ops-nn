#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
        "kernel": {
            "quant_batch_matmul_v3": "quant_batch_matmul_v3_golden"
        }
}

import torch
import os
import numpy as np
from ml_dtypes import bfloat16

def quant_batch_matmul_v3_golden(x1, x2, scale, offset = None, bias = None, pertoken_scale = None, * ,dtype: int,
                                 transpose_x1: bool = False, transpose_x2: bool = False,
                                 group_size:int = 0, **kwargs):


    isAscend910B = False
    if kwargs['short_soc_version'] in ("Ascend910B", "Ascend910_93"):
        isAscend910B = True
    deq_scale = scale

    testcase_name = kwargs['testcase_name']

    x1_dtype = x1.dtype.name
    x2_dtype = x2.dtype.name
    deq_scale_dtype = deq_scale.dtype.name
    bias_dtype = None
    if bias is not None:
        bias_dtype = bias.dtype.name

    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]
    trans_a = transpose_x1
    trans_b = transpose_x2

    groups = group_size
    group_size_m, group_size_n, group_size_k = unpack_groupsize(groups)

    # mxfp4、mxfp8，输入先和scale相乘，再去做MMAD
    is_mxFP = pertoken_scale is not None and \
                x1_dtype in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2") and \
                deq_scale_dtype in ("float8_e8m0",)
    # fp8和hif8，pertoken_scale代表2路scale, 2路scale是用fixpipe中运算的，需要进行高19位与操作，否则会出现精度问题
    is_two_scale = pertoken_scale is not None and x1_dtype in ("float8_e4m3fn", "float8_e5m2", "hifloat8") and \
                   not is_mxFP and deq_scale.shape[0] == 1 and pertoken_scale.shape[0] == 1
    # int8，pertoken_scale代表pertoken，pertoken不是在fixpipe中运算的，无需高19位与操作
    is_pertoken = pertoken_scale is not None and x1_dtype in ('int8', "float8_e4m3fn", "float8_e5m2", "hifloat8") and not is_two_scale
    is_950_compatible = out_dtype == 'bfloat16' and x1_dtype == 'int8' and deq_scale_dtype != 'uint64' and pertoken_scale is None
    is_bias_vec = (x1_dtype == 'int8' or x1_dtype == "int4") and bias is not None and bias_dtype in ('bfloat16', "float16", "float32")
    is_perblock = x1_dtype in ("float8_e4m3fn", "float8_e5m2", "hifloat8") and deq_scale_dtype in ("float32",) and len(deq_scale.shape) > 1 and len(pertoken_scale.shape) > 1

    # mxFP4
    if is_mxFP:
        deq_scale_mx = deq_scale
        pertoken_scale_mx = pertoken_scale
        if x1_dtype in ("float8_e4m3fn", "float8_e5m2"):
            x1 = torch.from_numpy(x1.astype(np.float32))

        if x2_dtype in ("float8_e4m3fn", "float8_e5m2"):
            x2 = torch.from_numpy(x2.astype(np.float32))
        # mxFP4单独处理transpose，统一使用(M,K)和(K,N)格式处理mxFP4
        if trans_a:
            x1 = np.swapaxes(x1, -1, -2)
            pertoken_scale_mx = np.swapaxes(pertoken_scale_mx, -1, -2)
            if len(pertoken_scale_mx.shape) == 3:
                pertoken_scale_mx = pertoken_scale_mx.reshape(pertoken_scale_mx.shape[0] * pertoken_scale_mx.shape[1], pertoken_scale_mx.shape[2])
            pertoken_scale_mx = np.swapaxes(pertoken_scale_mx, -1, -2)
        else:
            if len(pertoken_scale_mx.shape) == 3:
                pertoken_scale_mx = pertoken_scale_mx.reshape(pertoken_scale_mx.shape[0], pertoken_scale_mx.shape[1] * pertoken_scale_mx.shape[2])
        if trans_b:
            x2 = np.swapaxes(x2, -1, -2)
            if len(deq_scale_mx.shape) == 3:
                deq_scale_mx = deq_scale_mx.reshape(deq_scale_mx.shape[0], deq_scale_mx.shape[1] * deq_scale_mx.shape[2])
            deq_scale_mx = np.swapaxes(deq_scale_mx, -1, -2)
        else:
            deq_scale_mx = np.swapaxes(deq_scale_mx, -1, -2)
            if len(deq_scale_mx.shape) == 3:
                deq_scale_mx = deq_scale_mx.reshape(deq_scale_mx.shape[0] * deq_scale_mx.shape[1], deq_scale_mx.shape[2])

        k_dim = x1.shape[-1]
        if ceil_div(k_dim, 32) % 2 != 0:
            pertoken_scale_mx = pertoken_scale_mx[:, :-1]
            deq_scale_mx = deq_scale_mx[:-1, :]

        # broadcast，每个数对应32个数
        pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_mx, 32, axis=-1)
        deq_scale_mx_broadcast = np.repeat(deq_scale_mx, 32, axis=-2)
        x1_dims = len(x1.shape)
        x2_dims = len(x2.shape)
        x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]
        x2_pad_len = deq_scale_mx_broadcast.shape[-2] - x2.shape[-2]
        x1 = np.pad(x1, [(0, 0)] * (x1_dims -1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
        x2 = np.pad(x2, [(0, 0)] * (x2_dims -2) + [(0, x2_pad_len)] + [(0, 0)], mode='constant', constant_values=0)

        x1 = x1 * pertoken_scale_mx_broadcast
        x2 = x2 * deq_scale_mx_broadcast

    # 升精度 & 转torch
    if x1_dtype in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8"):
        x1 = torch.from_numpy(x1.astype(np.float32))
    elif x1_dtype in ("int4",):
        x1 = torch.from_numpy(x1.astype(np.int32)).to(torch.int32)
    else:
        x1 = torch.from_numpy(x1).to(torch.int32)
    if x2_dtype in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1", "float4_e1m2", "hifloat8"):
        x2 = torch.from_numpy(x2.astype(np.float32))
    elif x2_dtype in ("int4",):
        x2 = torch.from_numpy(x2.astype(np.int32)).to(torch.int32)
    else:
        x2 = torch.from_numpy(x2).to(torch.int32)

    if not (is_mxFP):
        if trans_a:
            array_trans = gen_axes_for_transpose(len(x1.shape) - 2, [1, 0])
            x1 = x1.permute(*array_trans)
        if trans_b:
            array_trans = gen_axes_for_transpose(len(x2.shape) - 2, [1, 0])
            x2 = x2.permute(*array_trans)

    if not is_perblock:
        out = torch.matmul(x1, x2)
    else:
        pass

    if is_mxFP: # mxFP不做scale处理
        pass
    elif deq_scale_dtype == "uint64":
        deq_scale = np.load(testcase_name + "_deq_scale.npy")
        deq_scale_tensor = torch.from_numpy(deq_scale)
    else:
        deq_scale = deq_scale.astype(np.float32)
        deq_scale_tensor = torch.from_numpy(deq_scale)

    if offset is not None:
        offset = np.load(testcase_name + "_offset.npy")
        offset = torch.from_numpy(offset)
        if deq_scale_tensor.shape[-1] < offset.shape[-1]:
            offset = offset[0]

    if bias is not None and bias_dtype == "int32":
        bias = torch.from_numpy(bias).to(torch.int32)
        out = torch.add(out, bias)

    if is_perblock:
        pertoken_scale_tensor = torch.from_numpy(pertoken_scale).to(torch.float32)

        if trans_a:
                array_trans = gen_axes_for_transpose(len(pertoken_scale_tensor.shape) - 2, [1, 0])
                pertoken_scale_tensor = pertoken_scale_tensor.permute(*array_trans)
        if trans_b:
                array_trans = gen_axes_for_transpose(len(deq_scale_tensor.shape) - 2, [1, 0])
                deq_scale_tensor = deq_scale_tensor.permute(*array_trans)

        batch_x1_shape = x1.shape[:-2]
        batch_x2_shape = x2.shape[:-2]
        batch_x1 = [int(i) for i in batch_x1_shape]
        batch_x2 = [int(i) for i in batch_x2_shape]
        import copy
        batch_out = copy.deepcopy(batch_x1) if len(batch_x1) > len(batch_x2)  else copy.deepcopy(batch_x2)
        if batch_x2 != batch_x1 and len(batch_x1) != 0 and len(batch_x2) != 0:
            idx= 1
            for item in reversed(batch_out):
                if idx <= len(batch_x1) and idx <= len(batch_x2):
                    batch_out[-idx] = max(batch_x1[-idx], batch_x2[-idx])
                elif idx <= len(batch_x1):
                    batch_out[-idx] = batch_x1[-idx]
                else:
                    batch_out[-idx] = batch_x2[-idx]
                idx = idx+1
        if batch_x1 != batch_out:
            new_x1_shape = batch_out + list(x1.shape[-2:])
            x1 = torch.broadcast_to(x1, new_x1_shape)
            new_s_shape = batch_out + list(pertoken_scale_tensor.shape[-2:])
            pertoken_scale_tensor = torch.broadcast_to(pertoken_scale_tensor, new_s_shape)

        if batch_x2 != batch_out:
            new_x2_shape = batch_out + list(x2.shape[-2:])
            x2 = torch.broadcast_to(x2, new_x2_shape)
            new_s_shape = batch_out + list(deq_scale_tensor.shape[-2:])
            deq_scale_tensor = torch.broadcast_to(deq_scale_tensor, new_s_shape)

        # batch轴合轴
        batch_all = 1
        if batch_out:
            batch_all = np.prod(batch_out)
            x1 = x1.reshape([batch_all] + list(x1.shape[-2:]))
            x2 = x2.reshape([batch_all] + list(x2.shape[-2:]))
            pertoken_scale_tensor = pertoken_scale_tensor.reshape([batch_all] + list(pertoken_scale_tensor.shape[-2:]))
            deq_scale_tensor = deq_scale_tensor.reshape([batch_all] + list(deq_scale_tensor.shape[-2:]))


    # Ascend_910B新增biasDtype float32 float16,当Ascend_910B时，不执行这段代码
    if not isAscend910B and not is_950_compatible and not is_bias_vec:
        # fp8 hif8 的bias 处理
        if bias is not None and bias_dtype == "float32" and not (is_mxFP):
            bias = torch.from_numpy(bias).to(torch.float32)
            out = torch.add(out, bias)
    if is_mxFP:
        if bias is not None:
            bias = torch.from_numpy(bias.astype(np.float32))
            out = (out + bias).numpy().astype(out_dtype)
        else:
            out = out.numpy().astype(out_dtype)
    elif is_perblock:
        # 已预处理：x1,x2,deq_scale和pertoken_scale全为非转置

        m = x1.shape[-2] # 非转置情况下，m是倒数第二维
        k = x1.shape[-1] # 非转置情况下，k是倒数第一维
        n = x2.shape[-1] # 非转置情况下，n是倒数第一维
        out = torch.zeros(m, n)
        if pertoken_scale_tensor.dim() > 2 or deq_scale_tensor.dim() > 2:
            out = torch.zeros(batch_all, m, n)
        pertoken_scale_tensor_m = torch.repeat_interleave(pertoken_scale_tensor, repeats=group_size_m, dim=-2)
        pertoken_scale_tensor_m = pertoken_scale_tensor_m[..., :m, :]
        deq_scale_tensor_n = torch.repeat_interleave(deq_scale_tensor, repeats=group_size_n, dim=-1)
        deq_scale_tensor_n = deq_scale_tensor_n[..., :n]
        if pertoken_scale_tensor.dim() > 2 or deq_scale_tensor.dim() > 2:
            for i in range(batch_all):
                for k_idx in range((k + group_size_k - 1) // group_size_k):
                    k_start = k_idx * group_size_k
                    k_end = min((k_idx + 1) * group_size_k, k)
                    scale_mul = pertoken_scale_tensor_m[i, :, k_idx].unsqueeze(1) * deq_scale_tensor_n[i, k_idx, :].unsqueeze(0)
                    out[i] += torch.matmul(x1[i, :, k_start:k_end], x2[i, k_start:k_end, :]) * scale_mul
        else:
            for k_idx in range((k + group_size_k - 1) // group_size_k):
                k_start = k_idx * group_size_k
                k_end = min((k_idx + 1) * group_size_k, k)
                scale_mul = pertoken_scale_tensor_m[:, k_idx].unsqueeze(1) * deq_scale_tensor_n[k_idx, :].unsqueeze(0)
                out += torch.matmul(x1[:, k_start:k_end], x2[k_start:k_end, :]) * scale_mul
        if out_dtype == 'bfloat16':
            out_dtype = bfloat16
        out = (out).numpy().astype(out_dtype) # 暂时只支持fp16/bf16/fp32

    else:
        if out_dtype == 'int8':
            out = f32_2_s9(out * deq_scale_tensor)
            if offset is not None:
                out = f32_2_s9(out) + f32_2_s9(offset)
            out = np.clip(out, -128, 127).numpy().astype(out_dtype)
        elif out_dtype == 'bfloat16':
            output_dtype = bfloat16
            if is_pertoken:
                pertoken_scale_slice = torch.unsqueeze(torch.from_numpy(pertoken_scale), dim=1).to(torch.float32)
                out = out * deq_scale_tensor * pertoken_scale_slice
            elif is_two_scale:
                two_scale = scale_generate(pertoken_scale * deq_scale)
                two_scale_tensor = torch.unsqueeze(torch.from_numpy(two_scale), dim=1).to(torch.float32)
                out = out * two_scale_tensor
            elif is_950_compatible:
                if deq_scale.shape[0] == 1 and (bias_dtype != "bfloat16" and bias_dtype != "float32"):
                    scale = scale_generate(deq_scale)
                    deq_scale_tensor = torch.unsqueeze(torch.from_numpy(scale), dim=1).to(torch.float32)
                out = out * deq_scale_tensor
            else:
                out = out * deq_scale_tensor
            # Ascend_910B新增biasDtype float32 float16, 这段代码扩展 or bias_dtype == "float32"
            if (isAscend910B or is_950_compatible) and bias is not None and (bias_dtype == "bfloat16" or bias_dtype == "float32"):
                bias_fp32 = torch.from_numpy(bias.astype(np.float32))
                out = (out + bias_fp32).numpy().astype(output_dtype)
            elif is_bias_vec:
                bias_fp32 = torch.from_numpy(bias.astype(np.float32))
                out = (out + bias_fp32).numpy().astype(output_dtype)
            else:
                out = out.numpy().astype(output_dtype)
        elif out_dtype == 'float16':
            if is_pertoken:
                pertoken_scale_slice = torch.unsqueeze(torch.from_numpy(pertoken_scale), dim=1).to(torch.float32)
                out = out * deq_scale_tensor * pertoken_scale_slice
            elif is_two_scale:
                two_scale = scale_generate(pertoken_scale * deq_scale)
                two_scale_tensor = torch.unsqueeze(torch.from_numpy(two_scale), dim=1).to(torch.float32)
                out = out * two_scale_tensor
            else:
                out = (out * deq_scale_tensor)
            # Ascend_910B新增biasDtype float32 float16
            if (isAscend910B and bias is not None and (bias_dtype == "float16" or bias_dtype == "float32")) or is_bias_vec:
                bias_fp32 = torch.from_numpy(bias.astype(np.float32))
                out = (out + bias_fp32).numpy().astype(out_dtype)
            else:
                out = out.numpy().astype(out_dtype)
        elif out_dtype == 'hifloat8':
            out = (out * deq_scale_tensor).numpy().astype(out_dtype)
        elif out_dtype == 'float8_e4m3fn':
            out = (out * deq_scale_tensor).numpy().astype(out_dtype)
        elif out_dtype == 'float32':
            if is_pertoken:
                pertoken_scale_slice = torch.unsqueeze(torch.from_numpy(pertoken_scale), dim=1).to(torch.float32)
                out = (out * deq_scale_tensor * pertoken_scale_slice).numpy().astype(out_dtype)
            elif is_two_scale:
                two_scale = scale_generate(pertoken_scale * deq_scale)
                two_scale_tensor = torch.unsqueeze(torch.from_numpy(two_scale), dim=1).to(torch.float32)
                out = (out * two_scale_tensor).numpy().astype(out_dtype)
            else:
                out = (out * deq_scale_tensor).numpy().astype(out_dtype)
        elif out_dtype == 'int32':
                out = out.numpy().astype(out_dtype)
        else:
            print("Please check whether this dtype '{out_dtype}' is supported")

    if os.path.exists(testcase_name + "_deq_scale.npy"):
        os.remove(testcase_name + "_deq_scale.npy")
    if os.path.exists(testcase_name + "_offset.npy"):
        os.remove(testcase_name + "_offset.npy")
    return out

def unpack_groupsize(group_size):
    group_size_M = (group_size >> 32) & 0xFFFF
    group_size_N = (group_size >> 16) & 0xFFFF
    group_size_K = group_size & 0xFFFF
    # 当前只有0和1的情况，后续需要推导再改
    if group_size_M == 0:
        group_size_M = 1
    if group_size_N == 0:
        group_size_N = 1
    return group_size_M, group_size_N, group_size_K

def ceil_div(a, b):
    return (a + b - 1) // b

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def f32_2_s9(array):
    array_round = np.round(array)
    array_round_clip = np.clip(array_round, -256, 255)
    return array_round_clip

def scale_generate(fp32_deq_scale):
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    #与高19位运算，模拟硬件
    uint32_deq_scale &= 0XFFFFE000
    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)

    return fp32_deq_scale