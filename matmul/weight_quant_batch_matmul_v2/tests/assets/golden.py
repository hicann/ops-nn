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
            "weight_quant_batch_matmul_v2": "weight_quant_batch_matmul_v2_golden"
        }
}

import torch
import copy
import os
import numpy as np

def weight_quant_batch_matmul_v2_golden(x1, weight, antiquant_scale, antiquant_offset = None, quant_scale =None, 
                                        quant_offset = None, bias =None, transpose_x: bool = False, 
                                        transpose_weight: bool = False, antiquant_group_size: int = 0,
                                        dtype: int = -1, inner_precise: int = 0,  **kwargs):

    trans_a = transpose_x
    trans_b = transpose_weight
    input_formats = kwargs['input_formats']
    weight_format = input_formats[1]
    output_dtypes = kwargs['output_dtypes']
    testcase_name = kwargs['testcase_name']

    arr_temp=np.array([1]).astype(x1.dtype)

    x1_dtype =  x1.dtype
    weight_dtype = weight.dtype.name
    antiquant_scale_dtype = antiquant_scale.dtype

    #后面需要直接把weight的tensor转换成x1的tensor，torch.dtype不支持，只能用这种丑陋的方法
    arr_temp=np.array([1]).astype(x1.dtype)
    if x1.dtype == "bfloat16":
        arr_temp = arr_temp.astype(np.float32)
        x1_tensor_dtype = torch.from_numpy(arr_temp).to(torch.bfloat16).dtype
    else:
        x1_tensor_dtype = torch.from_numpy(arr_temp).dtype

    soc_ver = kwargs['short_soc_version']
    full_soc_ver = kwargs['full_soc_version']

    if (trans_a is False):
        m, k = x1.shape[-2], x1.shape[-1]
    else:
        k, m = x1.shape[-2], x1.shape[-1]

    if soc_ver in ("Ascend910B", "Ascend910_93", "Ascend950") and weight_format == "ND":
        if (trans_b is False):
            k, n = weight.shape
        else:
            n, k = weight.shape

    # antiquant模式获取
    k_dim = 1 if trans_b else 0
    antiquant_mode = judge_antiquant_mode(antiquant_scale, k_dim)

    x1_k_dim =  0 if trans_a else 1
    k_size = x1.shape[x1_k_dim]

    # weight如果是NZ格式 golden采用ND的origin shape
    weight_format = input_formats[1]
    if (weight_format == "FRACTAL_NZ"):
        # 950的A16场景
        if (x1.dtype.name == "bfloat16" or x1.dtype.name == "float16") and (soc_ver in ("Ascend950")):
            weight = trans_nz2nd(weight)
        else:
            weight = trans_nz2nd(weight)
            k_size = weight.shape[k_dim]
        # Ascend950 wqbmmv2 weight-nz 格式 C0=16, 目前仅 A16W4 用到
        if soc_ver in ["Ascend950"]:
            k_size = weight.shape[k_dim]
        if (trans_b is False):
            k, n = weight.shape
        else:
            n, k = weight.shape

    # 910B MSD-非group性能优化场景(A16W8, m<=96, x矩阵不转置，不支持C8)
    if (("Ascend910B" in soc_ver) or ("Ascend910_93" in soc_ver)) and (trans_a is False) and (m <= 96) and ((weight_dtype == "int8") or (weight_dtype == "int4")) and (output_dtypes[0] != "int8") and (antiquant_scale.dtype.name != "uint64"):
        # 910B MSD-group性能优化场景(额外加限制 M<= groupsize/8, weight矩阵不转置)

        is_msd = kwargs['is_msd'] 
        print("msd group 高性能标志开关 inner_precise: ", inner_precise)

        if (antiquant_mode == "antiquant_pergroup"):
            group_size = antiquant_group_size
            if (m <= (group_size / 8)) and (trans_b is False):
                is_msd_group = judge_msd_group(x1, weight, group_size, full_soc_ver)
                if inner_precise == 1:
                    print("Current is 910B MSD-group High perf scence")
                    return process_int4_quant_new_m_16(x1, weight, group_size, antiquant_offset, antiquant_scale, x1_dtype, bias)
                elif (is_msd == 2): # MSD group场景
                    print("Current is 910B MSD-group perf scence")
                    return _weight_quant_batchmatmul_msd_group(x1, weight, antiquant_scale, antiquant_offset, bias, transpose_x, transpose_weight, group_size)
                elif (is_msd_group):
                    print("Current is 910B MSD-group perf scence")
                    return _weight_quant_batchmatmul_msd_group(x1, weight, antiquant_scale, antiquant_offset, bias, transpose_x, transpose_weight, group_size)

        if (is_msd == 1): # MSD非group场景
            return _weight_quant_batchmatmul_msd_nogroup(x1, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, transpose_x, transpose_weight, weight_format)
        elif ((antiquant_mode == "antiquant_perchannel") and (m <= 64) and (k <= 65504) and (n <= 65504) and (k % 32 == 0) and (n % 32 == 0)):
            # 910B A16W8 MSD非group开放泛化能力
            # 多核切K tiling策略无法精准适配golden(走msd nogroup精度失败需确认用例golden是否走msd_nogroup)
            splitK, tilingKeyHead = ComputeSplitK(m, n, k, weight_dtype, trans_b, full_soc_ver)
            if ((tilingKeyHead == 106) and (weight_dtype == "int4")) and ((trans_b is False) or (n < 2*m)):
                print("Current tiling is 106XXX, but trans_b=false or (n < 2m) not support A16W4 msd nogroup")
            elif ((tilingKeyHead == 106) or (weight_dtype == "int4")):
                return _weight_quant_batchmatmul_msd_group(x1, weight, antiquant_scale, antiquant_offset, bias, transpose_x, transpose_weight, group_size)
            else:
                return _weight_quant_batchmatmul_msd_nogroup(x1, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, transpose_x, transpose_weight, weight_format)
    else:
        print("Current is normal perf scence")

    #x1、weight输入处理
    x1 = torch.from_numpy(x1.astype(np.float32))
    is_f8_input =  'float8' in weight_dtype # ["hifloat8", "float8_e5m2", "float8_e4m3fn"]

    is_mxfp4 = "float4" in str(weight_dtype) and str(antiquant_scale_dtype) == "float8_e8m0"
    is_fp4 = "float4" in str(weight_dtype)

    if is_fp4:
        if x1_dtype.name == "bfloat16":
                weight = torch.from_numpy(weight.astype(np.float32)).to(torch.float32)
        else:
                weight = torch.from_numpy(weight.astype(np.float32).astype(np.float16)).to(torch.float32)
        if (trans_b is False):
            k, n = weight.shape
        else:
            n, k = weight.shape
    elif is_f8_input:
        if x1_dtype.name == "bfloat16":
            weight = torch.from_numpy(weight.astype(np.float32)).to(torch.bfloat16)
        else:
            weight = torch.from_numpy(weight.astype(np.float16))
    else:
        weight = torch.from_numpy(weight.astype(np.int32)).to(torch.int32)

    custom_splitk_flag = False
    antiquant_scale_uint64 = antiquant_scale.dtype.name == "uint64"

    # 在伪量化阶段bf16用fp32计算，fp16只能用fp16计算不能用fp32计算，模拟npu计算逻辑
    if x1_dtype.name == "bfloat16":
        if is_mxfp4:
            antiquant_scale = torch.from_numpy(antiquant_scale.astype(np.float32))
        elif is_f8_input:
            antiquant_scale = torch.from_numpy(antiquant_scale.astype(np.float32)).to(torch.bfloat16)
        else:
            weight = weight.to(torch.float32)
            antiquant_scale = torch.from_numpy(antiquant_scale.astype(np.float32))
    elif x1_dtype.name == "float16":
        weight = weight.to(torch.float16)

        if is_mxfp4:
            antiquant_scale = torch.from_numpy(antiquant_scale.astype(np.float32)).to(torch.float16)
        else:
            if antiquant_scale.dtype.name == "uint64":
                antiquant_scale = np.load(testcase_name + "_antiscale.npy").astype(np.float16)
                os.remove(testcase_name + "_antiscale.npy")
            antiquant_scale = torch.from_numpy(antiquant_scale)

    if trans_a:
        array_trans = gen_axes_for_transpose(len(x1.shape) - 2, [1, 0])
        x1 = x1.permute(*array_trans)
    if trans_b:
        array_trans = gen_axes_for_transpose(len(weight.shape) - 2, [1, 0])
        weight = weight.permute(*array_trans)
        if antiquant_scale.size() != torch.Size([1]):
            if antiquant_mode == "antiquant_perchannel":
                antiquant_scale = antiquant_scale.reshape(antiquant_scale.shape[0], 1)
            array_trans = gen_axes_for_transpose(len(antiquant_scale.shape) - 2, [1, 0])
            antiquant_scale = antiquant_scale.permute(*array_trans)

    # numpy(0,)  实际shape a.shape=(0,) 维度数len(a.shape)=1  第一个shape值len(a)=0
    if (antiquant_mode == "antiquant_pertensor") or (antiquant_mode == "antiquant_perchannel"):
        if (antiquant_offset is not None) and (len(antiquant_offset) != 0): # 不为None 且shape不为(0,)
            if x1_dtype.name == "bfloat16":
                antiquant_offset = torch.from_numpy(antiquant_offset.astype(np.float32))
                if is_f8_input:
                    antiquant_offset = antiquant_offset.to(torch.bfloat16)
            elif x1_dtype.name == "float16":
                antiquant_offset = torch.from_numpy(antiquant_offset)
            if trans_b:
                if antiquant_offset.size() != torch.Size([1]):
                    antiquant_offset = antiquant_offset.reshape(antiquant_offset.shape[0], 1)
                    array_trans = gen_axes_for_transpose(len(antiquant_offset.shape) - 2, [1, 0])
                    antiquant_offset = antiquant_offset.permute(*array_trans)
            if custom_splitk_flag:
                if x1_dtype.name == "bfloat16":
                    weight = weight + antiquant_offset
                else:
                    weight = ((weight + antiquant_offset) * antiquant_scale).to(torch.float32)
            else:
                weight = ((weight + antiquant_offset) * antiquant_scale).to(torch.float32)
        else:
            if antiquant_scale_uint64:
                pass
            elif custom_splitk_flag:
                if x1_dtype.name == "float16":
                    weight = weight * antiquant_scale
            else:
                weight = (weight * antiquant_scale).to(torch.float32)
        if is_f8_input:
            weight = weight.to(x1_tensor_dtype)
    else:
        # pergroup场景下先转numpy处理后再转回torch
        group_size = antiquant_group_size
        weight = weight.numpy()
        antiquant_scale = antiquant_scale.numpy()
        if (antiquant_offset is not None) and (len(antiquant_offset) != 0):
            if x1_dtype.name == "bfloat16":
                antiquant_offset = antiquant_offset.astype(np.float32)
            if trans_b:
                if len(antiquant_offset.shape) != 1: # 2维
                    antiquant_offset = np.transpose(antiquant_offset)
        for i in range(k_size):
            group_size_dim = i // group_size
            if (antiquant_offset is not None) and (len(antiquant_offset) != 0):
                weight[i,:] = (weight[i,:]  + antiquant_offset[group_size_dim,:]) * antiquant_scale[group_size_dim,:]
            else:
                weight[i,:] = weight[i,:] * antiquant_scale[group_size_dim,:]


        weight = torch.from_numpy(weight)
        antiquant_scale = torch.from_numpy(antiquant_scale)
        if (antiquant_offset is not None) and (len(antiquant_offset) != 0):
            antiquant_offset = torch.from_numpy(antiquant_offset)

    if x1_dtype.name == "bfloat16":
        x1 = x1.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
    elif antiquant_scale_dtype.name == "uint64":
        weight = weight.to(torch.float16) # np的golden在fixpipe模板下先转回FP16

    output_y = torch.matmul(x1.to(torch.float32), weight.to(torch.float32))

    if (custom_splitk_flag and x1_dtype.name == "bfloat16") or antiquant_scale_uint64:
        output_y = output_y * antiquant_scale

    if (bias is not None) and (bias.size != 0):
        bias = bias.astype(np.float32)
        bias = torch.from_numpy(bias)
        output_y = output_y + bias

    if x1_dtype.name == "bfloat16":
        output_y = output_y.to(torch.bfloat16).to(torch.float32)

    if output_dtypes[0] == "int8":
        quant_scale = np.load(testcase_name + "_scale.npy")
        quant_offset = np.load(testcase_name + "_quant_offset.npy")
        quant_scale = torch.from_numpy(quant_scale)
        quant_offset = torch.from_numpy(quant_offset)

        if quant_scale.shape != torch.Size([0]):
            output_y = torch.clamp(torch.round(output_y * quant_scale), -256, 255).to(torch.int16)
        if quant_offset.shape != torch.Size([0]):
            output_y = torch.clamp(output_y + torch.clamp(torch.round(quant_offset), -256, 255), -128, 127)
        else:
            output_y = torch.clamp(output_y, -128, 127)

        os.remove(testcase_name + "_scale.npy")
        os.remove(testcase_name + "_quant_offset.npy")

        output_y = output_y.numpy()
    else:
        output_y = output_y.numpy()
        output_y = output_y.astype(x1_dtype, copy=False) # 不申请内存

    return output_y

# kn : n1 k1 k0 n0
# nk : k1 n1 n0 k0
def trans_nz2nd(input_data):
    nd_data = input_data.transpose(1, 2, 0, 3).reshape(input_data.shape[1] * input_data.shape[2], input_data.shape[0] * input_data.shape[3])
    return nd_data

def transpose_input(x, trans):
    if trans:
        array_trans = gen_axes_for_transpose(len(x.shape) - 2, [1, 0])
        return x.transpose(*array_trans)
    return x

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def judge_antiquant_mode(antiquant_scale, k_dim):
    antiquant_mode = "antiquant_perchannel"  # 1,n  n,1  n
    antiquant_scale_dim_num = len(antiquant_scale.shape)
    if (antiquant_scale_dim_num == 1) and (antiquant_scale.shape[0] == 1):
        antiquant_mode = "antiquant_pertensor"  # 1
    elif antiquant_scale_dim_num == 2:
        if antiquant_scale.shape[k_dim] != 1:
            antiquant_mode = "antiquant_pergroup"

    return antiquant_mode

# 校验泛化用例是否走msd-group方案
def judge_msd_group(x1, weight, group_size, full_soc_version):
    # 外层已约束(不支持C8;仅支持A矩阵、B矩阵均非转置;m范围[1,96];并且m<=groupsize/8;仅支持pergroup场景ND格式
    m, k = x1.shape
    k, n = weight.shape
    weight_dtype = weight.dtype.name

    if full_soc_version in ("Ascend910B1", "Ascend910B2"):
        coreNum = 24
    elif full_soc_version in ("Ascend910B3", "Ascend910B4"):
        coreNum = 20
    else:
        return False
    nkMax = 512 * 1024

    # 当前A16W8 msd-group泛化能力未开放,仅支持is_msd=2标记
    if weight_dtype == "int8":
        return False

    # groupsize的约束只能是64，128（且128仅A16W4支持）
    judge = ((weight_dtype == "int4") and ((group_size == 64) or (group_size == 128))) or (
            (weight_dtype == "int8") and (group_size == 64))
    if not judge:
        return judge
    # k轴应小于13952，且应满足groupSize对齐的限制(k % groupsize == 0)。n轴应满足64对齐的限制
    judge = ((k < 13952) and (k % group_size == 0) and (n % 64 == 0))
    if not judge:
        return judge
    # A16W4，额外增加k要满足2对齐限制
    if weight_dtype == "int4":
        if (k % 2 != 0):
            return False

    # A16W8场景，n,k限制：nk的范围应满足如下条件：CeilAlign(CeilDiv(k, coreNum / CeilDiv(n, 2048)), groupSize) * CeilAlign(2 * m, 32) + 2 * 2048 * groupSize <= 512 * 1024。
    # A16W4场景，n,k限制：nk的范围应满足如下条件：CeilAlign(CeilDiv(k, coreNum / CeilDiv(n, 2048)), groupSize) * CeilAlign(4 * m, 32) / 2 + 2 * 2048 * groupSize / 2 <= 512 * 1024。
    if weight_dtype == "int8":
        judgeG1 = align(ceil_div(int(k), coreNum // ceil_div(int(n), 2048)), group_size
                        ) * align(2 * int(m), 32) + 2 * 2048 * group_size // 2
        if judgeG1 > nkMax:
            return False
    elif weight_dtype == "int4":
        judgeG1 = align(ceil_div(int(k), coreNum // ceil_div(int(n), 2048)), group_size
                        ) * align(4 * int(m), 32) // 2 + 2 * 2048 * group_size // 2
        if judgeG1 > nkMax:
            return False

    return True

def align(a, b):
    return ceil_div(a, b) * b

def ceil_div(a, b):
    return (a + b - 1) // b

def _weight_quant_batchmatmul_msd_group(x, w, scale, offset, bias, transpose_x, transpose_weight, group_size):
    x_dtype = x.dtype
    w_dtype = w.dtype.name

    f1 = np.float32(7.49)
    f2 = np.float32(14.98)

    if transpose_x:
        x = np.transpose(x)
    if transpose_weight:
        w = np.transpose(w)
    m, k = x.shape
    k, n = w.shape

    if offset is None:
        offset = np.zeros((scale.shape[0], scale.shape[1]), dtype = np.float32)  # antiquant_offset为空时全填充0

    sumResult = np.zeros((m, k // group_size))
    maxResult = np.zeros((m, k // group_size))
    batchScaleMulOffsetResult = []

    a1 = np.zeros((m, k), dtype=np.float32)
    for gid in range(k // group_size):
        sumResult[:, gid] = np.sum(x.astype(np.float32)[:, gid * group_size:(gid + 1) * group_size], axis=1)
        maxResult[:, gid] = np.max(abs(x[:, gid * group_size:(gid + 1) * group_size]).astype(np.float32),
                                   axis=1)  # * 1.001
        for mIdx in range(m):
            if w_dtype == "int8":
                a1[mIdx, gid * group_size:(gid + 1) * group_size] = \
                    x[mIdx, gid * group_size:(gid + 1) * group_size].astype(np.float32) / maxResult[mIdx, gid] * 127
            elif w_dtype == "int4":
                a1[mIdx, gid * group_size:(gid + 1) * group_size] = \
                    x[mIdx, gid * group_size:(gid + 1) * group_size].astype(np.float32) / maxResult[mIdx, gid] * f1
        batchScaleMulOffsetResult.append(np.matmul(sumResult[:, gid].reshape(-1, 1),
                                                   (scale[gid, :].astype(np.float32) * offset[gid, :].astype(np.float32)
                                                    ).reshape(1, -1)))
    scaleMulOffsetResult = copy.deepcopy(batchScaleMulOffsetResult[0])

    for gid in range(1, k // group_size):
        scaleMulOffsetResult += batchScaleMulOffsetResult[gid]

    if w_dtype == "int8":
        a2 = a1 - np.round(a1)
        a2 = a2 * 254
        a2 = np.round(a2)
        a1a2 = np.concatenate([np.round(a1), a2], axis=0).astype(np.int8).reshape(2 * m, -1)
    elif w_dtype == "int4":
        a2 = a1 - np.round(a1)
        a2 = a2 * f2
        a3 = a2 - np.round(a2)
        a3 = a3 * f2
        a1a2 = np.concatenate([np.round(a1), np.round(a2), np.round(a3)], axis=0).astype(np.int8).reshape(3 * m, -1)

    batchResult = []
    for gid in range(k // group_size):
        batchResult.append(np.matmul(a1a2[:, gid * group_size:(gid + 1) * group_size].astype(np.int32),
                                     w[gid * group_size:(gid + 1) * group_size, :]).astype(np.int32))
    v4Result = np.zeros((m, n)).astype(np.float32)
    if w_dtype == "int8":
        for mIdx in range(m):
            smallIdx = m + mIdx
            v4Result[mIdx, :] = (copy.deepcopy(batchResult[0])[mIdx, :].astype(np.float32) / 127
                                 + copy.deepcopy(batchResult[0])[smallIdx, :].astype(np.float32) / (127 * 254)
                                 ) * maxResult[mIdx, 0].astype(np.float32) * scale[0, :].astype(np.float32)
            for gid in range(1, k // group_size):
                v4Result[mIdx, :] += (copy.deepcopy(batchResult[gid])[mIdx, :].astype(np.float32) / 127
                                      + copy.deepcopy(batchResult[gid])[smallIdx, :].astype(np.float32) / (127 * 254)
                                      ) * maxResult[mIdx, gid].astype(np.float32) * scale[gid, :].astype(np.float32)
    elif w_dtype == "int4":
        for gid in range(0, k // group_size):
            v4Result += (copy.deepcopy(batchResult[gid])[0:m, :].astype(np.float32) * (np.float32(1) / f1)
                         + copy.deepcopy(batchResult[gid])[m:2 * m, :].astype(np.float32) * (np.float32(1) / (f1 * f2))
                         + copy.deepcopy(batchResult[gid])[2 * m:, :].astype(np.float32) / (f1 * f2 * f2)
                         ) * maxResult[:, gid:gid + 1].astype(np.float32) * scale.astype(np.float32)[gid, :] \
                            .astype(np.float32)

    y = scaleMulOffsetResult + v4Result

    if (bias is not None) and (bias.size != 0):
        y = y + bias.astype(np.float32)

    y = y.astype(x_dtype)

    return y

def _weight_quant_batchmatmul_msd_nogroup(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, transpose_x, transpose_weight, weight_format):
    order = 2
    # 910B A16W8 MSD-nogroup(当前weightNZ只支持per_channel)
    if weight_format == "FRACTAL_NZ":
        weight = trans_nz2nd(weight)
    if transpose_x:
        x = np.transpose(x)
    if transpose_weight:
        weight = np.transpose(weight)
        if antiquant_offset is not None:
            antiquant_offset = antiquant_offset.T
        antiquant_scale = antiquant_scale.T

    new_x, a_max, a_sum = msd_nogroup_pre_process(x, order)

    y = np.matmul(new_x.astype(np.int32), weight.astype(np.int32))
    y = msd_nogroup_post_process(y, a_max, a_sum, order, antiquant_offset, antiquant_scale,
                                 quant_offset, quant_scale, bias)

    return y

def msd_nogroup_pre_process(a, order):
    a_f32 = a.astype(np.float32)
    a_max = np.max(abs(a_f32), axis=1, keepdims=True) * 1.001
    res = []
    factor = [np.power(128, i + 1) for i in range(order)]
    prev_value = a_f32 / a_max

    for i in range(order):
        if i > 0:
            prev_value = prev_value - res[i - 1] / np.power(128, i)
        tmp = np.floor(factor[i] * prev_value)
        res.append(tmp)

    a_sum = np.sum(a_f32, axis=1, keepdims=True)
    return np.concatenate(res, axis=0).astype(np.int8), a_max, a_sum

def msd_nogroup_post_process(c, a_max, a_sum, order, antiquant_offset, antiquant_scale,
                             quant_offset=None, quant_scale=None, bias=None):
    c_f32 = c.astype(np.float32)
    antiquant_scale_f32 = antiquant_scale.astype(np.float32)

    order_m, _ = c.shape
    m = order_m // order

    acc_c = c_f32[:m, :] * (float(1) / float(128))
    for i in range(1, order):
        acc_c += c_f32[i * m:(i + 1) * m, :] * (np.float32(1) / np.float32(np.power(128, i + 1)))

    res = a_max * acc_c
    # f32 version
    if antiquant_offset is not None:
        antiquant_offset_size = antiquant_offset.size
        # antiquant_offset支持(1,n) (n,1) (n,) (1,)
        antiquant_offset = antiquant_offset.reshape(1, antiquant_offset_size)
        antiquant_offset_f32 = antiquant_offset.astype(np.float32)
        brcb_antiquant_offset = np.matmul(a_sum, antiquant_offset_f32)
        res = res + brcb_antiquant_offset

    res = res * antiquant_scale_f32

    if bias is not None:
        res = res + bias

    return res.astype(antiquant_scale.dtype)

# 计算切K值和预期tilingKey
def ComputeSplitK(m, n, k, weight_dtype, transB, soc_version):
    #非多核切K场景判断
    if(k <= 13696) and (n <= 32000) and (weight_dtype == "int8"):
        return k, 6

    #A16W8 MSD_nogroup ND格式 多核切K适配
    aivNum = 48
    if("Ascend910B3" in soc_version) or ("Ascend910B4" in soc_version) or ("Ascend910_93" in soc_version):
        aivNum = 40
    
    kBlockNum = 1
    #n k 差距超过30倍， 需要考虑对k切多份, 该值为经验值
    if k//n >= 30 and (not transB):
        #根据经验，当weight不转置时，k切4份及以上，可以增加n方向并行度以提高性能
        kBlockNum = 4

    kAlignSize = 64
    if (transB):
        #转置场景优先保证cache line,使MTE2更好
        if(weight_dtype == "int8"):
            kAlignSize = 512
        else:
            kAlignSize = 1024
    
    for kBlock in range (kBlockNum, 7):
        singleCoreK = np.int32(np.ceil(np.ceil(k/kBlock) / kAlignSize) * kAlignSize)
        #单core容忍的k范围有限， 根据ub切分， 最大支持的规格为12 * 1024
        if singleCoreK <= 0 or singleCoreK > 12 * 1024:
            continue
        #前处理基本块大小为12 * 1024
        v1BaseM = 12 * 1024 // singleCoreK
        #根据workspace空间反算N轴的切分， C1C2在workspace上多份， 避免同步开销引起的性能劣化
        singleNSize = 16 * 1024 *1024 // (kBlock * 2 * m * 4)
        singleNSize = singleNSize / 256 * 256 #向下对齐到256， 保证非尾块处理效率
        #vec一次处理的标准块是64 * 128.按照n=128划分n方向计算一轮cube的n最大切分
        aivBaseN = 128
        if singleNSize > aivBaseN * aivNum:
            singleNSize = aivBaseN * aivNum
        if singleNSize > n:
            singleNSize = n
        singleCoreNSize = CeilAlign(CeilDiv(singleNSize, aivNum), kAlignSize)
        #后处理的n方向切分数量
        if singleCoreNSize > singleNSize:
            postProcessNBlockNum = 1
        else:
            postProcessNBlockNum = np.ceil(singleNSize / singleCoreNSize)
        postProcessMBlockNum = aivNum // postProcessNBlockNum
        postProcessSingleCoreM = CeilDiv(m, postProcessMBlockNum)
        if postProcessSingleCoreM <= 0:
            continue
        return singleCoreK, 106
    return k, 106

def process_int4_quant_new_m_16(x, w, group_size, offset, scale, output_info, bias, Expansion=3):
    # 传入进来已经是transpose了
    m, k = x.shape
    k, n = w.shape
    if offset is None or offset == () or offset.shape == ():
        offset = np.zeros((k // group_size, n), dtype = np.float32) #antiquant_offset为空时全填充0
    sumResult = np.zeros((m, k // group_size)).astype(np.float32)
    maxResult = np.zeros((m, k // group_size)).astype(np.float32)
    batchScaleMulOffsetResult = []

    x = x.astype(np.float32)
    a1 = np.zeros((m, k), dtype = np.float32)
    for gid in range(k // group_size):
        sumResult[:, gid] = np.sum(x.astype(np.float32)[:, gid * group_size:(gid + 1) * group_size], axis = 1)
        maxResult[:, gid] = np.max(abs(x[:, gid * group_size:(gid + 1) * group_size]).astype(np.float32), axis = 1)
        a1[:, gid * group_size:(gid + 1) * group_size] = x[:, gid * group_size:(gid + 1) * group_size].astype(np.float32) / maxResult[:,gid,np.newaxis] * 7.49

        batchScaleMulOffsetResult.append(np.matmul(sumResult[:, gid].reshape(-1, 1), (scale[gid, :].astype(np.float32) * offset[gid, :].astype(np.float32)).reshape(1, -1)))
    scaleMulOffsetResult = copy.deepcopy(batchScaleMulOffsetResult[0])
    for gid in range(1, k // group_size):
        scaleMulOffsetResult += batchScaleMulOffsetResult[gid]

    a_list = []
    a_list.append(np.round(copy.deepcopy(a1)))
    for i in range(1, Expansion):
        a1 = a1 - np.round(a1)
        a1 = a1  * 14.98
        a_list.append(np.round(copy.deepcopy(a1)))

    a1a2 = np.concatenate(a_list, axis = 0).astype(np.int8).reshape(Expansion * m, -1)

    batchResult = []
    for gid in range(k // group_size):
        batchResult.append(np.matmul(a1a2[:,gid * group_size:(gid + 1) * group_size].astype(np.int32), w[gid * group_size:(gid + 1) * group_size, :]).astype(np.float16))

    tResult = copy.deepcopy(batchResult[0])[0:(0 +1 ) * m , :].astype(np.float16)
    for i in range(1, Expansion):
        tResult += copy.deepcopy(batchResult[0])[(i) * m:(i + 1) * m, :].astype(np.float16) * np.float16(1 / pow(14.98, i))

    v4Result = tResult.astype(np.float32) / (7.49) * maxResult[:, 0].reshape(-1, 1).astype(np.float32) * scale[0, :].reshape(1, -1).astype(np.float32)

    for gid in range (1, k // group_size):
        aResult = copy.deepcopy(batchResult[gid])[0:(0 + 1) * m, :].astype(np.float16)
        for i in range(1, Expansion):
            aResult += copy.deepcopy(batchResult[gid])[(i) * m:(i +1 ) * m, :].astype(np.float16) * np.float16(1 / pow(14.98, i))
        v4Result += aResult.astype(np.float32) / (7.49) * maxResult[:, gid].reshape(-1, 1).astype(np.float32) * scale[gid, :].reshape(1, -1).astype(np.float32)

    y = scaleMulOffsetResult + v4Result
    if bias is not None:
        y = y + bias.astype(np.float32)
    y = y.astype(output_info)
    return y

def CeilAlign(a, b):
    return (a + b - 1) // b * b

def CeilDiv(a, b):
    return (a + b - 1) // b