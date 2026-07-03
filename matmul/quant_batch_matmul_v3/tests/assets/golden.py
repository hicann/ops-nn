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
import numpy as np
import en_dtypes
import ml_dtypes
np_fp4_e2m1 = en_dtypes.float4_e2m1
np_fp4_e1m2 = en_dtypes.float4_e1m2
np_hif8 = en_dtypes.hifloat8
np_mx_scale = en_dtypes.float8_e8m0
np_bfloat16 = ml_dtypes.bfloat16
np_fp8_e4m3 = ml_dtypes.float8_e4m3fn
np_fp8_e5m2 = ml_dtypes.float8_e5m2
np_int4 = ml_dtypes.int4

__golden__ = {
	     "kernel": {
	         "quant_batch_matmul_v3": "quant_batch_matmul_v3_golden"
	     }
	 }


# 参数命名与算子定义(V3 op index)及V5 API的对应关系:
#   x1          → 算子入参0 x1 (左矩阵)                → V5: x1
#   x2          → 算子入参1 x2 (右矩阵)                → V5: x2
#   scale       → 算子入参2 scale (x2量化scale)         → V5: x2Scale; golden接收的是customize_inputs处理后的deq_scale
#   offset      → 算子入参3 offset (x2量化offset)       → V5: x2Offset
#   bias        → 算子入参4 bias                        → V5: bias
#   pertoken_scale → 算子入参5 pertoken_scale (x1量化scale) → V5: x1Scale
#   y           → 算子出参0 y (结果)                    → V5: out
#
# 内部变量命名遵循V5语义, 与算子入参名存在明显关联:
#   x2_scale    → 算子入参scale (x2侧量化scale, customize_inputs后为deq_scale)
#   x1_scale    → 算子入参pertoken_scale (x1侧量化scale)
#   scale_dtype → 算子入参scale的原始dtype (通过kwargs传入)
#   y_dtype     → 算子出参y的dtype
#
# T-C场景简化说明:
#   customize_inputs已将UINT64 scale转为float32 deq_scale(含高19位截断, 等价于scale_generate).
#   offset不从uint64 scale中提取, 保留原始输入参数(与TTK框架行为一致:
#   原始offset输入为None时不参与requant计算, 仅当显式提供时才使用).
#   golden()接收的x2_scale在两种T-C场景下均为float32/bf16,
#   bias位置由is_bias_vec决定(与TTK golden及_compute_pertoken一致):
#   非is_bias_vec时bias在×x2Scale前加入(int32域/float32域加法),
#   is_bias_vec时bias在×x2Scale后加入(fixpipe后处理).
#   T-C-static(uint64)和T-C-dynamic(float32/bf16)合并为单个_compute_tc函数.

def quant_batch_matmul_v3_golden(x1, x2, scale, offset=None, bias=None, pertoken_scale=None,
                                 *, dtype: int, transpose_x1: bool = False, transpose_x2: bool = False,
                                 group_size: int = 0, **kwargs):
    scale_dtype = kwargs.get('scale_dtype', _dtype_to_str(scale.dtype))
    x1, x2, scale, offset, bias, pertoken_scale = customize_inputs(
        x1, x2, scale, offset, bias, pertoken_scale,
        dtype=dtype, transpose_x1=transpose_x1, transpose_x2=transpose_x2,
        group_size=group_size, **kwargs)
    y_dtype = kwargs.get('output_dtypes', ['float32'])[0]
    x1_dtype = _dtype_to_str(x1.dtype)
    x2_dtype = _dtype_to_str(x2.dtype)
    bias_dtype = _dtype_to_str(bias.dtype) if bias is not None else None
    x2_scale = scale.astype(np.float32) if scale.dtype != np_mx_scale else scale
    x1_scale = pertoken_scale
    group_size_m, group_size_n, group_size_k = _unpack_groupsize(group_size)
    quant_mode = _determine_quant_mode(
        x1_dtype, x2_dtype, x1_scale, x2_scale, scale_dtype, group_size_n)
    do_scale_gen = _needs_scale_generate(x1_dtype, x2_scale, bias_dtype, scale_dtype)
    is_bias_vec = x1_dtype in ('int8', 'int4') and \
                  bias_dtype is not None and \
                  bias_dtype in ('bfloat16', 'float16', 'float32')
    if offset is not None:
        offset = offset.astype(np.float32)
        if x2_scale.shape[-1] < offset.shape[-1]:
            offset = offset[0]
    if quant_mode == "G-G":
        y = _compute_mx(x1, x2, x2_scale, x1_scale, bias, bias_dtype,
                        transpose_x1, transpose_x2, y_dtype)
    elif quant_mode == "B-B":
        y = _compute_perblock(x1, x2, x2_scale, x1_scale,
                              group_size_m, group_size_n, group_size_k,
                              transpose_x1, transpose_x2, y_dtype, bias)
    else:
        if x1_dtype in ('int8', 'int4') and x2_dtype in ('int8', 'int4'):
            x1 = x1.astype(np.int32)
            x2 = x2.astype(np.int32)
        else:
            x1 = x1.astype(np.float32)
            x2 = x2.astype(np.float32)
        if transpose_x1:
            x1 = np.transpose(x1, _gen_axes_for_transpose(len(x1.shape) - 2, [1, 0]))
        if transpose_x2:
            x2 = np.transpose(x2, _gen_axes_for_transpose(len(x2.shape) - 2, [1, 0]))
        matmul_out = np.matmul(x1, x2)
        if y_dtype == 'int8':
            y = _compute_requant(matmul_out, x2_scale, offset, bias, bias_dtype)
        elif y_dtype == 'int32':
            y = _compute_int32(matmul_out, bias, bias_dtype)
        elif quant_mode == "K-C":
            y = _compute_pertoken(matmul_out, x2_scale, x1_scale, bias, bias_dtype,
                                  is_bias_vec, y_dtype, x1_dtype)
        elif quant_mode == "T-C":
            y = _compute_tc(matmul_out, x2_scale, bias, bias_dtype,
                            do_scale_gen, is_bias_vec, y_dtype)
        else:
            if matmul_out.dtype == np.int32:
                matmul_out = matmul_out.astype(np.float32)
            y = _cast_output_dtype(matmul_out * x2_scale, y_dtype)
    return [y]



def customize_inputs(x1, x2, scale, offset=None, bias=None, pertoken_scale=None,
                     *, dtype: int, transpose_x1: bool = False, transpose_x2: bool = False,
                     group_size: int = 0, **kwargs):
    if scale.dtype in (np.uint64, np.int64):
        deq_scale = _u64_to_deq_scale(scale)
        if offset is not None:
            offset = _u64_to_offset(scale)
    else:
        deq_scale = scale
    input_formats = kwargs.get('input_formats', ())
    input_ori_shapes = kwargs.get('input_ori_shapes', ())
    if len(input_formats) > 1 and input_formats[1] == 'FRACTAL_NZ':
        ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
        x2 = _nz_to_nd(x2, ori_shape)
    if len(input_formats) > 2 and input_formats[2] == 'FRACTAL_NZ' and \
       kwargs.get('scale_dtype') not in ("uint64", "int64"):
        ori_shape = input_ori_shapes[2] if len(input_ori_shapes) > 2 else None
        deq_scale = _nz_to_nd(deq_scale, ori_shape)
    return x1, x2, deq_scale, offset, bias, pertoken_scale


def pre_compare(*outputs, **kwargs):
    return list(outputs)


class QuantBatchMatmulV3Assets:
    """quant_batch_matmul_v3 算子的全部资产（每个都是可选的）"""
    golden = quant_batch_matmul_v3_golden
    customize_inputs = customize_inputs
    pre_compare = pre_compare

    tolerance = {
        "float32": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0, "max_re_rtol": 5.0, "rmse_rtol": 2.0,
            "small_value": 1e-6, "small_value_atol": 1e-9,
        },
        "float16": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0, "max_re_rtol": 10.0, "rmse_rtol": 2.0,
            "small_value": 0.001, "small_value_atol": 1e-5,
        },
        "bfloat16": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0, "max_re_rtol": 10.0, "rmse_rtol": 2.0,
            "small_value": 0.001, "small_value_atol": 1e-5,
        },
        "float8_e5m2": {
            "standard": "BenchmarkCompareStandard",
            "avg_re_rtol": 2.0, "max_re_rtol": 10.0, "rmse_rtol": 2.0,
            "small_value": 0.001, "small_value_atol": 1e-5,
        },
        "int32": {"standard": "BinaryMatch"},
        "int8": {"standard": "IsClose", "rtol": 0, "atol": 1},
    }


# ----------------------------------------------------------------------------
# 量化模式计算函数 — 每种量化类型一个函数, 函数前注释给出计算基础流程
# ----------------------------------------------------------------------------

def _compute_mx(x1, x2, x2_scale, x1_scale, bias, bias_dtype,
                transpose_x1, transpose_x2, y_dtype):
    # MX(G-G) 计算流程:
    # 1. x1/x2转换为float32
    # 2. 将x1Scale(E8M0)预乘到x1, 将x2Scale(E8M0)预乘到x2 (含transpose对齐和K轴padding)
    # 3. matmul(x1, x2) → 结果已含scale效果
    # 4. +bias(FP32 opt)
    # 5. cast到y_dtype
    x2_scale_mx = x2_scale.copy().astype(np.float32)
    x1_scale_mx = x1_scale.copy().astype(np.float32)
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)

    if transpose_x1:
        x1 = np.swapaxes(x1, -1, -2)
        x1_scale_mx = np.swapaxes(x1_scale_mx, -1, -2)
        if len(x1_scale_mx.shape) == 3:
            x1_scale_mx = x1_scale_mx.reshape(
                x1_scale_mx.shape[0] * x1_scale_mx.shape[1], x1_scale_mx.shape[2])
        x1_scale_mx = np.swapaxes(x1_scale_mx, -1, -2)
    else:
        if len(x1_scale_mx.shape) == 3:
            x1_scale_mx = x1_scale_mx.reshape(
                x1_scale_mx.shape[0], x1_scale_mx.shape[1] * x1_scale_mx.shape[2])

    if transpose_x2:
        x2 = np.swapaxes(x2, -1, -2)
        if len(x2_scale_mx.shape) == 3:
            x2_scale_mx = x2_scale_mx.reshape(
                x2_scale_mx.shape[0], x2_scale_mx.shape[1] * x2_scale_mx.shape[2])
        x2_scale_mx = np.swapaxes(x2_scale_mx, -1, -2)
    else:
        x2_scale_mx = np.swapaxes(x2_scale_mx, -1, -2)
        if len(x2_scale_mx.shape) == 3:
            x2_scale_mx = x2_scale_mx.reshape(
                x2_scale_mx.shape[0] * x2_scale_mx.shape[1], x2_scale_mx.shape[2])

    k_dim = x1.shape[-1]
    if _ceil_div(k_dim, 32) % 2 != 0:
        x1_scale_mx = x1_scale_mx[:, :-1]
        x2_scale_mx = x2_scale_mx[:-1, :]

    x1_scale_mx_br = np.repeat(x1_scale_mx, 32, axis=-1)
    x2_scale_mx_br = np.repeat(x2_scale_mx, 32, axis=-2)

    x1_dims = len(x1.shape)
    x2_dims = len(x2.shape)
    x1_pad_len = x1_scale_mx_br.shape[-1] - x1.shape[-1]
    x2_pad_len = x2_scale_mx_br.shape[-2] - x2.shape[-2]
    if x1_pad_len > 0:
        x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)],
                     mode='constant', constant_values=0)
    if x2_pad_len > 0:
        x2 = np.pad(x2, [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)],
                     mode='constant', constant_values=0)

    x1 = x1 * x1_scale_mx_br
    x2 = x2 * x2_scale_mx_br

    y = np.matmul(x1, x2)
    if bias is not None:
        y = _cast_output_dtype(y + bias.astype(np.float32), y_dtype)
    else:
        y = _cast_output_dtype(y, y_dtype)
    return y


def _compute_perblock(x1, x2, x2_scale, x1_scale,
                      group_size_m, group_size_n, group_size_k,
                      transpose_x1, transpose_x2, y_dtype, bias=None):
    # B-B(PerBlock) 计算流程:
    # 1. x1/x2转换为float32
    # 2. 对x1Scale/x2Scale做transpose对齐x1/x2的行列方向
    # 3. broadcast对齐batch维度
    # 4. x1Scale沿M轴repeat(groupSizeM), x2Scale沿N轴repeat(groupSizeN)
    # 5. 分K轴block级累加: matmul(x1[:,k_start:k_end], x2[k_start:k_end,:]) × (x1Scale[:,k_idx] × x2Scale[k_idx,:])
    # 6. cast到y_dtype
    # 注: B-B不支持bias; 仅FP8/HIF8可走PerBlock(INT8/INT4走V4)
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x1_scale_f = x1_scale.astype(np.float32)
    x2_scale_f = x2_scale.astype(np.float32)

    if transpose_x1:
        x1 = np.transpose(x1, _gen_axes_for_transpose(len(x1.shape) - 2, [1, 0]))
        x1_scale_f = np.transpose(x1_scale_f, _gen_axes_for_transpose(len(x1_scale_f.shape) - 2, [1, 0]))
    if transpose_x2:
        x2 = np.transpose(x2, _gen_axes_for_transpose(len(x2.shape) - 2, [1, 0]))
        x2_scale_f = np.transpose(x2_scale_f, _gen_axes_for_transpose(len(x2_scale_f.shape) - 2, [1, 0]))

    batch_x1 = list(x1.shape[:-2])
    batch_x2 = list(x2.shape[:-2])
    batch_ps = list(x1_scale_f.shape[:-2])
    batch_ds = list(x2_scale_f.shape[:-2])

    all_batches = [b for b in [batch_x1, batch_x2, batch_ps, batch_ds] if b]
    if not all_batches:
        batch_out = []
    else:
        max_len = max(len(b) for b in all_batches)
        batch_out = list(all_batches[0])
        for b in all_batches[1:]:
            padded_b = [1] * (max_len - len(b)) + b
            padded_out = [1] * (max_len - len(batch_out)) + batch_out
            for idx in range(max_len):
                padded_out[idx] = max(padded_out[idx], padded_b[idx])
            batch_out = padded_out

    if batch_out:
        if batch_x1 != batch_out:
            x1 = np.broadcast_to(x1, batch_out + list(x1.shape[-2:]))
        if batch_x2 != batch_out:
            x2 = np.broadcast_to(x2, batch_out + list(x2.shape[-2:]))
        if batch_ps != batch_out:
            x1_scale_f = np.broadcast_to(x1_scale_f, batch_out + list(x1_scale_f.shape[-2:]))
        if batch_ds != batch_out:
            x2_scale_f = np.broadcast_to(x2_scale_f, batch_out + list(x2_scale_f.shape[-2:]))

        batch_all = int(np.prod(batch_out))
        x1 = x1.reshape([batch_all] + list(x1.shape[-2:]))
        x2 = x2.reshape([batch_all] + list(x2.shape[-2:]))
        x1_scale_f = x1_scale_f.reshape([batch_all] + list(x1_scale_f.shape[-2:]))
        x2_scale_f = x2_scale_f.reshape([batch_all] + list(x2_scale_f.shape[-2:]))

    m = x1.shape[-2]
    k = x1.shape[-1]
    n = x2.shape[-1]

    x1_scale_m = np.repeat(x1_scale_f, group_size_m, axis=-2)
    x1_scale_m = x1_scale_m[..., :m, :]
    x2_scale_n = np.repeat(x2_scale_f, group_size_n, axis=-1)
    x2_scale_n = x2_scale_n[..., :n]

    has_batch = x1.ndim > 2
    if has_batch:
        y = np.zeros((batch_all, m, n), dtype=np.float32)
        for i in range(batch_all):
            for k_idx in range(_ceil_div(k, group_size_k)):
                k_start = k_idx * group_size_k
                k_end = min((k_idx + 1) * group_size_k, k)
                x1_s_col = np.expand_dims(x1_scale_m[i, :, k_idx], axis=1)
                x2_s_row = np.expand_dims(x2_scale_n[i, k_idx, :], axis=0)
                scale_mul = x1_s_col * x2_s_row
                y[i] += np.matmul(x1[i, :, k_start:k_end], x2[i, k_start:k_end, :]) * scale_mul
        if batch_out:
            y = y.reshape(batch_out + [m, n])
    else:
        y = np.zeros((m, n), dtype=np.float32)
        for k_idx in range(_ceil_div(k, group_size_k)):
            k_start = k_idx * group_size_k
            k_end = min((k_idx + 1) * group_size_k, k)
            x1_s_col = np.expand_dims(x1_scale_m[:, k_idx], axis=1)
            x2_s_row = np.expand_dims(x2_scale_n[k_idx, :], axis=0)
            scale_mul = x1_s_col * x2_s_row
            y += np.matmul(x1[:, k_start:k_end], x2[k_start:k_end, :]) * scale_mul

    if bias is not None:
        y = y + bias.astype(np.float32)
    return _cast_output_dtype(y, y_dtype)


def _compute_pertoken(matmul_out, x2_scale, x1_scale, bias, bias_dtype,
                      is_bias_vec, y_dtype, x1_dtype):
    # K-C(PerToken参与) 计算流程:
    # 1. INT32 bias前加 (int32域加法, 与硬件fixpipe一致)
    # 2. 转float32: 判断isDoubleScale → merged_scale或分离scale乘法 (float32域)
    # 3. float32 bias(非is_bias_vec)和is_bias_vec的bias均在scale乘法后加入 (与硬件fixpipe一致)
    # 4. cast到y_dtype
    out = matmul_out
    if bias is not None and bias_dtype == "int32":
        out = out + bias

    out = out.astype(np.float32)
    is_two_scale = x1_dtype in ("float8_e4m3fn", "float8_e5m2", "hifloat8") and \
                   x1_scale.shape[0] == 1 and x2_scale.shape[0] == 1
    if is_two_scale:
        merged_scale = x2_scale.astype(np.float32) * x1_scale.astype(np.float32)
        merged_scale_slice = np.expand_dims(merged_scale, axis=1)
        out = out * merged_scale_slice
    else:
        x1_scale_slice = np.expand_dims(x1_scale, axis=1).astype(np.float32)
        out = out * x2_scale.astype(np.float32) * x1_scale_slice

    if not is_bias_vec and bias is not None and bias_dtype == "float32":
        out = out + bias.astype(np.float32)
    if is_bias_vec and bias is not None:
        return _cast_output_dtype(out + bias.astype(np.float32), y_dtype)
    if y_dtype == 'float32':
        return out.astype(np.float32)
    return _cast_output_dtype(out, y_dtype)


def _compute_tc(matmul_out, x2_scale, bias, bias_dtype,
                do_scale_gen, is_bias_vec, y_dtype):
    # T-C(PerTensor/PerChannel, x1Scale不参与) 计算流程:
    # customize_inputs已将UINT64 scale转为float32 deq_scale,
    # T-C-static(uint64)和T-C-dynamic(float32/bf16)合并处理, bias位置由is_bias_vec决定.
    #
    # 计算流程:
    # 1. 非is_bias_vec: +bias (int32域/float32域加法, 反量化前加入, 与TTK golden及_compute_pertoken一致)
    # 2. 转float32
    # 3. 若do_scale_gen=True: 截断x2Scale高19位
    # 4. x2Scale若1维则expand到2维; out = out × x2Scale (float32域乘法)
    # 5. is_bias_vec时+bf16/fp16/fp32 bias (fixpipe后处理, 反量化后加入)
    # 6. cast到y_dtype
    out = matmul_out

    if not is_bias_vec and bias is not None:
        if bias_dtype == "int32":
            out = out + bias
        elif bias_dtype in ("float32", "bfloat16"):
            out = out.astype(np.float32) + bias.astype(np.float32)

    out = out.astype(np.float32)
    if do_scale_gen:
        x2_scale = _scale_generate(x2_scale)
    if x2_scale.ndim == 1:
        x2_scale = np.expand_dims(x2_scale, axis=0)
    out = out * x2_scale.astype(np.float32)

    if is_bias_vec and bias is not None:
        return _cast_output_dtype(out + bias.astype(np.float32), y_dtype)
    if y_dtype == 'float32':
        return out.astype(np.float32)
    return _cast_output_dtype(out, y_dtype)


def _compute_requant(matmul_out, x2_scale, offset, bias, bias_dtype):
    # Requant(y=INT8) 计算流程 — T-C的输出dtype变体:
    # 1. +INT32 bias (int32域加法, 与TTK一致, 反量化前加入)
    # 2. 转float32: f32_2_s9(out × x2Scale) — float32域乘法 + 9bit量化截断
    # 3. +f32_2_s9(offset) (若有)
    # 4. clip[-128, 127] → int8
    # 注: 仅INT8×INT8+UINT64 scale支持requant
    out = matmul_out
    if bias is not None and bias_dtype == "int32":
        out = out + bias

    out = out.astype(np.float32)
    out = _f32_2_s9(out * x2_scale.astype(np.float32))
    if offset is not None:
        out = _f32_2_s9(out) + _f32_2_s9(offset)
    return np.clip(out, -128, 127).astype(np.int8)


def _compute_int32(matmul_out, bias, bias_dtype):
    # 纯整数(y=INT32) 计算流程 — scale不参与计算的输出dtype变体:
    # 1. +INT32 bias (int32域加法, 与TTK一致); float32 bias在float32域加法
    # 2. cast到int32
    # 注: 仅INT8×INT8支持; scale不参与, bias位置不影响最终结果
    out = matmul_out
    if bias is not None:
        if bias_dtype == "int32":
            out = out + bias
        else:
            out = out.astype(np.float32) + bias.astype(np.float32)
    return out.astype(np.int32)


# ----------------------------------------------------------------------------
# 量化模式判定与辅助函数
# ----------------------------------------------------------------------------

def _determine_quant_mode(x1_dtype, x2_dtype, x1_scale, x2_scale,
                          scale_dtype, group_size_n):
    # 量化模式判定流程对齐op_tiling代码: 双轴(x1+x2)独立判定 → 组合映射
    # Step1判x2(x2Scale侧): P1(MX)→P2(PerBlock)→P3(PerTensor)→P4(PerChannel)
    # Step2判x1(x1Scale侧): P0(skip)→P1(MX)→P2(PerBlock)→P3(PERTENSOR/isDoubleScale)→P4(PERTOKEN)
    # Step3组合映射 → 量化场景名 + 公式
    # 输出dtype变体(y=INT8/INT32)不影响量化模式判定, 由_compute_requant/_compute_int32处理
    # T-C不再区分static/dynamic子模式: customize_inputs已将UINT64转为float32 deq_scale,
    #   bias位置由is_bias_vec决定(非is_bias_vec→scale前加, is_bias_vec→scale后加)
    #
    # 模式名(与op_tiling代码boolean flags的对应):
    #   G-G   = isMxPerGroup  (MX_PERGROUP+MX_PERGROUP)
    #   B-B   = isPerBlock    (PERBLOCK+PERBLOCK, 仅FP8/HIF8)
    #   K-C   = isPertoken+isPerChannel/isPerTensor (PERTOKEN+PERCHANNEL/PERTENSOR,
    #           或PERTENSOR+PERTENSOR(isDoubleScale); 公式统一, numpy广播无需区分)
    #   T-C   = isPerTensor/isPerChannel (DEFAULT+PERTENSOR/PERCHANNEL, bias位置由is_bias_vec决定)
    fp8_hif8 = ("float8_e4m3fn", "float8_e5m2", "hifloat8")
    fp4_fp8 = ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2")

    # Step1 P1 + Step2 P1: MX — both x1 and x2 must be fp4/fp8
    # FP8×FP4 mx伪量化已去除: x1和x2必须同属fp4或fp8族
    if x1_scale is not None and \
       x1_dtype in fp4_fp8 and x2_dtype in fp4_fp8 and \
       scale_dtype == "float8_e8m0":
        return "G-G"

    # Step1 P2 + Step2 P2: PerBlock (B-B) — 仅FP8/HIF8, 不含INT8/INT4
    # 代码条件: isFp8Hif8Input ∧ scaleDtype==FP32 ∧ scaleShapeLen>1
    #            ∧ (scaleShapeLen>2 ∨ scaleDim0!=1 ∨ groupSizeN>1)
    #            isFp8Hif8Input ∧ perTokenScaleDtype==FP32 ∧ pertokenLen>1
    if x1_scale is not None and \
       x1_dtype in fp8_hif8 and \
       scale_dtype == "float32" and \
       _dtype_to_str(x1_scale.dtype) == "float32" and \
       x2_scale.ndim > 1 and \
       (x2_scale.ndim > 2 or x2_scale.shape[0] != 1 or group_size_n > 1) and \
       x1_scale.ndim > 1:
        return "B-B"

    # Step2 P3-P4: x1Scale参与 — K-C/T-C动态FP8/HIF8/T-T动态(isDoubleScale)
    # 公式统一: out = matmul × x2Scale × x1Scale [+bias]
    if x1_scale is not None:
        return "K-C"

    # Step2 P0: x1Scale不参与(pertoken_scale=null) — T-C
    # bias位置由is_bias_vec决定(与TTK golden及_compute_pertoken一致):
    #   非is_bias_vec → bias在scale前加入 (int32域/float32域加法)
    #   is_bias_vec   → bias在scale后加入 (fixpipe后处理)
    return "T-C"


def _needs_scale_generate(x1_dtype, x2_scale, bias_dtype, scale_dtype):
    # 950硬件对int8/int4输入做fixpipe精度截断(scale_generate高19位掩码)
    # 条件: int8/int4 + 单行x2Scale + bias不是bf16/f32 + x2Scale原始dtype非uint64/int64
    # uint64/int64的deq_scale已是截断精度(customize_inputs中_u64_to_deq_scale已做0xFFFFE000掩码),
    # 即使apply scale_generate也无副作用(已是截断值再截断=原值), 但为避免不必要的计算, 仍跳过
    if x1_dtype not in ("int8", "int4"):
        return False
    if scale_dtype in ("uint64", "int64"):
        return False
    if x2_scale.shape[0] != 1:
        return False
    if bias_dtype in ("bfloat16", "float32"):
        return False
    return True


def _u64_to_deq_scale(u64_scale):
    deq_u32 = u64_scale.astype(np.uint32).copy()
    deq_u32 &= np.uint32(0xFFFFE000)
    return deq_u32.view(np.float32).reshape(u64_scale.shape)


def _u64_to_offset(u64_scale):
    raw = (u64_scale.astype(np.uint64) >> np.uint64(37)) & np.uint64(0x1FF)
    raw = raw.astype(np.int64)
    sign_mask = np.int64(0x100)
    raw = np.where(raw & sign_mask, raw - np.int64(0x200), raw)
    return raw.astype(np.float32).reshape(u64_scale.shape)


def _scale_generate(fp32_array):
    u32 = fp32_array.view(np.uint32).copy()
    u32 &= np.uint32(0xFFFFE000)
    return u32.view(np.float32)


def _f32_2_s9(array):
    return np.clip(np.round(array), -256, 255)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _unpack_groupsize(group_size):
    gs_m = (group_size >> 32) & 0xFFFF
    gs_n = (group_size >> 16) & 0xFFFF
    gs_k = group_size & 0xFFFF
    if gs_m == 0:
        gs_m = 1
    if gs_n == 0:
        gs_n = 1
    if gs_k == 0:
        gs_k = 1
    return gs_m, gs_n, gs_k


def _gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def _cast_output_dtype(arr, dtype_name):
    dtype_map = {
        "float16": np.float16, "float32": np.float32, "int32": np.int32,
        "int8": np.int8, "bfloat16": np_bfloat16,
        "hifloat8": np_hif8, "float8_e4m3fn": np_fp8_e4m3,
        "float8_e5m2": np_fp8_e5m2,
    }
    target = dtype_map.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)


def _dtype_to_str(dtype):
    dtype_map = {
        np.float16: "float16", np.float32: "float32", np.float64: "float64",
        np.int8: "int8", np.int32: "int32", np.uint64: "uint64",
        np_bfloat16: "bfloat16", np_int4: "int4",
        np_fp8_e4m3: "float8_e4m3fn", np_fp8_e5m2: "float8_e5m2",
        np_hif8: "hifloat8", np_mx_scale: "float8_e8m0",
        np_fp4_e2m1: "float4_e2m1", np_fp4_e1m2: "float4_e1m2",
    }
    return dtype_map.get(dtype, str(dtype))


def _nz_to_nd(data, ori_shape):
    if ori_shape is None:
        raise ValueError("ori_shape is required for NZ→ND conversion to remove fractal padding")
    shape = data.shape
    batch_dims = len(shape) - 4
    perm = list(range(batch_dims)) + [batch_dims + 1, batch_dims + 2, batch_dims + 0, batch_dims + 3]
    data = np.transpose(data, perm)
    m1 = shape[batch_dims + 1]
    m0_actual = shape[batch_dims + 2]
    n1 = shape[batch_dims + 0]
    n0_actual = shape[batch_dims + 3]
    data = data.reshape(*shape[:batch_dims], m1 * m0_actual, n1 * n0_actual)
    target_M = ori_shape[-2]
    target_N = ori_shape[-1]
    data = data[..., :target_M, :target_N]
    return data
