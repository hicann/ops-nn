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
"""
quant_batch_matmul_v3 kernel golden 实现。

本模块包含 V3 kernel 的 TestSpec 类和入口函数 qbmmv3_kernel_compute。
"""

import os
import sys
import numpy as np

# Import common utilities
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../common/tests/st/arch35"
    ),
)
from matmul_golden_util import (
    dtype_to_str,
    cast_output_dtype,
    nz_to_nd,
)
from matmul_quant_util import (
    unpack_groupsize,
    u64_to_deq_scale,
    u64_to_offset,
    needs_scale_generate,
    scale_generate,
    sanitize_e8m0_scale,
    gen_axes_for_transpose,
    ceil_div,
    np_mx_scale,
)


# PerBlock 模式支持的输入 dtype（FP8/HIF8，不含 FP4）
PERBLOCK_INPUT_DTYPE = ("float8_e4m3fn", "float8_e5m2", "hifloat8")
# MX (G-G) 模式支持的输入 dtype（FP4/FP8，不含 HIF8）
MX_INPUT_DTYPE = ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2")
# 整数量化输入 dtype（走 integer cube + fixpipe 流水线）
INT_QUANT_INPUT_DTYPE = ("int8", "int4")
# 触发 epilogue 后加的浮点 bias dtype
FLOAT_BIAS_DTYPE = ("float32", "bfloat16", "float16")


# ============================================================================
# V3-specific quantization mode determination
# ============================================================================


def determine_quant_mode(
    x1,
    x2,
    scale,
    pertoken_scale,
    scale_dtype,
    transpose_x1,
    transpose_x2,
    group_size_n,
    output_shapes,
):
    """
    判定量化模式，完全对齐 tiling 的 SetX1QuantMode + SetX2QuantMode + SetQuantMode。

    对应 tiling 代码:
    - SetX2QuantMode: tiling_base.cpp:458-483
    - SetX1QuantMode: tiling_base.cpp:484-512
    - SetQuantMode:   tiling_base.cpp:528-557

    返回:
      "G-G"             — MX (scale_dtype == float8_e8m0)
      "B-B"             — PerBlock (FP8/HIF8 + float32 + scale/pertoken 多维)
      "T-T-doublescale" — FP8/HIF8 + pertoken[-1]==1 + scale[-1]==1 → Cube: merge + 19bit 截断
      "T-C-doublescale" — FP8/HIF8 + pertoken[-1]==1 + scale[-1]==N → Mix: 分离乘法, 无截断
      "K-C"             — pertoken[-1]==M + scale[-1]==N → Mix: 分离乘法, 无截断
      "T-T"             — 无 pertoken + scale[-1]==1
      "T-C"             — 无 pertoken + scale[-1]==N
    """
    x1_dtype = dtype_to_str(x1.dtype)
    is_fp8_hif8 = x1_dtype in PERBLOCK_INPUT_DTYPE

    # M, N 从 output shape 获取（infershape 保证输出为 [..., M, N]）
    out_shape = output_shapes[0] if output_shapes else None
    if out_shape is not None and len(out_shape) >= 2:
        m_size = out_shape[-2]
        n_size = out_shape[-1]
    else:
        m_size = x1.shape[-1] if transpose_x1 else x1.shape[-2]
        n_size = x2.shape[-2] if transpose_x2 else x2.shape[-1]

    # ---- X2 (scale) 模式 (SetX2QuantMode) ----
    if scale_dtype == "float8_e8m0":
        return "G-G"

    scale_last = scale.shape[-1]
    is_x2_perblock = (
        is_fp8_hif8
        and scale_dtype == "float32"
        and scale.ndim > 1
        and (scale.ndim > 2 or scale.shape[0] != 1 or group_size_n > 1)
    )
    is_x2_pertensor = scale_last == 1
    is_x2_perchannel = not is_x2_pertensor and scale_last == n_size

    # ---- X1 (pertoken_scale) 模式 (SetX1QuantMode) ----
    if pertoken_scale is None:
        if is_x2_perblock:
            return "B-B"
        return "T-T" if is_x2_pertensor else "T-C"

    pertoken_last = pertoken_scale.shape[-1]

    is_x1_perblock = (
        is_fp8_hif8 and pertoken_scale.dtype == np.float32 and pertoken_scale.ndim > 1
    )
    if is_x1_perblock and is_x2_perblock:
        return "B-B"

    # doubleScale: 仅 FP8/HIF8 + pertoken 1维 + pertoken[-1]==1 (SetX1QuantMode line 496-497)
    is_double_scale = is_fp8_hif8 and pertoken_scale.ndim == 1 and pertoken_last == 1

    # pertoken: pertoken 1维 + pertoken[-1]==M (SetX1QuantMode line 498-499)
    is_pertoken = pertoken_scale.ndim == 1 and pertoken_last == m_size

    if is_double_scale:
        # IsCubeBasicApiCapable = isDoubleScale && !isPerChannel (tiling_util.h:132)
        if is_x2_perchannel:
            return "T-C-doublescale"  # Mix: 分离乘法, 无截断
        return "T-T-doublescale"  # Cube: merge + 19bit 截断

    if is_pertoken:
        return "K-C"  # Mix: 分离乘法, 无截断

    return "T-T" if is_x2_pertensor else "T-C"


# ============================================================================
# V3 compute functions
# ============================================================================


def compute_mx(
    x1, x2, x2_scale, x1_scale, bias, bias_dtype, transpose_x1, transpose_x2, y_dtype
):
    """
    G-G (MX) 量化模式计算。

    计算流程:
    1. x1/x2 转换为 float32
    2. 将 x1Scale(E8M0) 预乘到 x1，将 x2Scale(E8M0) 预乘到 x2（含 transpose 对齐和 K 轴 padding）
    3. matmul(x1, x2) → 结果已含 scale 效果
    4. + bias (FP32 opt)
    5. cast 到 y_dtype
    """
    x2_scale_mx = x2_scale.copy().astype(np.float32)
    x1_scale_mx = x1_scale.copy().astype(np.float32)
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)

    if transpose_x1:
        x1 = np.swapaxes(x1, -1, -2)
        x1_scale_mx = np.swapaxes(x1_scale_mx, -1, -2)
        if len(x1_scale_mx.shape) == 3:
            x1_scale_mx = x1_scale_mx.reshape(
                x1_scale_mx.shape[0] * x1_scale_mx.shape[1], x1_scale_mx.shape[2]
            )
        x1_scale_mx = np.swapaxes(x1_scale_mx, -1, -2)
    else:
        if len(x1_scale_mx.shape) == 3:
            x1_scale_mx = x1_scale_mx.reshape(
                x1_scale_mx.shape[0], x1_scale_mx.shape[1] * x1_scale_mx.shape[2]
            )

    if transpose_x2:
        x2 = np.swapaxes(x2, -1, -2)
        if len(x2_scale_mx.shape) == 3:
            x2_scale_mx = x2_scale_mx.reshape(
                x2_scale_mx.shape[0], x2_scale_mx.shape[1] * x2_scale_mx.shape[2]
            )
        x2_scale_mx = np.swapaxes(x2_scale_mx, -1, -2)
    else:
        x2_scale_mx = np.swapaxes(x2_scale_mx, -1, -2)
        if len(x2_scale_mx.shape) == 3:
            x2_scale_mx = x2_scale_mx.reshape(
                x2_scale_mx.shape[0] * x2_scale_mx.shape[1], x2_scale_mx.shape[2]
            )

    k_dim = x1.shape[-1]
    if ceil_div(k_dim, 32) % 2 != 0:
        x1_scale_mx = x1_scale_mx[:, :-1]
        x2_scale_mx = x2_scale_mx[:-1, :]

    x1_scale_mx_broadcast = np.repeat(x1_scale_mx, 32, axis=-1)
    x2_scale_mx_broadcast = np.repeat(x2_scale_mx, 32, axis=-2)

    x1_dims = len(x1.shape)
    x2_dims = len(x2.shape)
    x1_pad_len = x1_scale_mx_broadcast.shape[-1] - x1.shape[-1]
    x2_pad_len = x2_scale_mx_broadcast.shape[-2] - x2.shape[-2]
    if x1_pad_len > 0:
        x1 = np.pad(
            x1,
            [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)],
            mode="constant",
            constant_values=0,
        )
    if x2_pad_len > 0:
        x2 = np.pad(
            x2,
            [(0, 0)] * (x2_dims - 2) + [(0, x2_pad_len)] + [(0, 0)],
            mode="constant",
            constant_values=0,
        )

    x1 = x1 * x1_scale_mx_broadcast
    x2 = x2 * x2_scale_mx_broadcast

    y = np.matmul(x1, x2)
    if bias is not None:
        y = cast_output_dtype(y + bias.astype(np.float32), y_dtype)
    else:
        y = cast_output_dtype(y, y_dtype)
    return y


def compute_perblock(
    x1,
    x2,
    x2_scale,
    x1_scale,
    group_size_m,
    group_size_n,
    group_size_k,
    transpose_x1,
    transpose_x2,
    y_dtype,
    bias=None,
):
    """
    B-B (PerBlock) 量化模式计算。

    计算流程:
    1. x1/x2 转换为 float32
    2. 对 x1Scale/x2Scale 做 transpose 对齐 x1/x2 的行列方向
    3. broadcast 对齐 batch 维度
    4. x1Scale 沿 M 轴 repeat(groupSizeM)，x2Scale 沿 N 轴 repeat(groupSizeN)
    5. 分 K 轴 block 级累加: matmul(x1[:,k_start:k_end], x2[k_start:k_end,:]) × (x1Scale[:,k_idx] × x2Scale[k_idx,:])
    6. cast 到 y_dtype

    注: B-B 不支持 bias；仅 FP8/HIF8 可走 PerBlock（INT8/INT4 走 V4）
    """
    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x1_scale_fp32 = x1_scale.astype(np.float32)
    x2_scale_fp32 = x2_scale.astype(np.float32)

    if transpose_x1:
        x1 = np.transpose(x1, gen_axes_for_transpose(len(x1.shape) - 2, [1, 0]))
        x1_scale_fp32 = np.transpose(
            x1_scale_fp32, gen_axes_for_transpose(len(x1_scale_fp32.shape) - 2, [1, 0])
        )
    if transpose_x2:
        x2 = np.transpose(x2, gen_axes_for_transpose(len(x2.shape) - 2, [1, 0]))
        x2_scale_fp32 = np.transpose(
            x2_scale_fp32, gen_axes_for_transpose(len(x2_scale_fp32.shape) - 2, [1, 0])
        )

    batch_x1 = list(x1.shape[:-2])
    batch_x2 = list(x2.shape[:-2])
    batch_x1_scale = list(x1_scale_fp32.shape[:-2])
    batch_x2_scale = list(x2_scale_fp32.shape[:-2])

    all_batches = [b for b in [batch_x1, batch_x2, batch_x1_scale, batch_x2_scale] if b]
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
        if batch_x1_scale != batch_out:
            x1_scale_fp32 = np.broadcast_to(
                x1_scale_fp32, batch_out + list(x1_scale_fp32.shape[-2:])
            )
        if batch_x2_scale != batch_out:
            x2_scale_fp32 = np.broadcast_to(
                x2_scale_fp32, batch_out + list(x2_scale_fp32.shape[-2:])
            )

        batch_all = int(np.prod(batch_out))
        x1 = x1.reshape([batch_all] + list(x1.shape[-2:]))
        x2 = x2.reshape([batch_all] + list(x2.shape[-2:]))
        x1_scale_fp32 = x1_scale_fp32.reshape(
            [batch_all] + list(x1_scale_fp32.shape[-2:])
        )
        x2_scale_fp32 = x2_scale_fp32.reshape(
            [batch_all] + list(x2_scale_fp32.shape[-2:])
        )

    m = x1.shape[-2]
    k = x1.shape[-1]
    n = x2.shape[-1]

    x1_scale_m_expanded = np.repeat(x1_scale_fp32, group_size_m, axis=-2)
    x1_scale_m_expanded = x1_scale_m_expanded[..., :m, :]
    x2_scale_n_expanded = np.repeat(x2_scale_fp32, group_size_n, axis=-1)
    x2_scale_n_expanded = x2_scale_n_expanded[..., :n]

    has_batch = x1.ndim > 2
    if has_batch:
        y = np.zeros((batch_all, m, n), dtype=np.float32)
        for i in range(batch_all):
            for k_idx in range(ceil_div(k, group_size_k)):
                k_start = k_idx * group_size_k
                k_end = min((k_idx + 1) * group_size_k, k)
                x1_scale_col = np.expand_dims(x1_scale_m_expanded[i, :, k_idx], axis=1)
                x2_scale_row = np.expand_dims(x2_scale_n_expanded[i, k_idx, :], axis=0)
                scale_col_row = x1_scale_col * x2_scale_row
                y[i] += (
                    np.matmul(x1[i, :, k_start:k_end], x2[i, k_start:k_end, :])
                    * scale_col_row
                )
        if batch_out:
            y = y.reshape(batch_out + [m, n])
    else:
        y = np.zeros((m, n), dtype=np.float32)
        for k_idx in range(ceil_div(k, group_size_k)):
            k_start = k_idx * group_size_k
            k_end = min((k_idx + 1) * group_size_k, k)
            x1_scale_col = np.expand_dims(x1_scale_m_expanded[:, k_idx], axis=1)
            x2_scale_row = np.expand_dims(x2_scale_n_expanded[k_idx, :], axis=0)
            scale_col_row = x1_scale_col * x2_scale_row
            y += np.matmul(x1[:, k_start:k_end], x2[k_start:k_end, :]) * scale_col_row

    if bias is not None:
        y = y + bias.astype(np.float32)
    return cast_output_dtype(y, y_dtype)


def compute_pertoken(
    matmul_out, x2_scale, x1_scale, bias, bias_dtype, is_bias_vec, y_dtype, x1_dtype
):
    """
    K-C (PerToken) 量化模式计算。

    计算流程:
    1. INT32 bias 前加（int32 域加法，与硬件 fixpipe 一致）
    2. 转 float32：分离 scale 乘法（out × x2Scale × x1Scale，float32 域）
    3. float bias（非 is_bias_vec）和 is_bias_vec 的 bias 均在 scale 乘法后加入
    4. cast 到 y_dtype
    """
    out = matmul_out
    if bias is not None and bias_dtype == "int32":
        out = out + bias

    out = out.astype(np.float32)
    x1_scale_slice = np.expand_dims(x1_scale, axis=1).astype(np.float32)
    out = out * x2_scale.astype(np.float32) * x1_scale_slice

    if not is_bias_vec and bias is not None and bias_dtype in FLOAT_BIAS_DTYPE:
        out = out + bias.astype(np.float32)
    if is_bias_vec and bias is not None:
        return cast_output_dtype(out + bias.astype(np.float32), y_dtype)
    if y_dtype == "float32":
        return out.astype(np.float32)
    return cast_output_dtype(out, y_dtype)


def compute_tc(
    matmul_out,
    x2_scale,
    bias,
    bias_dtype,
    do_scale_gen,
    is_bias_vec,
    y_dtype,
    pertoken_scale=None,
    x1_dtype=None,
    scale_dtype=None,
    do_merge_trunc=False,
):
    """
    T-C/T-T 量化模式计算。

    customize_inputs 已将 UINT64 scale 转为 float32 deq_scale，
    T-C-static(uint64) 和 T-C-dynamic(float32/bf16) 合并处理，bias 位置由 is_bias_vec 决定。

    doubleScale 按 kernel 流水线区分（由 do_merge_trunc 控制）：
    - Cube 路径 (do_merge_trunc=True and not is_bias_vec): merge=scale*pertoken → scale_generate 19-bit 截断 → apply
      对应 qbmm_cube_on_the_fly.h:159-162: deqScale = scale * pertoken; & DEQ_SCALE_MUL
      IsNonMxCubeBasicApiCapable 排除 IsFp8OrHif8TTFloatBiasMix 场景 (tiling_util.h:176-180)
    - Mix  路径 (is_bias_vec 或 do_merge_trunc=False): 分离乘法 out*scale → *pertoken，无截断
      对应 qbmm_mix_online_dynamic.h:684-707: Muls(out, scaleScalar) → Muls(out, pertokenScalar)
      IsFp8OrHif8TTFloatBiasMix 场景 bias 在 scale 后加 (isBiasEpilogue_=True)

    计算流程:
    1. 非 is_bias_vec: + bias（int32 域/float32 域加法，反量化前加入）
    2. 转 float32
    3. doubleScale: cube 路径合并乘法 + 19-bit 截断；mix 路径分离乘法
    4. 普通 T-C/T-T: 若 do_scale_gen=True 截断 x2Scale 高 19 位 → out × x2Scale
    5. is_bias_vec 时 + float bias（fixpipe 后处理，反量化后加入）
    6. cast 到 y_dtype
    """
    out = matmul_out

    if not is_bias_vec and bias is not None:
        if bias_dtype == "int32":
            out = out + bias
        elif bias_dtype in FLOAT_BIAS_DTYPE:
            out = out.astype(np.float32) + bias.astype(np.float32)

    out = out.astype(np.float32)

    if pertoken_scale is not None:
        if do_merge_trunc and not is_bias_vec:
            # Cube 路径: merge + 19-bit 截断 (qbmm_cube_on_the_fly.h:159-162)
            # IsNonMxCubeBasicApiCapable 排除 IsFp8OrHif8TTFloatBiasMix 场景
            merged = x2_scale.astype(np.float32) * pertoken_scale.astype(np.float32)
            merged = scale_generate(merged)
            if merged.ndim == 1:
                merged = np.expand_dims(merged, axis=0)
            out = out * merged
        else:
            # Mix 路径: 分离乘法, 无截断 (qbmm_mix_online_dynamic.h:684-707)
            if do_scale_gen:
                x2_scale = scale_generate(x2_scale)
            if x2_scale.ndim == 1:
                x2_scale = np.expand_dims(x2_scale, axis=0)
            out = out * x2_scale.astype(np.float32) * pertoken_scale.astype(np.float32)
    else:
        if do_scale_gen:
            x2_scale = scale_generate(x2_scale)
        if x2_scale.ndim == 1:
            x2_scale = np.expand_dims(x2_scale, axis=0)
        out = out * x2_scale.astype(np.float32)

    if is_bias_vec and bias is not None:
        return cast_output_dtype(out + bias.astype(np.float32), y_dtype)
    if y_dtype == "float32":
        return out.astype(np.float32)
    return cast_output_dtype(out, y_dtype)


def compute_requant(matmul_out, x2_scale, offset, bias, bias_dtype):
    """
    Requant (y=INT8) 计算流程 — T-C 的输出 dtype 变体。

    计算流程:
    1. + INT32 bias（int32 域加法，反量化前加入）
    2. 转 float32: f32_2_s9(out × x2Scale) — float32 域乘法 + 9bit 量化截断
    3. + f32_2_s9(offset)（若有）
    4. clip[-128, 127] → int8

    注: 仅 INT8×INT8+UINT64 scale 支持 requant
    """
    from matmul_quant_util import f32_2_s9

    out = matmul_out
    if bias is not None and bias_dtype == "int32":
        out = out + bias

    out = out.astype(np.float32)
    out = f32_2_s9(out * x2_scale.astype(np.float32))
    if offset is not None:
        out = f32_2_s9(out) + f32_2_s9(offset)
    return np.clip(out, -128, 127).astype(np.int8)


def compute_int32(matmul_out, bias, bias_dtype):
    """
    纯整数 (y=INT32) 计算流程 — scale 不参与计算的输出 dtype 变体。

    计算流程:
    1. + INT32 bias（int32 域加法）；float32 bias 在 float32 域加法
    2. cast 到 int32

    注: 仅 INT8×INT8 支持；scale 不参与，bias 位置不影响最终结果
    """
    out = matmul_out
    if bias is not None:
        if bias_dtype == "int32":
            out = out + bias
        else:
            out = out.astype(np.float32) + bias.astype(np.float32)
    return out.astype(np.int32)


def qbmmv3_kernel_compute(
    x1,
    x2,
    scale,
    offset,
    bias,
    pertoken_scale,
    *,
    dtype,
    transpose_x1,
    transpose_x2,
    group_size,
    **kwargs,
):
    """
    V3 kernel 级别 numpy 模拟计算入口。

    参数签名对应 V3 算子入参：x1, x2, scale, offset, bias, pertoken_scale

    根据量化模式分发到对应的 compute 函数：
    - G-G (MX): compute_mx
    - B-B (PerBlock): compute_perblock
    - K-C (PerToken): compute_pertoken
    - T-C (PerTensor/PerChannel): compute_tc / compute_requant / compute_int32
    """
    # 获取 scale 的原始 dtype（用于判断是否需要 scale_generate）
    scale_dtype = kwargs.get("scale_dtype", dtype_to_str(scale.dtype))

    # 获取输出 dtype
    y_dtype = kwargs.get("output_dtypes", ["float32"])[0]

    # 转换 dtype 为字符串
    x1_dtype = dtype_to_str(x1.dtype)
    x2_dtype = dtype_to_str(x2.dtype)
    bias_dtype = dtype_to_str(bias.dtype) if bias is not None else None

    # 准备 scale（MX 模式不转 float32）
    x2_scale = scale.astype(np.float32) if scale.dtype != np_mx_scale else scale
    x1_scale = pertoken_scale

    # 解包 group_size
    group_size_m, group_size_n, group_size_k = unpack_groupsize(group_size)

    # 判定量化模式
    output_shapes = kwargs.get("output_ori_shapes", kwargs.get("output_shapes", ()))
    quant_mode = determine_quant_mode(
        x1,
        x2,
        x2_scale,
        x1_scale,
        scale_dtype,
        transpose_x1,
        transpose_x2,
        group_size_n,
        output_shapes,
    )

    # 判断是否需要 scale_generate
    do_scale_gen = needs_scale_generate(x1_dtype, x2_scale, bias_dtype, scale_dtype)

    # 判断 bias 是否为向量（影响 bias 加法位置）
    is_bias_vec = bias_dtype is not None and (
        (x1_dtype in INT_QUANT_INPUT_DTYPE and bias_dtype in FLOAT_BIAS_DTYPE)
        or (
            x1_dtype in PERBLOCK_INPUT_DTYPE
            and bias_dtype in FLOAT_BIAS_DTYPE
            and pertoken_scale is not None
        )
    )

    # 准备 offset
    if offset is not None:
        offset = offset.astype(np.float32)
        if x2_scale.shape[-1] < offset.shape[-1]:
            offset = offset[0]

    # 根据量化模式分发计算
    if quant_mode == "G-G":
        y = compute_mx(
            x1,
            x2,
            x2_scale,
            x1_scale,
            bias,
            bias_dtype,
            transpose_x1,
            transpose_x2,
            y_dtype,
        )
    elif quant_mode == "B-B":
        y = compute_perblock(
            x1,
            x2,
            x2_scale,
            x1_scale,
            group_size_m,
            group_size_n,
            group_size_k,
            transpose_x1,
            transpose_x2,
            y_dtype,
            bias,
        )
    else:
        # T-C / K-C 模式：先做 matmul
        if x1_dtype in INT_QUANT_INPUT_DTYPE and x2_dtype in INT_QUANT_INPUT_DTYPE:
            x1 = x1.astype(np.int32)
            x2 = x2.astype(np.int32)
        else:
            x1 = x1.astype(np.float32)
            x2 = x2.astype(np.float32)

        if transpose_x1:
            x1 = np.transpose(
                x1,
                [x for x in range(len(x1.shape) - 2)]
                + [len(x1.shape) - 1, len(x1.shape) - 2],
            )
        if transpose_x2:
            x2 = np.transpose(
                x2,
                [x for x in range(len(x2.shape) - 2)]
                + [len(x2.shape) - 1, len(x2.shape) - 2],
            )

        matmul_out = np.matmul(x1, x2)

        # 根据输出 dtype 和量化模式选择 compute 函数
        if y_dtype == "int8":
            y = compute_requant(matmul_out, x2_scale, offset, bias, bias_dtype)
        elif y_dtype == "int32":
            y = compute_int32(matmul_out, bias, bias_dtype)
        elif quant_mode == "K-C":
            y = compute_pertoken(
                matmul_out,
                x2_scale,
                x1_scale,
                bias,
                bias_dtype,
                is_bias_vec,
                y_dtype,
                x1_dtype,
            )
        elif quant_mode == "T-T-doublescale":
            y = compute_tc(
                matmul_out,
                x2_scale,
                bias,
                bias_dtype,
                do_scale_gen,
                is_bias_vec,
                y_dtype,
                pertoken_scale=x1_scale,
                x1_dtype=x1_dtype,
                scale_dtype=scale_dtype,
                do_merge_trunc=True,
            )
        elif quant_mode == "T-C-doublescale":
            y = compute_tc(
                matmul_out,
                x2_scale,
                bias,
                bias_dtype,
                do_scale_gen,
                is_bias_vec,
                y_dtype,
                pertoken_scale=x1_scale,
                x1_dtype=x1_dtype,
                scale_dtype=scale_dtype,
                do_merge_trunc=False,
            )
        elif quant_mode in ("T-C", "T-T"):
            y = compute_tc(
                matmul_out,
                x2_scale,
                bias,
                bias_dtype,
                do_scale_gen,
                is_bias_vec,
                y_dtype,
                pertoken_scale=None,
                x1_dtype=x1_dtype,
                scale_dtype=scale_dtype,
            )
        else:
            # fallback
            if matmul_out.dtype == np.int32:
                matmul_out = matmul_out.astype(np.float32)
            y = cast_output_dtype(matmul_out * x2_scale, y_dtype)

    return [y]


class QuantBatchMatmulV3TestSpec:
    """quant_batch_matmul_v3 算子的 TestSpec 类。"""

    @staticmethod
    def golden(
        x1,
        x2,
        scale,
        offset=None,
        bias=None,
        pertoken_scale=None,
        *,
        dtype: int,
        transpose_x1: bool = False,
        transpose_x2: bool = False,
        group_size: int = 0,
        **kwargs,
    ):
        """
        V3 kernel golden: NZ→ND 转换 + UINT64 解码 + 调用 kernel 计算。

        处理流程：
        1. customize_inputs: UINT64→deq_scale, NZ→ND
        2. 调用 qbmmv3_kernel_compute
        """
        # customize_inputs 逻辑
        if scale.dtype in (np.uint64, np.int64):
            deq_scale = u64_to_deq_scale(scale)
            if offset is not None:
                offset = u64_to_offset(scale)
        else:
            deq_scale = scale

        input_formats = kwargs.get("input_formats", ())
        input_ori_shapes = kwargs.get("input_ori_shapes", ())

        if len(input_formats) > 1 and input_formats[1] == "FRACTAL_NZ":
            ori_shape = input_ori_shapes[1] if len(input_ori_shapes) > 1 else None
            x2 = nz_to_nd(x2, ori_shape)

        if (
            len(input_formats) > 2
            and input_formats[2] == "FRACTAL_NZ"
            and kwargs.get("scale_dtype") not in ("uint64", "int64")
        ):
            ori_shape = input_ori_shapes[2] if len(input_ori_shapes) > 2 else None
            deq_scale = nz_to_nd(deq_scale, ori_shape)

        # 调用 kernel compute
        return qbmmv3_kernel_compute(
            x1,
            x2,
            deq_scale,
            offset,
            bias,
            pertoken_scale,
            dtype=dtype,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            group_size=group_size,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale,
        offset=None,
        bias=None,
        pertoken_scale=None,
        *,
        dtype: int,
        transpose_x1: bool = False,
        transpose_x2: bool = False,
        group_size: int = 0,
        **kwargs,
    ):
        """
        输入预处理：清洗 E8M0 NaN + UINT64 scale 生成。
        """
        testcase_name = kwargs.get("testcase_name", "unknown")
        output_dtypes = kwargs.get("output_dtypes", ["float32"])
        out_dtype = output_dtypes[0]

        # 清洗 E8M0 NaN
        x1 = sanitize_e8m0_scale(x1, testcase_name)
        x2 = sanitize_e8m0_scale(x2, testcase_name)
        scale = sanitize_e8m0_scale(scale, testcase_name)
        pertoken_scale = sanitize_e8m0_scale(pertoken_scale, testcase_name)

        # UINT64 scale 生成
        if scale.dtype in ("uint64", "int64"):
            deq_scale_shape = scale.shape
            scale_orig_dtype = scale.dtype
            # 生成随机 scale
            fp32_deq_scale = np.random.uniform(
                low=-5, high=5, size=deq_scale_shape
            ).astype(np.float32)
            uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(
                deq_scale_shape
            )
            uint32_deq_scale &= 0xFFFFE000

            if out_dtype != "int8":
                scale = np.zeros(deq_scale_shape, np.uint64)
                scale |= uint32_deq_scale.astype(np.uint64)
            else:
                # INT8 输出需要 offset
                s9_offset = 0
                if offset is not None:
                    fp32_offset = np.random.uniform(
                        low=-5, high=5, size=offset.shape
                    ).astype(np.float32)
                    s9_offset = (
                        np.clip(np.round(fp32_offset), -256, 255)
                        .astype(int)
                        .reshape(offset.shape)
                    )
                    s9_offset &= 0x1FF
                    s9_offset = (
                        s9_offset[0]
                        if deq_scale_shape[-1] < offset.shape[-1]
                        else s9_offset
                    )
                scale = np.zeros(deq_scale_shape, np.uint64)
                scale |= uint32_deq_scale.astype(np.uint64)
                scale |= (np.asarray(s9_offset, dtype=np.int64) << 37).astype(np.uint64)
                scale |= 1 << 46

            if scale_orig_dtype == "int64":
                scale = scale.astype(np.int64)

        return x1, x2, scale, offset, bias, pertoken_scale


__spec__ = {
    "quant_batch_matmul_v3": "QuantBatchMatmulV3TestSpec",
}
