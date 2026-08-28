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
SwigluGroupGrad 算子输入数据与 golden 输出生成脚本。

用法：
    python3 gen_input_data.py [--output-dir DIR] [--cases CASE_SPEC]

功能：
    - 生成各种 shape 组合的测试输入数据（numpy .npy 格式）
    - 支持 BF16/FP16/FP32 三种精度
    - 支持 hasClamp=0/1, isWeight=0/1, isGroupIndex=0/1 的不同组合
    - 计算正确的 golden（期望输出）数据
    - 所有中间计算在 FP32 上完成（与 kernel 行为一致）
    - 保存为 .npy 文件方便后续 ST 测试使用

输出文件命名规则：
    testdata_{case_name}/
        input_grad_y_{dtype}.npy     -- gradY 输入
                                           (BF16 以 uint16 存储, 用 load_bf16_npy() 加载)
        input_x_{dtype}.npy               -- x 输入
                                           (BF16 以 uint16 存储, 用 load_bf16_npy() 加载)
        input_weight.npy                  -- weight 输入 (FP32, 仅 isWeight=1)
        input_y_origin_{dtype}.npy        -- yOrigin 输入 (仅 isWeight=1)
                                           (BF16 以 uint16 存储, 用 load_bf16_npy() 加载)
        input_group_index.npy             -- groupIndex 输入 (INT64, 仅 isGroupIndex=1)
        attr_clamp_limit.npy              -- clampLimit 属性 (FP32 scalar)
        golden_grad_x_fp32.npy            -- golden gradX (FP32, 精确参考值)
        golden_grad_x_{dtype}.npy         -- golden gradX (目标 dtype, BF16 以 uint16 存储)
        golden_grad_weight.npy            -- golden gradWeight 输出 (FP32, 仅 isWeight=1)
        case_info.json                    -- 测试用例元信息

    注意: numpy .npy 格式不支持直接保存 ml_dtypes.bfloat16 dtype。
    BF16 数据以 uint16 (原始位模式) 保存，加载后需用 .view(ml_dtypes.bfloat16) 转换。
    使用 load_npy() 辅助函数可自动处理此转换。

数学公式（与算子接口约束一致）：
    前向分解：
        F0: m_r = I(r < trunc)              -- groupIndex mask (trunc = Σ groupIndex)
        F1: g_tilde = min(c, g)             -- gate 单侧 clamp (仅上侧)
        F2: u_tilde = clip(u, -c, c)        -- up 双侧 clip
        F3: s = sigmoid(g_tilde) = 1/(1+exp(-g_tilde))
        F4: f = silu(g_tilde) = g_tilde * s
        F5: p = f * u_tilde
        F6: y = p * w_t                     -- w_t = weight 或 1.0
        F7: y_fp8 = Q(y)                    -- STE 透传, 反向不感知

    反向梯度（链式法则）：
        silu'(g_tilde) = s + f - f*s        -- 数值改写, 避免 ∞·0
        m_g = I(g < c)                       -- 开区间约定 (g=c => m_g=0)
        m_u = I(-c < u < c)                  -- 开区间约定 (u=±c => m_u=0)
        m_r = I(r < trunc)                   -- groupIndex mask

        dg = gradY * silu'(g_tilde) * u_tilde * w_t * m_g * m_r
        du = gradY * f * w_t * m_u * m_r
        gradWeight = Σ(gradY · yOrigin, axis=-1)   -- 无 mask!
        gradX[..., :H] = dg;  gradX[..., H:] = du
"""

import argparse
import json
import os

import numpy as np

# BF16 support via ml_dtypes (available in numpy 1.26+ environments with ml_dtypes)
try:
    import ml_dtypes

    BF16_DTYPE = ml_dtypes.bfloat16
    HAS_BF16 = True
except ImportError:
    HAS_BF16 = False

# ============================================================================
# Dtype utility functions
# ============================================================================

DTYPE_MAP = {
    "bf16": None,  # Will be set below
    "fp16": np.float16,
    "fp32": np.float32,
}

if HAS_BF16:
    DTYPE_MAP["bf16"] = BF16_DTYPE


def get_dtype(name):
    """Get numpy dtype by short name (bf16/fp16/fp32)."""
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype name: {name}. Supported: bf16, fp16, fp32")
    dt = DTYPE_MAP[name]
    if dt is None:
        raise RuntimeError("BF16 dtype not available (ml_dtypes not installed)")
    return dt


def dtype_bytes(name):
    """Return bytes per element for a dtype name."""
    return (
        {True: 2, False: 2}[name == "bf16"]
        if name == "bf16"
        else {2: 2, 4: 4}[np.dtype(get_dtype(name)).itemsize]
    )


def cast_to_target(array, dtype_name):
    """Cast a FP32 numpy array to the target dtype, with proper rounding."""
    target_dt = get_dtype(dtype_name)
    if dtype_name == "bf16":
        # ml_dtypes.bfloat16 cast: round-to-nearest-even (RINT), matching kernel behavior
        return array.astype(target_dt)
    elif dtype_name == "fp16":
        # FP16 cast: round-to-nearest-even
        return array.astype(target_dt)
    else:
        # FP32: no cast needed
        return array.astype(target_dt)


def cast_to_fp32(array):
    """Cast any dtype array to FP32 for computation.

    Handles: native FP32, FP16, ml_dtypes.bfloat16, and uint16-view BF16.
    """
    if array.dtype == np.uint16 and HAS_BF16:
        # This is a BF16 array stored as uint16 (raw bit pattern)
        return array.view(BF16_DTYPE).astype(np.float32)
    elif HAS_BF16 and array.dtype == BF16_DTYPE:
        return array.astype(np.float32)
    else:
        return array.astype(np.float32)


def save_npy_bf16_safe(arr, filepath):
    """Save a numpy array to .npy file with BF16-safe handling.

    BF16 arrays are saved as uint16 (raw bit pattern) because numpy .npy
    format does not support ml_dtypes.bfloat16 dtype natively.
    Other dtypes (FP16, FP32, INT64) are saved directly.
    """
    if HAS_BF16 and arr.dtype == BF16_DTYPE:
        # Save BF16 as uint16 (preserves exact bit patterns)
        arr_uint16 = arr.view(np.uint16)
        np.save(filepath, arr_uint16)
    else:
        np.save(filepath, arr)


def load_npy(filepath, dtype_name=None):
    """Load a numpy array from .npy file with BF16-safe handling.

    If dtype_name='bf16', converts uint16 storage to ml_dtypes.bfloat16.
    Other dtypes are loaded directly.

    Args:
        filepath:   Path to .npy file
        dtype_name: Expected dtype name ('bf16', 'fp16', 'fp32', or None for auto)

    Returns:
        numpy array in the requested dtype
    """
    arr = np.load(filepath)
    if dtype_name == "bf16" and HAS_BF16 and arr.dtype == np.uint16:
        return arr.view(BF16_DTYPE)
    elif dtype_name == "bf16" and not HAS_BF16:
        # Return as uint16 if ml_dtypes unavailable
        return arr
    return arr


# ============================================================================
# Golden computation (matches golden.py)
# ============================================================================


def compute_golden(
    grad_y, x, weight=None, y_origin=None, group_index=None, clamp_limit=-1.0
):
    """
    Compute golden (expected) output for SwigluGroupGrad.

    All computation is done in FP32 (matching kernel behavior).
    Input arrays can be any supported dtype; they are cast to FP32 internally.

    Args:
        grad_y: numpy array [..., H], BF16/FP16/FP32
        x:           numpy array [..., 2H], BF16/FP16/FP32
        weight:      numpy array [..., 1], FP32, or None
        y_origin:    numpy array [..., H], BF16/FP16/FP32, or None
        group_index: numpy array [G], INT64, or None
        clamp_limit: float, -1.0 = no clamp

    Returns:
        (grad_x, grad_weight):
            grad_x:      FP32 numpy array [..., 2H]
            grad_weight: FP32 numpy array [..., 1], or None
    """
    if (weight is None) != (y_origin is None):
        raise ValueError("weight and y_origin must be provided together")

    # ── Handle empty tensor ──
    if x.size == 0:
        grad_x = np.zeros(x.shape, dtype=np.float32)
        grad_weight = (
            np.zeros(weight.shape, dtype=np.float32) if weight is not None else None
        )
        return grad_x, grad_weight

    # ── Promote to FP32 ──
    x_f = cast_to_fp32(x)
    dy_f = cast_to_fp32(grad_y)

    # ── Split x into gate (g) and up (u) ──
    H = x_f.shape[-1] // 2
    g = x_f[..., :H]  # gate branch (original, before clamp)
    u = x_f[..., H:]  # up branch   (original, before clip)

    # ── Clamp ──
    c = float(clamp_limit)
    if c != -1.0 and not c > 0.0:
        raise ValueError(
            f"clamp_limit must be -1.0 (no clamp) or > 0.0, but got {clamp_limit}"
        )
    has_clamp = c > 0.0

    if has_clamp:
        g_tilde = np.minimum(g, c)
        u_tilde = np.clip(u, -c, c)
        # Open interval masks (critical for boundary correctness)
        m_g = (g < c).astype(np.float32)
        m_u = ((u > -c) & (u < c)).astype(np.float32)
    else:
        g_tilde = g
        u_tilde = u
        m_g = np.ones_like(g, dtype=np.float32)
        m_u = np.ones_like(u, dtype=np.float32)

    # ── GroupIndex row mask ──
    outer_shape = g.shape[:-1]
    total_rows = int(np.prod(outer_shape))

    if group_index is not None:
        trunc = int(np.sum(group_index))
        valid_rows = min(trunc, total_rows)
        m_r_flat = np.zeros(total_rows, dtype=np.float32)
        if valid_rows > 0:
            m_r_flat[:valid_rows] = 1.0
        m_r = m_r_flat.reshape(outer_shape + (1,))
    else:
        m_r = np.ones(outer_shape + (1,), dtype=np.float32)

    # ── Weight ──
    if weight is not None:
        w = cast_to_fp32(weight)
    else:
        w = np.ones(outer_shape + (1,), dtype=np.float32)

    # ── Core FP32 computation ──
    # Sigmoid
    s = 1.0 / (1.0 + np.exp(-g_tilde))
    # SiLU
    f = g_tilde * s
    # SiLU derivative (numerical rewrite)
    silu_prime = s + f - f * s

    # dg = gradY * silu'(g_tilde) * u_tilde * w_t * m_g * m_r
    dg = dy_f * silu_prime * u_tilde * w * m_g * m_r

    # du = gradY * f * w_t * m_u * m_r
    du = dy_f * f * w * m_u * m_r

    # ── Concat to grad_x ──
    grad_x = np.concatenate([dg, du], axis=-1)

    # ── gradWeight ──
    grad_weight = None
    if weight is not None:
        y_origin_f = cast_to_fp32(y_origin)
        # grad_weight = Σ(gradY · yOrigin) along hidden dim — NO mask!
        grad_weight = np.sum(dy_f * y_origin_f, axis=-1, keepdims=True)

    return grad_x, grad_weight


# ============================================================================
# Input data generation
# ============================================================================


def generate_forward_output(x, weight=None, clamp_limit=-1.0):
    """
    Generate the forward output y_origin for ClampedSwiglu (needed for gradWeight).

    Forward: y = silu(g_tilde) * u_tilde * w_t
    """
    x_f = cast_to_fp32(x)
    H = x_f.shape[-1] // 2
    g = x_f[..., :H]
    u = x_f[..., H:]

    c = float(clamp_limit)
    if c > 0:
        g_tilde = np.minimum(g, c)
        u_tilde = np.clip(u, -c, c)
    else:
        g_tilde = g
        u_tilde = u

    s = 1.0 / (1.0 + np.exp(-g_tilde))
    f = g_tilde * s
    p = f * u_tilde

    if weight is not None:
        w = cast_to_fp32(weight)
        y = p * w
    else:
        y = p

    return y


def make_random_array(shape, dtype_name, seed=None, value_range=(-3.0, 3.0)):
    """Generate a random numpy array of given shape and dtype."""
    rng = np.random.default_rng(seed)
    # Generate in FP32 then cast to target dtype (ensures proper rounding)
    fp32_data = rng.uniform(value_range[0], value_range[1], size=shape).astype(
        np.float32
    )
    return cast_to_target(fp32_data, dtype_name)


def make_group_index(total_rows, num_groups, seed=None):
    """
    Generate a groupIndex tensor [G] of INT64.

    Each element represents the number of tokens for that group.
    The sum of groupIndex should be <= total_rows.
    """
    rng = np.random.default_rng(seed)
    # Distribute total_rows across num_groups
    # Use random partitioning: each group gets at least 1 row
    if num_groups <= 0:
        num_groups = 1
    if total_rows < num_groups:
        # Not enough rows for all groups; some groups get 0
        gi = np.zeros(num_groups, dtype=np.int64)
        gi[:total_rows] = 1
        return gi

    # Random partition: generate num_groups-1 random cut points
    cuts = rng.choice(range(1, total_rows), size=num_groups - 1, replace=False)
    cuts = np.sort(cuts)
    cuts = np.concatenate([[0], cuts, [total_rows]])
    gi = np.diff(cuts).astype(np.int64)
    return gi


# ============================================================================
# Test case definitions
# ============================================================================

# Shape configurations: (name, x_shape, H). x is always [..., 2H].
SHAPE_CONFIGS = [
    # Small shapes
    ("tiny_2d", (4, 32), None),  # T=4, 2H=32, H=16
    ("tiny_3d", (2, 4, 32), None),  # B=2, S=4, 2H=32, H=16
    ("small_2d", (8, 64), None),  # T=8, 2H=64, H=32
    ("small_2d_h256", (4, 512), None),  # T=4, 2H=512, H=256
    # Medium shapes
    ("medium_2d", (32, 256), None),  # T=32, 2H=256, H=128
    ("medium_2d_h1024", (16, 2048), None),  # T=16, 2H=2048, H=1024
    # Large shapes (performance-oriented)
    ("large_2d", (2048, 4096), None),  # T=2048, 2H=4096, H=2048
    ("large_2d_h2048", (1024, 2048), None),  # T=1024, 2H=2048, H=1024
]

# Feature combination matrix: (hasClamp, isWeight, isGroupIndex)
FEATURE_COMBOS = [
    # Basic: no optional features
    (0, 0, 0),  # No clamp, no weight, no groupIndex — simplest path
    # Clamp only
    (1, 0, 0),  # With clamp only — tests mask computation
    # Weight only (requires yOrigin)
    (0, 1, 0),  # With weight — tests w_t broadcast + gradWeight
    # Weight + Clamp
    (1, 1, 0),  # Weight + Clamp — tests full mask + w_t + gradWeight
    # GroupIndex only
    (0, 0, 1),  # With groupIndex — tests m_r mask + trunc filtering
    # GroupIndex + Clamp
    (1, 0, 1),  # GroupIndex + Clamp — tests m_r + m_g + m_u
    # GroupIndex + Weight
    (0, 1, 1),  # GroupIndex + Weight — tests m_r + w_t + gradWeight trunc
    # All features
    (1, 1, 1),  # Full combination — tests all paths simultaneously
]

# Dtype configurations
DTYPE_CONFIGS = ["bf16", "fp16", "fp32"]

# Clamp limit values to use when hasClamp=1
CLAMP_LIMIT_VALUES = [3.0, 5.0, 10.0]


# ============================================================================
# Main generation logic
# ============================================================================


def generate_case(
    case_name,
    x_shape,
    dtype_name,
    has_clamp,
    is_weight,
    is_group_index,
    clamp_limit,
    seed=42,
    output_dir=None,
):
    """
    Generate one complete test case: input data + golden output.

    Args:
        case_name:    String identifier for this case
        x_shape:      Shape tuple for x tensor (..., 2H)
        dtype_name:   'bf16', 'fp16', or 'fp32'
        has_clamp:    0 or 1
        is_weight:    0 or 1
        is_group_index: 0 or 1
        clamp_limit:  float value (0.0 when has_clamp=0)
        seed:         Random seed for reproducibility
        output_dir:   Directory to save .npy files (default: auto-generated)

    Returns:
        case_info dict with metadata about this test case
    """
    rng_seed = seed
    # Derive shapes
    H = x_shape[-1] // 2
    outer_dims = x_shape[:-1]
    grad_y_shape = outer_dims + (H,)
    weight_shape = outer_dims + (1,)
    y_origin_shape = outer_dims + (H,)
    grad_x_shape = x_shape
    grad_weight_shape = outer_dims + (1,)

    total_rows = int(np.prod(outer_dims))
    num_groups = min(4, total_rows)  # Use at most 4 groups for groupIndex

    # ── Generate input data ──

    # grad_y: random values in moderate range
    grad_y = make_random_array(
        grad_y_shape, dtype_name, seed=rng_seed, value_range=(-1.0, 1.0)
    )  # grad values typically smaller
    rng_seed += 1000

    # x: random values covering various ranges for gate and up
    # Gate values: wider range to exercise clamp boundaries
    # Up values: moderate range
    x = make_random_array(x_shape, dtype_name, seed=rng_seed, value_range=(-5.0, 5.0))
    rng_seed += 1000

    # Insert some boundary values for clamp testing
    if has_clamp and clamp_limit > 0:
        x_fp32 = cast_to_fp32(x)
        # Insert values exactly at clamp boundary (g=c, u=±c) to test open-interval
        # For a small fraction of elements, set gate = clamp_limit exactly
        rng_np = np.random.default_rng(rng_seed + 1)
        # ~5% of elements at exact boundary
        flat_gate = x_fp32[..., :H].flatten()
        num_boundary = max(1, int(0.05 * flat_gate.size))
        boundary_indices = rng_np.choice(
            flat_gate.size, size=min(num_boundary, flat_gate.size), replace=False
        )
        flat_gate[boundary_indices] = clamp_limit  # g = c exactly
        # Also insert some u = ±c values
        flat_up = x_fp32[..., H:].flatten()
        num_boundary_u = max(1, int(0.05 * flat_up.size))
        boundary_indices_u = rng_np.choice(
            flat_up.size, size=min(num_boundary_u, flat_up.size), replace=False
        )
        flat_up[boundary_indices_u] = clamp_limit  # u = c exactly
        # Another set at u = -c
        boundary_indices_u_neg = rng_np.choice(
            flat_up.size, size=min(num_boundary_u, flat_up.size), replace=False
        )
        flat_up[boundary_indices_u_neg] = -clamp_limit  # u = -c exactly

        # Reconstruct x from modified gate and up
        new_x_fp32 = np.concatenate(
            [
                flat_gate.reshape(x_fp32.shape[:-1] + (H,)),
                flat_up.reshape(x_fp32.shape[:-1] + (H,)),
            ],
            axis=-1,
        )
        x = cast_to_target(new_x_fp32, dtype_name)
        rng_seed += 2000

    # weight: random positive weights (MoE router weights)
    weight = None
    y_origin = None
    if is_weight:
        weight_rng = np.random.default_rng(rng_seed)
        weight_fp32 = weight_rng.uniform(0.5, 2.0, size=weight_shape).astype(np.float32)
        weight = weight_fp32  # weight is always FP32
        rng_seed += 1000

        # y_origin: compute from forward pass (matching real usage)
        y_origin_fp32 = generate_forward_output(
            x, weight=weight, clamp_limit=clamp_limit
        )
        y_origin = cast_to_target(y_origin_fp32, dtype_name)

    # group_index: random partitioning of total_rows
    group_index = None
    if is_group_index:
        group_index = make_group_index(total_rows, num_groups, seed=rng_seed)
        rng_seed += 1000

    # ── Compute golden output ──
    grad_x_golden, grad_weight_golden = compute_golden(
        grad_y,
        x,
        weight=weight,
        y_origin=y_origin,
        group_index=group_index,
        clamp_limit=clamp_limit,
    )

    # Cast golden grad_x back to input dtype (matching kernel output dtype)
    # The golden computation was in FP32; cast to target dtype for comparison
    grad_x_golden_target = cast_to_target(grad_x_golden, dtype_name)
    # grad_weight is always FP32 (no cast needed)

    # ── Save to files ──
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "testdata", case_name
        )
    os.makedirs(output_dir, exist_ok=True)

    def save_npy(arr, filename):
        filepath = os.path.join(output_dir, filename)
        save_npy_bf16_safe(arr, filepath)
        return filepath

    # Save inputs
    saved_files = {}
    saved_files["grad_y"] = save_npy(grad_y, f"input_grad_y_{dtype_name}.npy")
    saved_files["x"] = save_npy(x, f"input_x_{dtype_name}.npy")

    if weight is not None:
        saved_files["weight"] = save_npy(weight, "input_weight.npy")
    if y_origin is not None:
        saved_files["y_origin"] = save_npy(y_origin, f"input_y_origin_{dtype_name}.npy")
    if group_index is not None:
        saved_files["group_index"] = save_npy(group_index, "input_group_index.npy")

    # Save attributes
    clamp_limit_arr = np.array([clamp_limit], dtype=np.float32)
    saved_files["clamp_limit"] = save_npy(clamp_limit_arr, "attr_clamp_limit.npy")

    # Save golden outputs
    # Always save FP32 golden for precise comparison (regardless of input dtype)
    saved_files["grad_x_golden_fp32"] = save_npy(
        grad_x_golden, "golden_grad_x_fp32.npy"
    )
    # Also save target-dtype golden for dtype-specific comparison
    saved_files["grad_x_golden"] = save_npy(
        grad_x_golden_target, f"golden_grad_x_{dtype_name}.npy"
    )

    if grad_weight_golden is not None:
        saved_files["grad_weight_golden"] = save_npy(
            grad_weight_golden, "golden_grad_weight.npy"
        )

    # ── Save case info ──
    case_info = {
        "case_name": case_name,
        "dtype": dtype_name,
        "x_shape": list(x_shape),
        "grad_y_shape": list(grad_y_shape),
        "grad_x_shape": list(grad_x_shape),
        "H": H,
        "has_clamp": has_clamp,
        "is_weight": is_weight,
        "is_group_index": is_group_index,
        "clamp_limit": clamp_limit,
        "total_rows": total_rows,
        "weight_shape": list(weight_shape) if is_weight else None,
        "y_origin_shape": list(y_origin_shape) if is_weight else None,
        "grad_weight_shape": list(grad_weight_shape) if is_weight else None,
        "group_index_shape": list(group_index.shape) if is_group_index else None,
        "group_index_sum": int(np.sum(group_index)) if is_group_index else None,
        "num_groups": num_groups if is_group_index else None,
        "seed": seed,
        "saved_files": {k: os.path.basename(v) for k, v in saved_files.items()},
    }

    with open(os.path.join(output_dir, "case_info.json"), "w") as f:
        json.dump(case_info, f, indent=2)

    return case_info


def generate_all_cases(
    output_dir=None,
    shape_filter=None,
    dtype_filter=None,
    combo_filter=None,
    seed_base=42,
):
    """
    Generate all test cases based on the defined configurations.

    Args:
        output_dir:     Base directory for output (default: tests/assets/testdata/)
        shape_filter:   List of shape config names to include (None = all)
        dtype_filter:   List of dtype names to include (None = all)
        combo_filter:   List of (hasClamp, isWeight, isGroupIndex) tuples (None = all)
        seed_base:      Starting seed for reproducibility

    Returns:
        List of case_info dicts
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "testdata"
        )

    all_cases = []
    case_counter = 0

    for shape_name, x_shape, _ in SHAPE_CONFIGS:
        if shape_filter and shape_name not in shape_filter:
            continue

        for dtype_name in DTYPE_CONFIGS:
            if dtype_filter and dtype_name not in dtype_filter:
                continue
            if dtype_name == "bf16" and not HAS_BF16:
                print("[WARN] Skipping BF16 case (ml_dtypes not available)")
                continue

            for has_clamp, is_weight, is_group_index in FEATURE_COMBOS:
                if (
                    combo_filter
                    and (has_clamp, is_weight, is_group_index) not in combo_filter
                ):
                    continue

                # Choose clamp_limit value
                if has_clamp:
                    # Use different clamp values for different shapes to add variety
                    clamp_limit = CLAMP_LIMIT_VALUES[
                        case_counter % len(CLAMP_LIMIT_VALUES)
                    ]
                else:
                    clamp_limit = -1.0

                # Build case name
                case_name = (
                    f"{shape_name}_{dtype_name}"
                    f"_hc{has_clamp}_iw{is_weight}_gi{is_group_index}"
                    f"_cl{clamp_limit:.1f}"
                )

                seed = seed_base + case_counter * 10000

                try:
                    case_info = generate_case(
                        case_name=case_name,
                        x_shape=x_shape,
                        dtype_name=dtype_name,
                        has_clamp=has_clamp,
                        is_weight=is_weight,
                        is_group_index=is_group_index,
                        clamp_limit=clamp_limit,
                        seed=seed,
                        output_dir=os.path.join(output_dir, case_name),
                    )
                    all_cases.append(case_info)
                    print(f"[OK] Generated: {case_name}")
                except Exception as e:
                    print(f"[ERR] Failed: {case_name} — {e}")

                case_counter += 1

    # Save summary
    summary_path = os.path.join(output_dir, "all_cases_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_cases": len(all_cases),
                "cases": all_cases,
                "generation_config": {
                    "shape_configs": [(n, list(s)) for n, s, _ in SHAPE_CONFIGS],
                    "dtype_configs": DTYPE_CONFIGS,
                    "feature_combos": FEATURE_COMBOS,
                    "clamp_limit_values": CLAMP_LIMIT_VALUES,
                },
            },
            f,
            indent=2,
        )

    print(f"\nTotal cases generated: {len(all_cases)}")
    print(f"Summary saved to: {summary_path}")
    return all_cases


# ============================================================================
# Special edge/boundary test cases
# ============================================================================


def generate_edge_cases(output_dir=None, seed_base=99999):
    """
    Generate special edge and boundary test cases.

    These include:
    - Empty tensor (T=0)
    - gradY all zeros
    - clampLimit = 0 (all gates clamped to ≤0)
    - Values at exact clamp boundary (g=c, u=±c)
    - Large gate values (g_tilde → +∞ regime)
    - All positive/negative gate values
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "testdata_edge"
        )

    edge_cases = []

    # 1. Empty tensor (T=0)
    case_name = "edge_empty_2d_fp32"
    case_info = generate_case(
        case_name=case_name,
        x_shape=(0, 32),
        dtype_name="fp32",
        has_clamp=0,
        is_weight=0,
        is_group_index=0,
        clamp_limit=-1.0,
        seed=seed_base,
        output_dir=os.path.join(output_dir, case_name),
    )
    edge_cases.append(case_info)
    print(f"[OK] Edge case: {case_name}")

    # 2. gradY all zeros → gradX all zeros, gradWeight all zeros
    case_name = "edge_zero_grad_fp32"
    x_shape = (8, 64)
    H = 32
    grad_y = np.zeros((8, H), dtype=np.float32)
    x_data = make_random_array(
        x_shape, "fp32", seed=seed_base + 1, value_range=(-3.0, 3.0)
    )
    weight = (
        np.random.default_rng(seed_base + 2)
        .uniform(0.5, 2.0, size=(8, 1))
        .astype(np.float32)
    )
    y_origin = generate_forward_output(x_data, weight=weight, clamp_limit=3.0)

    out_dir = os.path.join(output_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "input_grad_y_fp32.npy"), grad_y)
    np.save(os.path.join(out_dir, "input_x_fp32.npy"), x_data)
    np.save(os.path.join(out_dir, "input_weight.npy"), weight)
    np.save(
        os.path.join(out_dir, "input_y_origin_fp32.npy"),
        cast_to_target(y_origin, "fp32"),
    )
    np.save(
        os.path.join(out_dir, "attr_clamp_limit.npy"), np.array([3.0], dtype=np.float32)
    )

    grad_x_g, grad_w_g = compute_golden(
        grad_y, x_data, weight=weight, y_origin=y_origin, clamp_limit=3.0
    )
    np.save(os.path.join(out_dir, "golden_grad_x_fp32.npy"), grad_x_g)
    np.save(os.path.join(out_dir, "golden_grad_weight.npy"), grad_w_g)

    edge_cases.append(
        {
            "case_name": case_name,
            "dtype": "fp32",
            "x_shape": list(x_shape),
            "has_clamp": 1,
            "is_weight": 1,
            "is_group_index": 0,
            "clamp_limit": 3.0,
            "description": "gradY all zeros",
        }
    )
    print(f"[OK] Edge case: {case_name}")

    # 3. clampLimit = 0 (but hasClamp=0 → no clamp applied, NOT the "all gates ≤0" case)
    #    clampLimit=0 means no clamp (m_g≡1, m_u≡1)
    #    The "all gate values clamped to ≤0" extreme scenario is when clamp_limit
    #    is very small (e.g., 0.001) and all g values exceed it
    case_name = "edge_tiny_clamp_fp32"
    x_shape = (8, 64)
    # Generate x with all gate values > tiny clamp_limit
    x_fp32 = (
        np.random.default_rng(seed_base + 3)
        .uniform(1.0, 5.0, size=x_shape)
        .astype(np.float32)
    )
    x_data = x_fp32.astype(np.float32)
    out_dir = os.path.join(output_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)

    tiny_clamp = 0.001  # Very small clamp → most gates fully clamped
    grad_y = make_random_array(
        (8, 32), "fp32", seed=seed_base + 4, value_range=(-1.0, 1.0)
    )

    np.save(os.path.join(out_dir, "input_grad_y_fp32.npy"), grad_y)
    np.save(os.path.join(out_dir, "input_x_fp32.npy"), x_data)
    np.save(
        os.path.join(out_dir, "attr_clamp_limit.npy"),
        np.array([tiny_clamp], dtype=np.float32),
    )

    grad_x_g, _ = compute_golden(grad_y, x_data, clamp_limit=tiny_clamp)
    np.save(os.path.join(out_dir, "golden_grad_x_fp32.npy"), grad_x_g)

    edge_cases.append(
        {
            "case_name": case_name,
            "dtype": "fp32",
            "x_shape": list(x_shape),
            "has_clamp": 1,
            "is_weight": 0,
            "is_group_index": 0,
            "clamp_limit": tiny_clamp,
            "description": "Very small clamp_limit, most values clamped",
        }
    )
    print(f"[OK] Edge case: {case_name}")

    # 4. Large gate values (g_tilde → +∞ regime) — tests numerical stability of silu'
    case_name = "edge_large_gate_fp32"
    x_shape = (4, 64)
    # Gate values very large, up values moderate
    x_fp32 = (
        np.random.default_rng(seed_base + 5)
        .uniform(50.0, 100.0, size=x_shape)
        .astype(np.float32)
    )
    # Set up values to moderate range
    x_fp32[..., 32:] = np.random.default_rng(seed_base + 6).uniform(
        -3.0, 3.0, size=(4, 32)
    )
    x_data = x_fp32
    out_dir = os.path.join(output_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)

    grad_y = make_random_array(
        (4, 32), "fp32", seed=seed_base + 7, value_range=(-1.0, 1.0)
    )

    np.save(os.path.join(out_dir, "input_grad_y_fp32.npy"), grad_y)
    np.save(os.path.join(out_dir, "input_x_fp32.npy"), x_data)
    np.save(
        os.path.join(out_dir, "attr_clamp_limit.npy"),
        np.array([-1.0], dtype=np.float32),
    )

    grad_x_g, _ = compute_golden(grad_y, x_data, clamp_limit=-1.0)
    np.save(os.path.join(out_dir, "golden_grad_x_fp32.npy"), grad_x_g)

    edge_cases.append(
        {
            "case_name": case_name,
            "dtype": "fp32",
            "x_shape": list(x_shape),
            "has_clamp": 0,
            "is_weight": 0,
            "is_group_index": 0,
            "clamp_limit": 0.0,
            "description": "Large gate values (silu numerical stability)",
        }
    )
    print(f"[OK] Edge case: {case_name}")

    # 5. Exact clamp boundary values (g=c, u=c, u=-c) — tests open-interval mask
    case_name = "edge_exact_boundary_fp32"
    x_shape = (8, 64)
    H = 32
    c = 5.0
    x_fp32 = np.full(x_shape, c, dtype=np.float32)  # All gate values = c
    x_fp32[..., H:] = np.full((8, H), c, dtype=np.float32)  # All up values = c
    # Mix in some non-boundary values
    rng = np.random.default_rng(seed_base + 8)
    x_fp32[:4, :H] = rng.uniform(-3.0, 4.9, size=(4, H))  # Below c
    x_fp32[:4, H:] = rng.uniform(-4.9, 4.9, size=(4, H))  # Within bounds

    x_data = x_fp32
    grad_y = make_random_array(
        (8, 32), "fp32", seed=seed_base + 9, value_range=(-1.0, 1.0)
    )

    out_dir = os.path.join(output_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "input_grad_y_fp32.npy"), grad_y)
    np.save(os.path.join(out_dir, "input_x_fp32.npy"), x_data)
    np.save(
        os.path.join(out_dir, "attr_clamp_limit.npy"), np.array([c], dtype=np.float32)
    )

    grad_x_g, _ = compute_golden(grad_y, x_data, clamp_limit=c)
    np.save(os.path.join(out_dir, "golden_grad_x_fp32.npy"), grad_x_g)

    edge_cases.append(
        {
            "case_name": case_name,
            "dtype": "fp32",
            "x_shape": list(x_shape),
            "has_clamp": 1,
            "is_weight": 0,
            "is_group_index": 0,
            "clamp_limit": c,
            "description": "Exact clamp boundary values (open-interval mask test)",
        }
    )
    print(f"[OK] Edge case: {case_name}")

    # 6. All-negative gate values (silu(g) near 0 for negative g)
    case_name = "edge_negative_gate_fp32"
    x_shape = (4, 64)
    x_fp32 = (
        np.random.default_rng(seed_base + 10)
        .uniform(-10.0, -0.1, size=(4, 32))
        .astype(np.float32)
    )
    x_fp32_up = (
        np.random.default_rng(seed_base + 11)
        .uniform(-3.0, 3.0, size=(4, 32))
        .astype(np.float32)
    )
    x_data = np.concatenate([x_fp32, x_fp32_up], axis=-1)
    grad_y = make_random_array(
        (4, 32), "fp32", seed=seed_base + 12, value_range=(-1.0, 1.0)
    )

    out_dir = os.path.join(output_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "input_grad_y_fp32.npy"), grad_y)
    np.save(os.path.join(out_dir, "input_x_fp32.npy"), x_data)
    np.save(
        os.path.join(out_dir, "attr_clamp_limit.npy"),
        np.array([-1.0], dtype=np.float32),
    )

    grad_x_g, _ = compute_golden(grad_y, x_data, clamp_limit=-1.0)
    np.save(os.path.join(out_dir, "golden_grad_x_fp32.npy"), grad_x_g)

    edge_cases.append(
        {
            "case_name": case_name,
            "dtype": "fp32",
            "x_shape": list(x_shape),
            "has_clamp": 0,
            "is_weight": 0,
            "is_group_index": 0,
            "clamp_limit": 0.0,
            "description": "All-negative gate values",
        }
    )
    print(f"[OK] Edge case: {case_name}")

    # Save edge cases summary
    summary_path = os.path.join(output_dir, "edge_cases_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"total_edge_cases": len(edge_cases), "cases": edge_cases}, f, indent=2
        )

    print(f"\nEdge cases generated: {len(edge_cases)}")
    print(f"Summary saved to: {summary_path}")
    return edge_cases


# ============================================================================
# CLI interface
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate input data and golden output for SwigluGroupGrad operator tests."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for output files (default: tests/assets/testdata/)",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="*",
        default=None,
        help="Filter shape configs by name (e.g., tiny_2d small_2d). Default: all",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="*",
        default=None,
        help="Filter dtypes (e.g., bf16 fp32). Default: all",
    )
    parser.add_argument(
        "--combos",
        type=str,
        nargs="*",
        default=None,
        help='Filter feature combos as "hc,iw,gi" (e.g., "0,0,0" "1,1,1"). Default: all',
    )
    parser.add_argument(
        "--edge", action="store_true", help="Also generate edge/boundary test cases"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate a minimal set of cases for quick testing",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse combo filter
    combo_filter = None
    if args.combos:
        combo_filter = []
        for c in args.combos:
            parts = c.split(",")
            combo_filter.append((int(parts[0]), int(parts[1]), int(parts[2])))

    # Quick mode: only generate a few representative cases
    if args.quick:
        shape_filter = ["tiny_2d", "small_2d"]
        dtype_filter = ["fp32"]
        combo_filter = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
    else:
        shape_filter = args.shapes
        dtype_filter = args.dtypes
        combo_filter = combo_filter

    print("=" * 60)
    print("SwigluGroupGrad Test Data Generation")
    print("=" * 60)

    if not HAS_BF16:
        print("[WARN] ml_dtypes not available — BF16 cases will be skipped")
    print()

    # Generate standard cases
    print("Generating standard test cases...")
    generate_all_cases(
        output_dir=args.output_dir,
        shape_filter=shape_filter,
        dtype_filter=dtype_filter,
        combo_filter=combo_filter,
        seed_base=args.seed,
    )

    # Generate edge cases if requested
    if args.edge:
        print("\nGenerating edge/boundary test cases...")
        edge_dir = os.path.join(
            args.output_dir
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata"),
            "_edge_cases",
        )
        generate_edge_cases(output_dir=edge_dir, seed_base=args.seed + 99999)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
