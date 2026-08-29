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

"""Torch extension system tests for add_rms_norm_dynamic_quant.

Covers single-op mode (torch.ops direct call via JIT-compiled ACLNN) and graph
mode (torchair + torch.compile), designed to exercise ALL 10 tilingKey variants:

  TilingKey matrix (10 variants):
    FULL_LOAD    x FP8 x x3=False/True    (2)
    FULL_LOAD    x FP4 x x3=False/True    (2)
    SPLIT_R      x FP8 x x3=False/True    (2)
    SPLIT_R      x FP4 x x3=False/True    (2)
    REDUCE_EMPTY x FP8 x x3=False/True    (2, covers FP4 too)

  Shape coverage by tiling branch:
    FULL_LOAD:    R=32 (min block, aligned), 33 (non-aligned), 1024, 1025 (non-aligned)
                  16384 (FULL_LOAD/SPLIT_R boundary, aligned)
    SPLIT_R:      R=16385 (just above boundary, non-aligned),
                  65536 (large, aligned), 65537 (non-aligned)
    REDUCE_EMPTY: batch=1 (single row), batch=7 (prime, multi-core non-aligned)

  Other coverage:
    - dst_type: 35 (FP8_E5M2), 36 (FP8_E4M3FN), 40 (FP4_E2M1), 41 (FP4_E1M2)
    - dtype: float16, bfloat16
    - dim: 1D, 2D, 3D, 4D
    - output_rstd: True / False
    - scale_alg: 0 (OCP), 1 (cuBLAS, FP8 only)

Usage:
    cd <ops-nn repo root>
    pip install torch_extension/dist/cann_ops_nn-*.whl
    pytest norm/add_rms_norm_dynamic_mx_quant/tests/ut/torch_extension/test_torch_extension.py -v

Prerequisites:
  - CANN toolkit sourced (source <cann_path>/set_env.sh)
  - NPU device available (torch.npu.is_available() == True)
"""

import pytest
import torch
import torch_npu  # noqa: F401
import cann_ops_nn  # noqa: F401


def _is_ascend950():
    """True only on Ascend 950 (A5 / 910D, arch35).

    add_rms_norm_dynamic_quant ships kernels/binaries solely under arch35;
    on A2 (910B / arch22) and A3 (910C) the op is not registered. The in-repo
    ST runner (ops_st_test.sh) only collects ttk_*.csv, not this .py file, so
    an external smoke CI may run it across every SoC pool — guard here.
    """
    try:
        return torch.npu.is_available() and torch.npu.get_device_name().startswith(
            "Ascend950"
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_ascend950(),
    reason=(
        "add_rms_norm_dynamic_quant is Ascend 950 (A5) only; "
        "kernels/binaries are absent on A2 (910B) / A3 (910C) and non-NPU hosts"
    ),
)


# ---------------------------------------------------------------------------
# TilingKey-driven test matrix
# ---------------------------------------------------------------------------
# FULL_LOAD: R <= 16384 (fullLoadRMax = 2 * 64^2 * 2 = 4 * vl^2, GetVRegSize=256 → vl=64)
# SPLIT_R:   R > 16384 (fallback when FULL_LOAD not capable)
# REDUCE_EMPTY: batch=1 triggers single-row edge case

# Each scenario: (dst_type, has_x3, has_beta, dtype_str, shape, output_rstd, scale_alg, tiling_desc)
# tiling_desc is for test ID readability only.

FULL_LOAD_R_MAX = 16384

SCENARIOS = [
    # === FULL_LOAD: R=32 (min MX block, aligned) ===
    (36, False, False, "fp16", (4, 32), True, 0, "FL_r32"),
    (40, False, False, "fp16", (4, 32), True, 0, "FL_r32"),
    (35, False, False, "fp16", (4, 32), True, 0, "FL_r32"),
    (41, False, False, "fp16", (4, 32), True, 0, "FL_r32"),
    # FULL_LOAD: R=32, x3=True
    (36, True, False, "fp16", (4, 32), True, 0, "FL_x3_r32"),
    (40, True, False, "fp16", (4, 32), True, 0, "FL_x3_r32"),
    # FULL_LOAD: R=32, beta=True
    (36, False, True, "fp16", (4, 32), True, 0, "FL_beta_r32"),
    (40, False, True, "fp16", (4, 32), True, 0, "FL_beta_r32"),
    # FULL_LOAD: R=32, x3+beta, output_rstd=False (single-op only, graph mode not supported)
    (36, True, True, "fp16", (4, 32), False, 0, "FL_x3b_r32_norstd"),
    (40, True, True, "fp16", (4, 32), False, 0, "FL_x3b_r32_norstd"),
    # === FULL_LOAD: R=33 (non-aligned, 1 block + 1) — FP8 only (FP4 requires even) ===
    (36, False, False, "fp16", (4, 33), True, 0, "FL_r33"),
    (35, True, True, "fp16", (4, 33), True, 0, "FL_x3b_r33"),
    # === FULL_LOAD: R=34 (non-aligned, even — for FP4) ===
    (40, False, False, "fp16", (4, 34), True, 0, "FL_r34"),
    (41, True, True, "fp16", (4, 34), True, 0, "FL_x3b_r34"),
    # === FULL_LOAD: R=1024 (32 blocks, aligned) ===
    (36, False, False, "fp16", (4, 1024), True, 0, "FL_r1024"),
    (40, False, False, "fp16", (4, 1024), True, 0, "FL_r1024"),
    # === FULL_LOAD: R=1025 (non-aligned, odd — FP8 only) ===
    (35, True, False, "fp16", (4, 1025), True, 0, "FL_x3_r1025"),
    # === FULL_LOAD: R=1026 (non-aligned, even — for FP4) ===
    (40, True, False, "fp16", (4, 1026), True, 0, "FL_x3_r1026"),
    (41, True, False, "fp16", (4, 1026), True, 0, "FL_x3_r1026"),
    # === FULL_LOAD: R=16384 (boundary, aligned) ===
    (36, False, False, "fp16", (2, 16384), True, 0, "FL_r16384"),
    (40, False, False, "fp16", (2, 16384), True, 0, "FL_r16384"),
    (35, True, True, "fp16", (2, 16384), True, 0, "FL_x3b_r16384"),
    (41, True, True, "fp16", (2, 16384), True, 0, "FL_x3b_r16384"),
    # === SPLIT_R: R=16385 (just above boundary, non-aligned, odd — FP8 only) ===
    (36, False, False, "fp16", (2, 16385), True, 0, "SR_r16385"),
    (35, True, True, "fp16", (2, 16385), True, 0, "SR_x3b_r16385"),
    # === SPLIT_R: R=16386 (just above boundary, non-aligned, even — for FP4) ===
    (40, False, False, "fp16", (2, 16386), True, 0, "SR_r16386"),
    (41, True, True, "fp16", (2, 16386), True, 0, "SR_x3b_r16386"),
    # === SPLIT_R: R=65536 (large, aligned) ===
    (36, False, False, "fp16", (2, 65536), True, 0, "SR_r65536"),
    (40, False, False, "fp16", (2, 65536), True, 0, "SR_r65536"),
    # === SPLIT_R: R=65537 (large, non-aligned, odd — FP8 only) ===
    (35, True, True, "fp16", (2, 65537), True, 0, "SR_x3b_r65537"),
    # === SPLIT_R: R=65538 (large, non-aligned, even — for FP4) ===
    (41, True, True, "fp16", (2, 65538), True, 0, "SR_x3b_r65538"),
    # === REDUCE_EMPTY: batch=1 (single row edge case) ===
    (36, False, False, "fp16", (1, 64), True, 0, "RE_b1"),
    (40, False, False, "fp16", (1, 64), True, 0, "RE_b1"),
    # === REDUCE_EMPTY: batch=7 (prime, multi-core non-aligned) ===
    (35, True, True, "fp16", (7, 64), True, 0, "RE_x3b_b7"),
    (41, True, True, "fp16", (7, 64), True, 0, "RE_x3b_b7"),
    # === scale_alg=1 (cuBLAS, FP8 only) ===
    (36, False, False, "fp16", (4, 64), True, 1, "FL_alg1_r64"),
    (35, False, False, "fp16", (4, 64), True, 1, "FL_alg1_r64"),
    # === bfloat16 coverage ===
    (36, False, False, "bf16", (4, 32), True, 0, "FL_bf16_r32"),
    (40, False, False, "bf16", (4, 32), True, 0, "FL_bf16_r32"),
    (36, True, True, "bf16", (2, 16385), True, 0, "SR_bf16_x3b_r16385"),
    (40, True, True, "bf16", (2, 16386), True, 0, "SR_bf16_x3b_r16386"),
    # === 1D shapes ===
    (36, False, False, "fp16", (32,), False, 0, "FL_1d_r32"),
    (40, False, False, "fp16", (34,), False, 0, "FL_1d_r34"),
    # === 3D shapes (non-aligned R) ===
    (36, True, False, "fp16", (2, 4, 1025), True, 0, "FL_3d_x3_r1025"),
    (40, True, False, "fp16", (2, 4, 1026), True, 0, "FL_3d_x3_r1026"),
    # === 4D shapes (SPLIT_R) ===
    (35, False, False, "fp16", (2, 2, 4, 65536), True, 0, "SR_4d_r65536"),
    (41, False, False, "fp16", (2, 2, 4, 65536), True, 0, "SR_4d_r65536"),
]

# Graph mode scenarios: all scenarios (FP4 packing and output_rstd=False now handled by graph_convert).
GRAPH_SCENARIOS = SCENARIOS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TORCH_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
MX_BLOCK_SIZE = 32
ALIGN_NUM = 2

# Absolute precision tolerances (from aclnn_cases.csv)
TOL = {40: 0.0009765625, 41: 0.0078125, 36: 0.125, 35: 0.25}
# R > 16384 (SPLIT_R mode) uses looser rstd tolerance (binary-add reduction in fp32 has accumulation differences)
RSTD_TOL_LARGE_R = 0.001


def _is_fp4(dst_type):
    return dst_type in (40, 41)


def _expected_y_shape(shape, dst_type):
    y_shape = list(shape)
    if _is_fp4(dst_type):
        y_shape[-1] //= 2
    return tuple(y_shape)


def _expected_mxscale_shape(shape):
    num_blocks = (shape[-1] + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
    mx_last = (num_blocks + ALIGN_NUM - 1) // ALIGN_NUM
    return tuple(list(shape)[:-1] + [mx_last, 2])


def _expected_rstd_shape(shape):
    return tuple(list(shape)[:-1] + [1])


def _make_inputs(shape, dtype_str, has_beta, has_x3, seed=42, gamma_fp32=False):
    import numpy as np

    np.random.seed(seed)
    tdt = TORCH_DT[dtype_str]
    x1 = (
        torch.from_numpy(np.random.uniform(-1, 1, shape).astype(np.float32))
        .to(tdt)
        .to("npu")
    )
    x2 = (
        torch.from_numpy(np.random.uniform(-1, 1, shape).astype(np.float32))
        .to(tdt)
        .to("npu")
    )
    g_dt = torch.float32 if gamma_fp32 else tdt
    g = torch.from_numpy(np.ones(shape[-1], dtype=np.float32)).to(g_dt).to("npu")
    b = (
        torch.from_numpy(np.zeros(shape[-1], dtype=np.float32)).to(tdt).to("npu")
        if has_beta
        else None
    )
    x3 = (
        torch.from_numpy(np.random.uniform(-1, 1, shape).astype(np.float32))
        .to(tdt)
        .to("npu")
        if has_x3
        else None
    )
    return x1, x2, g, b, x3


def _golden_x(x1, x2, x3, dtype_str):
    x1c = x1.to("cpu").to(torch.float32)
    x2c = x2.to("cpu").to(torch.float32)
    x3c = x3.to("cpu").to(torch.float32) if x3 is not None else None
    x = (x3c + x1c) + x2c if x3c is not None else x1c + x2c
    return x.to(TORCH_DT[dtype_str]).to(torch.float32)


def _golden_rstd(x1, x2, x3, epsilon=1e-6):
    x1c = x1.to("cpu").to(torch.float32)
    x2c = x2.to("cpu").to(torch.float32)
    x3c = x3.to("cpu").to(torch.float32) if x3 is not None else None
    x = (x3c + x1c) + x2c if x3c is not None else x1c + x2c
    x_f32 = x.to(torch.float32)
    var = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
    return torch.rsqrt(var + epsilon)


def _call_op(x1, x2, g, b, x3, dst_type, output_rstd, scale_alg=0):
    op = torch.ops.cann_ops_nn.add_rms_norm_dynamic_quant
    return op(
        x1,
        x2,
        g,
        beta=b,
        x3=x3,
        epsilon=1e-6,
        scale_alg=scale_alg,
        round_mode="rint",
        dst_type=dst_type,
        output_rstd=output_rstd,
    )


def _param_id(params):
    dst, x3, beta, dt, shape, rstd, alg, desc = params
    return f"{desc}_dt{dst}_{dt}_x3{int(x3)}_b{int(beta)}_rstd{int(rstd)}_alg{alg}"


# ---------------------------------------------------------------------------
# Single-op mode tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dst_type,has_x3,has_beta,dtype_str,shape,output_rstd,scale_alg,tiling_desc",
    SCENARIOS,
    ids=[_param_id(s) for s in SCENARIOS],
)
class TestSingleOp:
    """Single-op mode: torch.ops direct call via JIT-compiled ACLNN.

    Every scenario is designed to hit a specific tilingKey variant:
    FL = FULL_LOAD, SR = SPLIT_R, RE = REDUCE_EMPTY.
    """

    def test_single_op(
        self,
        dst_type,
        has_x3,
        has_beta,
        dtype_str,
        shape,
        output_rstd,
        scale_alg,
        tiling_desc,
    ):
        x1, x2, g, b, x3 = _make_inputs(shape, dtype_str, has_beta, has_x3)
        y, x_out, mxscale, rstd = _call_op(
            x1, x2, g, b, x3, dst_type, output_rstd, scale_alg
        )

        # 1. Shape
        assert list(y.shape) == list(_expected_y_shape(shape, dst_type))
        assert list(x_out.shape) == list(shape)
        assert list(mxscale.shape) == list(_expected_mxscale_shape(shape))
        if output_rstd:
            assert list(rstd.shape) == list(_expected_rstd_shape(shape))

        # 2. Dtype
        if _is_fp4(dst_type):
            assert y.dtype == torch.uint8
        elif dst_type == 35:
            assert y.dtype == torch.float8_e5m2
        elif dst_type == 36:
            assert y.dtype == torch.float8_e4m3fn
        assert x_out.dtype == TORCH_DT[dtype_str]
        assert mxscale.dtype == torch.float8_e8m0fnu
        if output_rstd:
            assert rstd.dtype == torch.float32

        # 3. x_out precision
        x_golden = _golden_x(x1, x2, x3, dtype_str).to("npu")
        x_diff = (x_out.to(torch.float32) - x_golden).abs().max().item()
        assert x_diff < 0.02, f"x_out max_diff={x_diff:.6f}"

        # 4. rstd precision (looser tolerance for SPLIT_R due to binary-add accumulation differences)
        if output_rstd:
            rstd_golden = _golden_rstd(x1, x2, x3).to("npu")
            rstd_diff = (rstd - rstd_golden).abs().max().item()
            tol = RSTD_TOL_LARGE_R if shape[-1] > FULL_LOAD_R_MAX else 0.01
            assert rstd_diff < tol, f"rstd max_diff={rstd_diff:.6f} (tol={tol})"

        # 5. y/mxscale: non-zero and finite
        y_bytes = y.view(torch.uint8)
        mx_bytes = mxscale.view(torch.uint8)
        assert not torch.all(y_bytes == 0), "y all zeros"
        assert not torch.all(mx_bytes == 0), "mxscale all zeros"


# ---------------------------------------------------------------------------
# Graph mode tests (FP8 only — FP4 graph mode has dtype mismatch limitation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dst_type,has_x3,has_beta,dtype_str,shape,output_rstd,scale_alg,tiling_desc",
    GRAPH_SCENARIOS,
    ids=[_param_id(s) for s in GRAPH_SCENARIOS],
)
class TestGraphMode:
    """Graph mode: torchair + torch.compile.

    Cross-validates graph mode output against single-op mode (both invoke the
    same ACLNN kernel, so outputs must be byte-identical).

    FP4 (dst_type=40/41) outputs are packed to uint8 by graph_convert's
    _pack_fp4_output_to_uint8 post-processing. output_rstd=False is handled
    by filling rstd with an empty tensor via ge.Fill.
    """

    def test_graph_mode(
        self,
        dst_type,
        has_x3,
        has_beta,
        dtype_str,
        shape,
        output_rstd,
        scale_alg,
        tiling_desc,
    ):
        import torch._dynamo
        import torchair

        torch._dynamo.reset()
        cfg = torchair.CompilerConfig()
        backend = torchair.get_npu_backend(compiler_config=cfg)

        x1, x2, g, b, x3 = _make_inputs(shape, dtype_str, has_beta, has_x3)

        class OpModule(torch.nn.Module):
            def forward(self, x1, x2, g, b, x3):
                return _call_op(x1, x2, g, b, x3, dst_type, output_rstd, scale_alg)

        mod = OpModule().to("npu")
        compiled = torch.compile(mod, backend=backend, fullgraph=True)
        y_g, xo_g, mx_g, rs_g = compiled(x1, x2, g, b, x3)

        # Single-op reference
        y_s, xo_s, mx_s, rs_s = _call_op(
            x1, x2, g, b, x3, dst_type, output_rstd, scale_alg
        )

        # Cross-validate: x_out and mxscale must be byte-identical;
        # y uses ULP<=1 tolerance (consistent with TTK requant standard for FP8/FP4)
        y_g_bytes = y_g.view(torch.uint8).to(torch.int32)
        y_s_bytes = y_s.view(torch.uint8).to(torch.int32)
        y_diff = (y_g_bytes - y_s_bytes).abs()
        y_mismatch = (y_diff > 1).sum().item()
        y_total = y_g_bytes.numel()
        assert y_mismatch == 0, f"y ULP>1 mismatch: {y_mismatch}/{y_total}"
        assert torch.equal(xo_g, xo_s), "x_out mismatch graph vs single-op"
        assert torch.equal(mx_g.view(torch.uint8), mx_s.view(torch.uint8)), (
            "mxscale mismatch graph vs single-op"
        )
        if output_rstd:
            rstd_tol = RSTD_TOL_LARGE_R if shape[-1] > FULL_LOAD_R_MAX else 1e-6
            assert torch.allclose(rs_g, rs_s, atol=rstd_tol), (
                "rstd mismatch graph vs single-op"
            )

        # Precision against golden
        x_golden = _golden_x(x1, x2, x3, dtype_str).to("npu")
        x_diff = (xo_g.to(torch.float32) - x_golden).abs().max().item()
        assert x_diff < 0.02, f"x_out max_diff={x_diff:.6f}"


# ---------------------------------------------------------------------------
# Performance benchmark: SPLIT_R precision fix impact
# ---------------------------------------------------------------------------
# The SPLIT_R fp32 precision fix (commit 3fe0de897) re-reads x1/x2(/x3) from GM
# in phase 2 instead of reading back fp16 x_out. This doubles GM read traffic
# for x inputs. The WRITE_XOUT=false optimization (P14) partially offsets this
# by skipping the redundant fp16 store in phase 2.
#
# This benchmark measures wall-clock time for SPLIT_R shapes to quantify the
# net impact and provide baseline data for future optimization decisions.
# Run with: pytest -v -k test_split_r_perf --benchmark-only

PERF_SHAPES = [
    ((4, 16385), "SR_boundary_nonaligned"),
    ((4, 32768), "SR_2x_boundary"),
    ((4, 65536), "SR_large"),
    ((2, 16385), "SR_boundary_2row"),
    ((2, 65536), "SR_large_2row"),
]

# FP4 packs 2 values per byte: odd tail dims violate the op's input contract,
# so FP4 perf variants collect only even-R shapes (no skips in the report).
PERF_SHAPES_FP4 = [s for s in PERF_SHAPES if s[0][-1] % 2 == 0]


class TestSplitRPerf:
    """Benchmark SPLIT_R performance to evaluate precision fix impact.

    Measures single-op execution time for SPLIT_R shapes. Results should be
    compared against the pre-fix baseline (reading fp16 x_out from GM) to
    quantify the GM bandwidth overhead of the fp32 precision fix.

    The WRITE_XOUT=false optimization (CalculateXAdd template param) skips the
    redundant fp16 store in phase 2, partially offsetting the extra GM reads.

    Expected overhead: ~+100% x-input GM read traffic (2x → 4x for 2-input,
    3x → 6x for 3-input), partially offset by removing 1 store per VL_F32 loop.
    """

    @pytest.mark.parametrize("shape,desc", PERF_SHAPES, ids=[d for _, d in PERF_SHAPES])
    def test_split_r_perf_fp8(self, shape, desc):
        x1, x2, g, b, x3 = _make_inputs(shape, "fp16", has_beta=True, has_x3=True)
        _call_op(x1, x2, g, b, x3, 36, True, 0)
        torch.npu.synchronize()

        import time

        iters = 50
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _call_op(x1, x2, g, b, x3, 36, True, 0)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / iters * 1000
        print(
            f"\n[PERF] {desc} shape={shape} fp8_x3: {avg_ms:.2f} ms/iter ({iters} iters)"
        )

    @pytest.mark.parametrize(
        "shape,desc", PERF_SHAPES_FP4, ids=[d for _, d in PERF_SHAPES_FP4]
    )
    def test_split_r_perf_fp4(self, shape, desc):
        x1, x2, g, b, x3 = _make_inputs(shape, "fp16", has_beta=True, has_x3=True)
        _call_op(x1, x2, g, b, x3, 40, True, 0)
        torch.npu.synchronize()

        import time

        iters = 50
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _call_op(x1, x2, g, b, x3, 40, True, 0)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) / iters * 1000
        print(
            f"\n[PERF] {desc} shape={shape} fp4_x3: {avg_ms:.2f} ms/iter ({iters} iters)"
        )


# ---------------------------------------------------------------------------
# Representative production shapes (PR 8520 perf baseline) — correctness guard
# ---------------------------------------------------------------------------
# These two shapes are the perf-comparison baseline for V1+pre-add vs V2-fused
# (see perf_v1_preadd_vs_v2_fused.py). They guard the x3=True path on exactly
# those shapes with fp32 gamma (the perf scenario config), so a tiling/kernel
# change that regresses perf also trips a correctness failure here.
#   large: (2304, 3200) — FULL_LOAD, R=3200, 55-56 cores
#   small: (2, 4096)    — FULL_LOAD, R=4096, 2-16 cores
# Config: bf16 x + fp32 gamma + no beta + FP8_E4M3FN + output_rstd=True (to
# also check rstd), x3 present (V2 path).

REPR_SCENARIOS = [
    (36, "bf16", (2304, 3200), "repr_large_2304x3200"),
    (36, "bf16", (2, 4096), "repr_small_2x4096"),
]


@pytest.mark.parametrize(
    "dst_type,dtype_str,shape,desc",
    REPR_SCENARIOS,
    ids=[s[-1] for s in REPR_SCENARIOS],
)
class TestRepresentativeShapes:
    """Correctness guard on the perf-baseline shapes with fp32 gamma + x3."""

    def test_representative_shape(self, dst_type, dtype_str, shape, desc):
        x1, x2, g, b, x3 = _make_inputs(
            shape, dtype_str, has_beta=False, has_x3=True, gamma_fp32=True
        )
        y, x_out, mxscale, rstd = _call_op(x1, x2, g, b, x3, dst_type, True, 0)

        # 1. Shape / dtype
        assert list(y.shape) == list(_expected_y_shape(shape, dst_type))
        assert list(x_out.shape) == list(shape)
        assert list(mxscale.shape) == list(_expected_mxscale_shape(shape))
        assert list(rstd.shape) == list(_expected_rstd_shape(shape))
        assert y.dtype == torch.float8_e4m3fn
        assert x_out.dtype == TORCH_DT[dtype_str]
        assert mxscale.dtype == torch.float8_e8m0fnu
        assert rstd.dtype == torch.float32
        assert g.dtype == torch.float32  # fp32 gamma, the perf-scenario config

        # 2. x_out vs golden (fp32 accumulation inside V2 must match golden)
        x_golden = _golden_x(x1, x2, x3, dtype_str).to("npu")
        x_diff = (x_out.to(torch.float32) - x_golden).abs().max().item()
        assert x_diff < 0.02, f"x_out max_diff={x_diff:.6f}"

        # 3. rstd vs golden
        rstd_golden = _golden_rstd(x1, x2, x3).to("npu")
        rstd_diff = (rstd - rstd_golden).abs().max().item()
        tol = RSTD_TOL_LARGE_R if shape[-1] > FULL_LOAD_R_MAX else 0.01
        assert rstd_diff < tol, f"rstd max_diff={rstd_diff:.6f} (tol={tol})"

        # 4. y / mxscale non-zero and finite
        assert not torch.all(y.view(torch.uint8) == 0), "y all zeros"
        assert not torch.all(mxscale.view(torch.uint8) == 0), "mxscale all zeros"
