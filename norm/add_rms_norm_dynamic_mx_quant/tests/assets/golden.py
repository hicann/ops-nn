#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Golden for add_rms_norm_dynamic_mx_quant (arch35).

Reuses the proven MX quantization golden (_mx_quantize) from dynamic_mx_quant.

Computation:
  x = x1 + x2                      (fp32; if x3 present: x = (x3 + x1) + x2)
  rstd = rsqrt(mean(x^2, -1) + eps)
  normed = x * rstd * gamma        (fp32; + beta if beta present)
  y, mxscale = MxQuant(normed)     (reuse _mx_quantize)

Output order (matches op def): y, x, mxscale, rstd.
When output_rstd is False, rstd is returned as None (TTK skips None outputs).
"""

import importlib.util
import os

import numpy as np
import torch

# Reuse the authoritative MX quant golden from dynamic_mx_quant.
# Load by file path to avoid a module-name collision (both files are golden.py).
_MX_QUANT_GOLDEN_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "quant",
        "dynamic_mx_quant",
        "tests",
        "assets",
        "golden.py",
    )
)
_spec = importlib.util.spec_from_file_location(
    "_dynamic_mx_quant_golden", _MX_QUANT_GOLDEN_PATH
)
_dynamic_mx_quant_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dynamic_mx_quant_golden)
_mx_quantize = _dynamic_mx_quant_golden._mx_quantize
DATA_TYPE_INT_TO_STR = _dynamic_mx_quant_golden.DATA_TYPE_INT_TO_STR

# Version lock: the monkey-patch below replaces _mx_calculate_share_exp on the
# upstream dynamic_mx_quant golden module. If upstream renames or removes it,
# the assignment would silently succeed (just adds a new attr) while _mx_quantize
# keeps calling the old implementation — producing wrong goldens with no error.
# This assert makes such drift fail loudly instead of silently.
_expected_upstream_symbols = (
    "_mx_quantize",
    "_mx_calculate_share_exp",
    "DATA_TYPE_INT_TO_STR",
)
for _sym in _expected_upstream_symbols:
    assert hasattr(_dynamic_mx_quant_golden, _sym), (
        f"Upstream dynamic_mx_quant golden no longer exports '{_sym}'. "
        f"The monkey-patch below is stale and must be updated. "
        f"Check quant/dynamic_mx_quant/tests/assets/golden.py for API changes."
    )

_EMAX_MAP = {
    "float8_e4m3fn": 8,
    "float8_e5m2": 15,
    "float4_e2m1": 2,
    "float4_e1m2": 0,
}


def _mx_calculate_share_exp_ocp(fp_array, scale_axis, mx_ele_dtype):
    emax = _EMAX_MAP[mx_ele_dtype]
    arr = np.abs(fp_array)

    if fp_array.dtype.name == "float16":
        fp16_bits = arr.view(np.uint16)
        fp16_exp = (fp16_bits >> 10) & 0x1F
        fp16_man = fp16_bits & 0x3FF
        is_inf_nan = fp16_exp == 0x1F
        is_zero = (fp16_exp == 0) & (fp16_man == 0)
        bf16_exp = fp16_exp.astype(np.int32) + 112
        bf16_exp[is_inf_nan] = 255
        bf16_exp[is_zero] = 0
    elif fp_array.dtype.name == "bfloat16":
        bf16_bits = arr.view(np.uint16)
        bf16_exp = ((bf16_bits >> 7) & 0xFF).astype(np.int32)
        is_inf_nan = bf16_exp == 255
        is_zero = (bf16_exp == 0) & ((bf16_bits & 0x7F) == 0)
        bf16_exp[is_inf_nan] = 255
        bf16_exp[is_zero] = 0
    else:
        fp32_bits = arr.view(np.uint32)
        bf16_exp = ((fp32_bits >> 23) & 0xFF).astype(np.int32)
        is_inf_nan = bf16_exp == 255
        is_zero = (bf16_exp == 0) & ((fp32_bits & 0x7FFFFF) == 0)
        bf16_exp[is_inf_nan] = 255
        bf16_exp[is_zero] = 0

    max_exp_np = np.max(bf16_exp, axis=scale_axis, keepdims=True).astype(np.float32)
    share_exp = max_exp_np - float(127 + emax)
    share_exp[max_exp_np == 0] = -float("inf")
    share_exp[max_exp_np == 255] = float("NaN")
    return share_exp


def _mx_quantize_ocp(*args, **kwargs):
    """Wrapper that scopes the OCP monkey-patch to a single call.

    Saves/restores the upstream _mx_calculate_share_exp around each invocation
    to prevent cross-test contamination if another test in the same process
    imports the upstream dynamic_mx_quant golden.
    """
    _original = _dynamic_mx_quant_golden._mx_calculate_share_exp
    _dynamic_mx_quant_golden._mx_calculate_share_exp = _mx_calculate_share_exp_ocp
    try:
        return _mx_quantize(*args, **kwargs)
    finally:
        _dynamic_mx_quant_golden._mx_calculate_share_exp = _original


__golden__ = {
    "kernel": {"add_rms_norm_dynamic_mx_quant": "add_rms_norm_dynamic_mx_quant_golden"},
    "aclnn": {
        "aclnnAddRmsNormDynamicMxQuant": "aclnn_add_rms_norm_dynamic_mx_quant_golden",
        "aclnnAddRmsNormDynamicMxQuantV2": "aclnn_add_rms_norm_dynamic_mx_quant_golden",
    },
}


def _to_fp32(arr):
    """numpy array (incl ml_dtypes.bfloat16) -> float32 numpy array."""
    return arr.astype(np.float32)


def _np_to_torch_fp32(arr):
    """numpy array (incl ml_dtypes.bfloat16) -> torch.float32 tensor."""
    return torch.from_numpy(_to_fp32(arr))


# Above this element count, golden switches to a chunked path that processes
# batch rows independently.  RMSNorm reduces only over the last dim, so every
# batch row is numerically independent — chunking along batch dims produces
# bit-identical results while bounding peak memory (the full path materialises
# ~8 full-size fp32 intermediates simultaneously; for 745M elements that is
# ~24 GiB, which exceeds the 32 GiB container cgroup limit).
_CHUNK_THRESHOLD = 200_000_000


def _golden_chunked(
    x1,
    x2,
    gamma,
    beta=None,
    x3=None,
    epsilon=1e-6,
    scale_alg=0,
    round_mode="rint",
    dst_type=40,
    output_rstd=False,
):
    """Chunked golden — identical math to the full path, processed per batch-slice.

    Flattens batch dims to (total_batch, norm_dim), processes rows in chunks of
    ~80M elements, pre-allocates full output arrays from the first chunk's
    output shapes, then fills them.  Peak memory ≈ inputs + outputs + one
    chunk of fp32 intermediates (~2 GiB) instead of ~8× full-size intermediates.
    """
    in_dtype = x1.dtype
    norm_dim = x1.shape[-1]
    batch_shape = x1.shape[:-1]
    total_batch = int(np.prod(batch_shape)) if batch_shape else 1

    # Target ~80M elements per chunk → ~2 GiB peak fp32 intermediates (8 tensors).
    chunk_rows = max(1, 80_000_000 // norm_dim)

    # Flatten to 2D for uniform slicing.  ascontiguousarray ensures the reshape
    # is a view (no copy) for C-contiguous arrays; for non-contiguous it copies
    # once, which is acceptable.
    x1_2d = np.ascontiguousarray(x1).reshape(total_batch, norm_dim)
    x2_2d = np.ascontiguousarray(x2).reshape(total_batch, norm_dim)
    x3_2d = (
        np.ascontiguousarray(x3).reshape(total_batch, norm_dim)
        if x3 is not None
        else None
    )

    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    gamma_t = _np_to_torch_fp32(gamma)  # 1D, small — full copy is negligible
    beta_t = _np_to_torch_fp32(beta) if beta is not None else None

    def _process_slice(start, end):
        """Compute golden for rows [start, end).  Returns (x_out, mxscale, y, rstd)."""
        x1_t = _np_to_torch_fp32(x1_2d[start:end])
        x2_t = _np_to_torch_fp32(x2_2d[start:end])
        if x3_2d is not None:
            x3_t = _np_to_torch_fp32(x3_2d[start:end])
            x = (x3_t + x1_t) + x2_t
            del x3_t
        else:
            x = x1_t + x2_t
        x_out_c = x.numpy().astype(in_dtype)

        normed = torch.nn.functional.rms_norm(
            x, [norm_dim], weight=gamma_t, eps=float(epsilon)
        )
        if beta_t is not None:
            normed = normed + beta_t
        normed_f = _to_fp32(normed.numpy().astype(in_dtype))

        inv_norm = 1.0 / float(norm_dim) if norm_dim != 0 else 0.0
        var = torch.sum(x * x, dim=-1, keepdim=True) * inv_norm
        rstd = torch.rsqrt(var + float(epsilon))

        scale_c, ele_c = _mx_quantize_ocp(
            normed_f,
            mx_ele_dtype=dst_type_str,
            axis=-1,
            block_size=32,
            round_mode=round_mode,
            scale_alg=scale_alg,
        )
        rstd_c = rstd.numpy().astype(np.float32) if output_rstd else None

        del x1_t, x2_t, x, normed, normed_f, var, rstd
        return x_out_c, scale_c, ele_c, rstd_c

    # First chunk: determine output element-shapes, then pre-allocate full arrays.
    first_end = min(chunk_rows, total_batch)
    x_out_0, mx_0, y_0, rstd_0 = _process_slice(0, first_end)

    y_full = np.empty((total_batch,) + y_0.shape[1:], dtype=y_0.dtype)
    x_full = np.empty((total_batch, norm_dim), dtype=in_dtype)
    mx_full = np.empty((total_batch,) + mx_0.shape[1:], dtype=mx_0.dtype)
    rstd_full = (
        np.empty(
            (total_batch,) + (rstd_0.shape[1:] if rstd_0 is not None else (1,)),
            dtype=np.float32,
        )
        if output_rstd
        else None
    )

    y_full[:first_end] = y_0
    x_full[:first_end] = x_out_0
    mx_full[:first_end] = mx_0
    if rstd_full is not None:
        rstd_full[:first_end] = rstd_0
    del x_out_0, mx_0, y_0, rstd_0

    for start in range(first_end, total_batch, chunk_rows):
        end = min(start + chunk_rows, total_batch)
        x_out_c, mx_c, y_c, rstd_c = _process_slice(start, end)
        y_full[start:end] = y_c
        x_full[start:end] = x_out_c
        mx_full[start:end] = mx_c
        if rstd_full is not None:
            rstd_full[start:end] = rstd_c
        del x_out_c, mx_c, y_c, rstd_c

    y_out = y_full.reshape(batch_shape + y_full.shape[1:])
    x_out = x_full.reshape(x1.shape)
    mx_out = mx_full.reshape(batch_shape + mx_full.shape[1:])
    rstd_out = (
        rstd_full.reshape(batch_shape + rstd_full.shape[1:])
        if rstd_full is not None
        else None
    )
    return y_out, x_out, mx_out, rstd_out


def add_rms_norm_dynamic_mx_quant_golden(
    x1,
    x2,
    gamma,
    beta=None,
    x3=None,
    epsilon=1e-6,
    scale_alg=0,
    round_mode="rint",
    dst_type=40,
    output_rstd=False,
    **kwargs,
):
    """Golden for add_rms_norm_dynamic_mx_quant.

    Args follow the op def order: inputs (x1, x2, gamma, beta, x3) then attrs
    (epsilon, scale_alg, round_mode, dst_type, output_rstd).
    All input tensors are numpy.ndarray; optional inputs beta/x3 are None when
    absent.  **kwargs carries TTK context (dtypes/shapes/formats, soc version,
    testcase_name, ...).

    Returns:
        (y, x, mxscale, rstd) matching the op def output order.  rstd is None
        when output_rstd is False (TTK skips None outputs).
    """
    in_dtype = x1.dtype
    norm_dim = x1.shape[-1]

    # Large shapes: use chunked path to avoid OOM (see _golden_chunked).
    if x1.size > _CHUNK_THRESHOLD:
        return _golden_chunked(
            x1,
            x2,
            gamma,
            beta=beta,
            x3=x3,
            epsilon=epsilon,
            scale_alg=scale_alg,
            round_mode=round_mode,
            dst_type=dst_type,
            output_rstd=output_rstd,
        )

    x1_t = _np_to_torch_fp32(x1)
    x2_t = _np_to_torch_fp32(x2)
    gamma_t = _np_to_torch_fp32(gamma)

    if x3 is not None:
        x = (_np_to_torch_fp32(x3) + x1_t) + x2_t
    else:
        x = x1_t + x2_t

    x_out = x.numpy().astype(in_dtype)

    normed = torch.nn.functional.rms_norm(
        x, [norm_dim], weight=gamma_t, eps=float(epsilon)
    )
    if beta is not None:
        normed = normed + _np_to_torch_fp32(beta)

    # Simulate kernel's fp32→T_X cast before MxQuant: CalculateY computes normed in
    # fp32 registers, then StoreTensorForDtypeTOut<T_X> casts to fp16/bf16 (T_X) for
    # MxQuant input. The golden must match this path — quantizing from fp16, not fp32.
    normed_f = _to_fp32(normed.numpy().astype(in_dtype))

    # norm_dim == 0 (empty reduction axis): avoid 1/0 ZeroDivisionError.
    # var=0, rstd=rsqrt(eps) is harmless — x is empty so normed/y/mxscale are
    # empty; kernel's REDUCE_EMPTY path also writes nothing for R=0.
    inv_norm = 1.0 / float(norm_dim) if norm_dim != 0 else 0.0
    var = torch.sum(x * x, dim=-1, keepdim=True) * inv_norm
    rstd = torch.rsqrt(var + float(epsilon))

    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    scale_array, ele_array = _mx_quantize_ocp(
        normed_f,
        mx_ele_dtype=dst_type_str,
        axis=-1,
        block_size=32,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )

    rstd_out = rstd.numpy().astype(np.float32) if output_rstd else None

    return ele_array, x_out, scale_array, rstd_out


def aclnn_add_rms_norm_dynamic_mx_quant_golden(*args, **kwargs):
    """ACLNN golden wrapper.

    TTK aclnn mode passes tensors and scalars positionally via build_args,
    matching the aclnnGetWorkspaceSize signature (output tensors skipped).

    V1 (aclnnAddRmsNormDynamicMxQuant) param order:
      x1, x2, gamma, beta, epsilon, scaleAlg, roundMode, dstType, outputRstd
    V2 (aclnnAddRmsNormDynamicMxQuantV2) param order:
      x1, x2, gamma, beta, x3, epsilon, scaleAlg, roundMode, dstType, outputRstd

    V1 has 4 tensors + 5 scalars = 9 positional args (no x3).
    V2 has 5 tensors + 5 scalars = 10 positional args (with x3, which may be None).
    Distinguish by arg count: len(args) >= 10 means V2 (x3 slot always present,
    even when x3 is None).  The old check `hasattr(args[4], "shape")` broke when
    V2 was called with x3=None — args[4] was None (no .shape), so the golden
    misdetected V2 as V1 and read epsilon from the wrong position.
    """

    def _to_np(t):
        if t is None:
            return None
        if hasattr(t, "numpy"):
            # torch.bfloat16 has no direct numpy conversion; view as int16 → ml_dtypes.bfloat16
            if hasattr(t, "dtype") and str(t.dtype) == "torch.bfloat16":
                from ml_dtypes import bfloat16 as _bf16

                return t.view(torch.int16).cpu().numpy().view(_bf16)
            return t.numpy()
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy()
        return t

    testcase_name = kwargs.get("testcase_name", "")
    has_x3_in_name = "x31" in testcase_name

    # When called from TTK ACLNN, output tensors (y, x, mxscale, rstd) are appended
    # positionally. When called directly (self-test), no output tensors are present.
    # V1 direct: 9 args (4 input + 5 scalar)
    # V1 TTK: 13 args (4 input + 5 scalar + 4 output)
    # V2-x3 direct: 10 args (5 input + 5 scalar)
    # V2-x3 TTK: 14 args (5 input + 5 scalar + 4 output)
    num_output_tensors = 4 if len(args) > 10 else 0
    input_and_scalar_args = (
        args[: len(args) - num_output_tensors] if num_output_tensors else args
    )
    is_v2 = has_x3_in_name or len(input_and_scalar_args) >= 10

    x1 = _to_np(input_and_scalar_args[0]) if len(input_and_scalar_args) > 0 else None
    x2 = _to_np(input_and_scalar_args[1]) if len(input_and_scalar_args) > 1 else None
    gamma = _to_np(input_and_scalar_args[2]) if len(input_and_scalar_args) > 2 else None
    beta = _to_np(input_and_scalar_args[3]) if len(input_and_scalar_args) > 3 else None

    if is_v2:
        if len(input_and_scalar_args) > 4:
            x3 = _to_np(input_and_scalar_args[4])
        else:
            x3 = None
        offset = 5
    else:
        x3 = None
        offset = 4

    epsilon = (
        input_and_scalar_args[offset]
        if len(input_and_scalar_args) > offset
        else kwargs.get("epsilon", 1e-6)
    )
    scale_alg = (
        input_and_scalar_args[offset + 1]
        if len(input_and_scalar_args) > offset + 1
        else kwargs.get("scaleAlg", kwargs.get("scale_alg", 0))
    )
    round_mode = (
        input_and_scalar_args[offset + 2]
        if len(input_and_scalar_args) > offset + 2
        else kwargs.get("roundMode", kwargs.get("round_mode", "rint"))
    )
    dst_type = (
        input_and_scalar_args[offset + 3]
        if len(input_and_scalar_args) > offset + 3
        else kwargs.get("dstType", kwargs.get("dst_type", 40))
    )
    output_rstd = (
        input_and_scalar_args[offset + 4]
        if len(input_and_scalar_args) > offset + 4
        else kwargs.get("outputRstd", kwargs.get("output_rstd", False))
    )

    return add_rms_norm_dynamic_mx_quant_golden(
        x1, x2, gamma, beta, x3, epsilon, scale_alg, round_mode, dst_type, output_rstd
    )


# === E2E spec for torch_extension (cann_ops_nn) ===========================
# TTK e2e mode loads this via __spec__; the Spec class delegates to the
# golden function above. Signature resolution comes from the op's own
# schema (torch.ops OpOverloadPacket._schemas) — no manual register().

__spec__ = {
    "torch.ops.cann_ops_nn.add_rms_norm_dynamic_quant": "AddRmsNormDynamicMxQuantE2ESpec",
}


class AddRmsNormDynamicMxQuantE2ESpec:
    """E2E spec: torch<->numpy adapter + delegates to golden function."""

    # torch_extension returns FP4 y packed as uint8 (2 values/byte, tail dim
    # halved) since torch has no native FP4 dtype; the golden returns unpacked
    # float4. pre_compare unpacks the NPU side so both carry the float4 dtype
    # and TTK resolves the requant standard (binary_equal would reject the
    # cross-dtype pair).
    @staticmethod
    def _unpack_packed_fp4(out, golden):
        from ttk.utilities import unpack_4bits

        if out is None or golden is None:
            return out, golden
        if not hasattr(out, "dtype") or str(out.dtype) != "uint8":
            return out, golden
        if "float4" not in str(getattr(golden, "dtype", "")):
            return out, golden
        dst = str(golden.dtype).split(".")[-1]
        unpacked = unpack_4bits(np.ascontiguousarray(out), dst)
        return unpacked.reshape(golden.shape), golden

    @classmethod
    def pre_compare(cls, y, x, mxscale, rstd, g_y, g_x, g_mxscale, g_rstd):
        y, g_y = cls._unpack_packed_fp4(y, g_y)
        return [y, x, mxscale, rstd, g_y, g_x, g_mxscale, g_rstd]

    @staticmethod
    def golden(
        x1,
        x2,
        gamma,
        beta=None,
        x3=None,
        epsilon=1e-6,
        scale_alg=0,
        round_mode="rint",
        dst_type=40,
        output_rstd=False,
        **kwargs,
    ):
        def _to_np(t):
            if t is None:
                return None
            if hasattr(t, "detach"):
                t = t.detach().cpu()
            if hasattr(t, "numpy"):
                try:
                    return t.numpy()
                except (TypeError, RuntimeError):
                    if t.dtype == torch.bfloat16:
                        # keep bf16: golden branches on input dtype to simulate
                        # the kernel's fp32->T_X cast before quantization;
                        # promoting to fp32 here would skip that cast
                        from ml_dtypes import bfloat16 as _np_bf16

                        return t.view(torch.uint16).numpy().view(_np_bf16)
                    return (
                        t.view(torch.uint8).numpy()
                        if t.element_size() == 1
                        else t.float().numpy()
                    )
            return np.asarray(t)

        dtypes = kwargs.get("dtypes", ("float16",) * 5)
        shapes = kwargs.get("shapes", None)
        formats = kwargs.get("formats", ("ND",) * 5)
        testcase_name = kwargs.get("testcase_name", "e2e")

        y_np, x_np, mx_np, rstd_np = add_rms_norm_dynamic_mx_quant_golden(
            _to_np(x1),
            _to_np(x2),
            _to_np(gamma),
            beta=_to_np(beta),
            x3=_to_np(x3),
            epsilon=epsilon,
            scale_alg=scale_alg,
            round_mode=round_mode,
            dst_type=dst_type,
            output_rstd=output_rstd,
            dtypes=dtypes,
            shapes=shapes,
            formats=formats,
            soc_version="ascend950",
            testcase_name=testcase_name,
        )

        # Return numpy directly; TTK result_to_numpy handles torch<->numpy
        # conversion (incl. FP4/Fp8). Returning torch here would double-convert.
        return [y_np, x_np, mx_np, rstd_np if rstd_np is not None else None]

    tolerance = {
        "float16": {
            "standard": "stat_rel_err",
            "mere": 0.0,
            "mare": 0.0,
            "threshold": 0.001,
        },
        "bfloat16": {
            "standard": "stat_rel_err",
            "mere": 0.0,
            "mare": 0.0,
            "threshold": 0.001,
        },
        "float32": {
            "standard": "stat_rel_err",
            "mere": 0.0,
            "mare": 0.0,
            "threshold": 0.0001,
        },
    }


if __name__ == "__main__":
    # Self-test: small example, verify shapes/dtypes and x3 effect.
    np.random.seed(42)
    N, D = 4, 64
    x1 = np.random.uniform(-1, 1, (N, D)).astype(np.float16)
    x2 = np.random.uniform(-1, 1, (N, D)).astype(np.float16)
    x3 = np.random.uniform(-1, 1, (N, D)).astype(np.float16)
    gamma = np.random.uniform(-1, 1, (D,)).astype(np.float16)

    # Case 1: x3 absent, FP8 E4M3, OCP, output_rstd=True
    y1, x1_out, mx1, rstd1 = add_rms_norm_dynamic_mx_quant_golden(
        x1,
        x2,
        gamma,
        beta=None,
        x3=None,
        dst_type=36,
        scale_alg=0,
        round_mode="rint",
        output_rstd=True,
    )
    print(
        f"[x3_absent] y={y1.shape}/{y1.dtype}, x={x1_out.shape}/{x1_out.dtype}, "
        f"mxscale={mx1.shape}/{mx1.dtype}, rstd={rstd1.shape}/{rstd1.dtype}"
    )

    # Case 2: x3 present, FP8 E4M3, OCP, output_rstd=True
    y2, x2_out, mx2, rstd2 = add_rms_norm_dynamic_mx_quant_golden(
        x1,
        x2,
        gamma,
        beta=None,
        x3=x3,
        dst_type=36,
        scale_alg=0,
        round_mode="rint",
        output_rstd=True,
    )
    print(
        f"[x3_present] y={y2.shape}/{y2.dtype}, x={x2_out.shape}/{x2_out.dtype}, "
        f"mxscale={mx2.shape}/{mx2.dtype}, rstd={rstd2.shape}/{rstd2.dtype}"
    )

    # Verify x3 actually affects output (x_out must differ)
    assert not np.array_equal(x1_out, x2_out), "x3 did NOT affect x output!"
    # Verify x3_absent x_out == (x1+x2) cast to fp16
    expected_x_absent = (x1.astype(np.float32) + x2.astype(np.float32)).astype(
        np.float16
    )
    assert np.array_equal(x1_out, expected_x_absent), "x3_absent x mismatch"
    # Verify x3_present x_out == ((x3+x1)+x2) cast to fp16 (kernel addition order)
    expected_x_present = (
        (x3.astype(np.float32) + x1.astype(np.float32)) + x2.astype(np.float32)
    ).astype(np.float16)
    assert np.array_equal(x2_out, expected_x_present), "x3_present x mismatch"

    # Verify rstd differs (x3 changes variance)
    assert not np.allclose(rstd1, rstd2), "x3 did NOT affect rstd!"

    # Case 3: FP4 E2M1, output_rstd=False -> rstd should be None
    y3, x3_out, mx3, rstd3 = add_rms_norm_dynamic_mx_quant_golden(
        x1,
        x2,
        gamma,
        beta=None,
        x3=None,
        dst_type=40,
        scale_alg=0,
        output_rstd=False,
    )
    assert rstd3 is None, "rstd should be None when output_rstd=False"
    print(f"[FP4 E2M1, no rstd] y={y3.shape}/{y3.dtype}, rstd={rstd3}")

    # Case 4: FP8 E5M2, cuBLAS scale_alg
    y4, x4, mx4, rstd4 = add_rms_norm_dynamic_mx_quant_golden(
        x1,
        x2,
        gamma,
        beta=None,
        x3=x3,
        dst_type=35,
        scale_alg=1,
        output_rstd=True,
    )
    print(
        f"[FP8 E5M2, cuBLAS] y={y4.shape}/{y4.dtype}, mxscale={mx4.shape}/{mx4.dtype}"
    )

    # Case 5: with beta
    beta = np.random.uniform(-1, 1, (D,)).astype(np.float16)
    y5, x5, mx5, rstd5 = add_rms_norm_dynamic_mx_quant_golden(
        x1,
        x2,
        gamma,
        beta=beta,
        x3=None,
        dst_type=36,
        scale_alg=0,
        output_rstd=True,
    )
    print(f"[with beta] y={y5.shape}/{y5.dtype}")

    # Verify beta actually affects y (y must differ from no-beta case)
    assert not np.array_equal(y1, y5), "beta did NOT affect y output!"

    # Case 6: aclnn V1 (no x3) — 9 positional args
    y_v1 = aclnn_add_rms_norm_dynamic_mx_quant_golden(
        x1, x2, gamma, beta, 1e-6, 0, "rint", 36, True
    )
    assert np.array_equal(y_v1[0], y5), "aclnn V1 y mismatch with kernel golden"
    assert y_v1[3] is not None, "aclnn V1 rstd should not be None when outputRstd=True"
    print(f"[aclnn V1, 9 args] y={y_v1[0].shape}, rstd is not None: OK")

    # Case 7: aclnn V2 with x3 present — 10 positional args
    y_ref_v2 = add_rms_norm_dynamic_mx_quant_golden(
        x1, x2, gamma, beta, x3, 1e-6, 0, "rint", 36, True
    )
    y_v2 = aclnn_add_rms_norm_dynamic_mx_quant_golden(
        x1, x2, gamma, beta, x3, 1e-6, 0, "rint", 36, True
    )
    assert np.array_equal(y_v2[0], y_ref_v2[0]), "aclnn V2 (x3 present) y mismatch"
    assert np.array_equal(y_v2[1], y_ref_v2[1]), "aclnn V2 (x3 present) x mismatch"
    print(f"[aclnn V2, 10 args, x3 present] y={y_v2[0].shape}: OK")

    # Case 8: aclnn V2 with x3=None — 10 positional args (the bug case)
    y_ref_none = add_rms_norm_dynamic_mx_quant_golden(
        x1, x2, gamma, beta, None, 1e-6, 0, "rint", 36, True
    )
    y_v2_none = aclnn_add_rms_norm_dynamic_mx_quant_golden(
        x1, x2, gamma, beta, None, 1e-6, 0, "rint", 36, True
    )
    assert np.array_equal(y_v2_none[0], y_ref_none[0]), (
        "aclnn V2 (x3=None) y mismatch — route detection bug!"
    )
    assert np.array_equal(y_v2_none[1], y_ref_none[1]), "aclnn V2 (x3=None) x mismatch"
    print("[aclnn V2, 10 args, x3=None] matches kernel golden: OK")

    # Case 9: aclnn V1 with beta=None — 9 positional args
    y_v1_nb = aclnn_add_rms_norm_dynamic_mx_quant_golden(
        x1, x2, gamma, None, 1e-6, 0, "rint", 40, False
    )
    assert y_v1_nb[3] is None, "aclnn V1 rstd should be None when outputRstd=False"
    print("[aclnn V1, beta=None, output_rstd=False] rstd=None: OK")

    # Case 10: all 4 dst_types with both scale_algs
    for dt in [40, 41, 36, 35]:
        for sa in [0, 1] if dt in (35, 36) else [0]:
            y_t, _, _, rstd_t = add_rms_norm_dynamic_mx_quant_golden(
                x1,
                x2,
                gamma,
                beta=beta,
                x3=x3,
                dst_type=dt,
                scale_alg=sa,
                output_rstd=True,
            )
            assert y_t is not None, f"Failed for dst_type={dt} scale_alg={sa}"
    print("[all dst_type×scale_alg combos] OK")

    print("SELF-TEST OK")
