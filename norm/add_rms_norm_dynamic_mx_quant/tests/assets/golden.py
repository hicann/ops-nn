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


"""Self-contained Add + RMSNorm + MX quantization reference for Ascend 950.

Inputs and RMSNorm arithmetic use fp32. The normalized value is explicitly
rounded to the original x1 storage dtype before MX quantization. TTK's Promote
mode must not remove this operator-defined intermediate conversion.

The MX encoding helpers come from quant/dynamic_mx_quant/tests/assets/golden.py
at the frozen baseline 7429363a. The OCP exponent rule and RMSNorm calculation
come from the operator's original golden at the same commit. No sibling golden
module is imported or monkey-patched.

Output order is y, x, mxscale, rstd. Only output_rstd=False suppresses rstd;
for an empty reduction axis, an enabled rstd contains NaN (REDUCE_EMPTY path).
"""

import copy
import re

import numpy as np
import torch
from ml_dtypes import bfloat16


__spec__ = {
    "add_rms_norm_dynamic_mx_quant": "AddRmsNormDynamicMxQuantTestSpec",
    "aclnnAddRmsNormDynamicMxQuant": "AclnnAddRmsNormDynamicMxQuantTestSpec",
    "aclnnAddRmsNormDynamicMxQuantV2": "AclnnAddRmsNormDynamicMxQuantV2TestSpec",
    "torch.ops.cann_ops_nn.add_rms_norm_dynamic_quant": "AddRmsNormDynamicMxQuantE2ESpec",
}

DATA_TYPE_INT_TO_STR = {
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    40: "float4_e2m1",
    41: "float4_e1m2",
}
_EMAX_MAP = {
    "float8_e4m3fn": 8,
    "float8_e5m2": 15,
    "float4_e2m1": 2,
    "float4_e1m2": 0,
}


def _numpy_mx_dtype(mx_ele_dtype):
    if mx_ele_dtype == "float4_e2m1":
        from en_dtypes import float4_e2m1

        return float4_e2m1
    if mx_ele_dtype == "float4_e1m2":
        from en_dtypes import float4_e1m2

        return float4_e1m2
    if mx_ele_dtype == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if mx_ele_dtype == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if mx_ele_dtype == "float8_e8m0":
        from en_dtypes import float8_e8m0

        return float8_e8m0
    raise ValueError(f"Unsupported mx dtype: {mx_ele_dtype}")


def _get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def _mx_reshape_to_blocks(fp_array, axis, block_size):
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = np.pad(fp_array, pad, "constant")
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape


def _mx_undo_reshape_to_blocks(fp_array, axis, orig_shape, padded_shape):
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array


def _mx_calculate_share_exp(fp_array, scale_axis, mx_ele_dtype):
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


def _mx_calculate_share_exp_nv(fp_array, scale_axis, mx_ele_dtype):
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(
        np.float32
    )
    s_fp32 = fp_abs_max / max_norm
    binary_ints = np.array(s_fp32.view(np.uint32))
    exponent_mask = np.uint32(0x7F800000)
    mantissa_mask = np.uint32(0x007FFFFF)
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(np.int16)
    mantissas = binary_ints & mantissa_mask
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    condition_2 = (exponents_int16 == 0) & (mantissas > 2**22)
    exponents_int16 = np.where(
        (condition_1 | condition_2), exponents_int16 + 1, exponents_int16
    )
    res = (exponents_int16 - 127).astype(np.float32)
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_round_mantissa(fp_array, round_mode):
    if round_mode in ("rint", "even"):
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r"e(\d+)m(\d+)", mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = fp_array / (2**share_exp)
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))).astype(
        fp_array.dtype, copy=False
    )
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    ret = ret / (2**private_exp) * (2**mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2**mantissa_bits) * (2**private_exp)
    max_norm = _get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def _pad_to_even(tensor, axis):
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
        )

    shape = tensor.shape
    length = shape[axis]

    if length % 2 == 0:
        return tensor

    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)

    padded_tensor = np.pad(tensor, pad_width, mode="constant", constant_values=2**-127)
    return padded_tensor


def _interleave(tensor, axis, n_group=2):
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
        )
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(
            f"Axis length ({length}) must be divisible by n_group ({n_group})"
        )

    group_length = length // n_group
    shape = list(tensor.shape)

    new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1 :]
    reshaped = tensor.reshape(new_shape)

    transpose_order = (
        list(range(0, axis + 1))
        + list(range(axis + 2, len(new_shape)))
        + [
            axis + 1,
        ]
    )

    transposed = reshaped.transpose(transpose_order)
    return transposed


def _mx_quantize(
    fp_array,
    mx_ele_dtype="float4_e2m1",
    axis=-1,
    block_size=32,
    round_mode="rint",
    scale_alg=0,
):
    if not isinstance(fp_array, np.ndarray):
        raise RuntimeError(
            f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}"
        )
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(
            f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}"
        )
    if mx_ele_dtype not in (
        "float4_e2m1",
        "float4_e1m2",
        "float8_e4m3fn",
        "float8_e5m2",
    ):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(
        fp_array, axis, block_size
    )
    if scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
    else:
        share_exp = _mx_calculate_share_exp_nv(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    ele_array = _mx_quantize_to_element_format(
        fp_array, share_exp, mx_ele_dtype, round_mode
    )
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    ele_dtype_np = _numpy_mx_dtype(mx_ele_dtype)
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    scale_array_pad = _pad_to_even(scale_array, axis=axis)

    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = _interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    scale_array = scale_array_pad.astype(_numpy_mx_dtype("float8_e8m0"), copy=False)

    return scale_array, ele_array


def _to_numpy(value):
    if value is None or isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            return value.view(torch.uint16).numpy().view(bfloat16)
        return value.numpy()
    return np.asarray(value)


def _storage_dtype(x1, metadata):
    # Snapshot fields are provided by this operator's TTK branch before
    # promotion. E2E's tensor_dtypes already retain the original dtype.
    for key in ("original_input_dtypes", "original_tensor_dtypes", "tensor_dtypes"):
        dtypes = metadata.get(key)
        if dtypes:
            return np.dtype(str(dtypes[0]).removeprefix("torch."))
    return x1.dtype


def _to_fp32(arr):
    """numpy array (incl ml_dtypes.bfloat16) -> float32 numpy array."""
    return arr.astype(np.float32)


def _np_to_torch_fp32(arr):
    """numpy array (incl ml_dtypes.bfloat16) -> torch.float32 tensor."""
    return torch.from_numpy(_to_fp32(arr))


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
    storage_dtype=None,
):
    """Chunked golden — identical math to the full path, processed per batch-slice.

    Flattens batch dims to (total_batch, norm_dim), processes rows in chunks of
    ~80M elements, pre-allocates full output arrays from the first chunk's
    output shapes, then fills them.  Peak memory ≈ inputs + outputs + one
    chunk of fp32 intermediates (~2 GiB) instead of ~8× full-size intermediates.
    """
    in_dtype = np.dtype(storage_dtype) if storage_dtype is not None else x1.dtype
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

        scale_c, ele_c = _mx_quantize(
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
    x1, x2, gamma, beta, x3 = (_to_numpy(t) for t in (x1, x2, gamma, beta, x3))
    in_dtype = _storage_dtype(x1, kwargs)
    norm_dim = x1.shape[-1]

    # Empty batches have empty outputs; an empty reduction axis has NaN rstd.
    # Skip reductions over empty arrays while preserving all output shapes.
    if x1.size == 0:
        scale_shape = x1.shape[:-1] + ((norm_dim + 63) // 64, 2)
        return (
            np.empty(x1.shape, dtype=_numpy_mx_dtype(DATA_TYPE_INT_TO_STR[dst_type])),
            np.empty(x1.shape, dtype=in_dtype),
            np.empty(scale_shape, dtype=_numpy_mx_dtype("float8_e8m0")),
            np.full(x1.shape[:-1] + (1,), np.nan, dtype=np.float32)
            if output_rstd
            else None,
        )

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
            storage_dtype=in_dtype,
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

    inv_norm = 1.0 / float(norm_dim)
    var = torch.sum(x * x, dim=-1, keepdim=True) * inv_norm
    rstd = torch.rsqrt(var + float(epsilon))

    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    scale_array, ele_array = _mx_quantize(
        normed_f,
        mx_ele_dtype=dst_type_str,
        axis=-1,
        block_size=32,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )

    rstd_out = rstd.numpy().astype(np.float32) if output_rstd else None

    return ele_array, x_out, scale_array, rstd_out


def aclnn_add_rms_norm_dynamic_mx_quant_golden(
    x1,
    x2,
    gamma,
    beta,
    epsilon,
    scale_alg,
    round_mode,
    dst_type,
    output_rstd,
    y_out=None,
    x_out=None,
    mxscale_out=None,
    rstd_out=None,
    **kwargs,
):
    """V1 header order, including output slots passed by the ACLNN param plan."""
    return add_rms_norm_dynamic_mx_quant_golden(
        x1,
        x2,
        gamma,
        beta,
        None,
        epsilon,
        scale_alg,
        round_mode,
        dst_type,
        output_rstd,
        **kwargs,
    )


def aclnn_add_rms_norm_dynamic_mx_quant_v2_golden(
    x1,
    x2,
    gamma,
    beta,
    x3_optional,
    epsilon,
    scale_alg,
    round_mode,
    dst_type,
    output_rstd,
    y_out=None,
    x_out=None,
    mxscale_out=None,
    rstd_out=None,
    **kwargs,
):
    """V2 has an explicit optional x3 slot; no case-name or arg-count inference."""
    return add_rms_norm_dynamic_mx_quant_golden(
        x1,
        x2,
        gamma,
        beta,
        x3_optional,
        epsilon,
        scale_alg,
        round_mode,
        dst_type,
        output_rstd,
        **kwargs,
    )


_TOLERANCE = {
    # Preserve the current TTK defaults; do not override numeric thresholds.
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
    "float32": {"standard": "mix_tolerance"},
    "float8_e4m3fn": {"standard": "mix_tolerance"},
    "float8_e5m2": {"standard": "mix_tolerance"},
    "float4_e2m1": {"standard": "binary_equal"},
    "float4_e1m2": {"standard": "binary_equal"},
    "float8_e8m0": {"standard": "binary_equal"},
}


def _independent_reference(
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
    """Independent NumPy reduction + representable-value lookup reference.

    This path does not call the primary golden or any of its MX helpers.
    E8M0 bytes are constructed from mathematical exponents, and element
    rounding searches the target format's finite positive value table.
    The torch provider supplies CPU input transport, not an NPU benchmark.
    """
    import en_dtypes
    import ml_dtypes

    def host(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                return value.float().numpy().astype(ml_dtypes.bfloat16)
            return value.numpy()
        return np.asarray(value)

    a, b, g, bias, extra = map(host, (x1, x2, gamma, beta, x3))
    storage = a.dtype
    element_type = {
        35: ml_dtypes.float8_e5m2,
        36: ml_dtypes.float8_e4m3fn,
        40: en_dtypes.float4_e2m1,
        41: en_dtypes.float4_e1m2,
    }[dst_type]
    columns = a.shape[-1]
    rows = int(np.prod(a.shape[:-1], dtype=np.int64))
    if a.size == 0:
        return (
            np.empty(a.shape, dtype=element_type),
            np.empty(a.shape, dtype=storage),
            np.empty(
                a.shape[:-1] + ((columns + 63) // 64, 2), dtype=en_dtypes.float8_e8m0
            ),
            np.full(a.shape[:-1] + (1,), np.nan, np.float32) if output_rstd else None,
        )
    summed = a.astype(np.float32)
    if extra is not None:
        summed = extra.astype(np.float32) + summed
    summed = summed + b.astype(np.float32)
    variance = np.mean(np.square(summed), axis=-1, keepdims=True, dtype=np.float32)
    reciprocal = np.reciprocal(np.sqrt(variance + np.float32(epsilon)))
    normalized = (summed * reciprocal) * g.astype(np.float32)
    if bias is not None:
        normalized = normalized + bias.astype(np.float32)
    normalized = normalized.astype(storage).astype(np.float32)

    block_count = (columns + 31) // 32
    blocked = np.pad(
        normalized.reshape(rows, columns), ((0, 0), (0, block_count * 32 - columns))
    )
    blocked = blocked.reshape(rows, block_count, 32)
    maximum = np.max(np.abs(blocked), axis=-1)
    emax = {35: 15, 36: 8, 40: 2, 41: 0}[dst_type]
    max_value = {35: 57344.0, 36: 448.0, 40: 6.0, 41: 1.75}[dst_type]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if scale_alg == 1 and dst_type in (35, 36):
            ratio = maximum / np.float32(max_value)
            fraction, exponent = np.frexp(ratio)
            exponent = exponent - (fraction == np.float32(0.5))
        else:
            _, exponent = np.frexp(maximum)
            exponent = exponent - 1 - emax
        exponent = np.where(maximum == 0, -127, exponent)
        exponent = np.clip(exponent, -127, 127).astype(np.int32)
        invalid = ~np.isfinite(maximum)
        scale = np.ldexp(np.ones_like(maximum, dtype=np.float32), exponent)
        scaled = blocked / scale[..., None]
    scaled = np.where(invalid[..., None], 0, scaled)

    sign_bit = 3 if dst_type in (40, 41) else 7
    raw_table = np.arange(1 << sign_bit, dtype=np.uint8)
    value_table = raw_table.view(element_type).astype(np.float32)
    finite = np.isfinite(value_table)
    raw_table, value_table = raw_table[finite], value_table[finite]
    magnitude = np.clip(np.abs(scaled), 0, max_value)
    upper = np.searchsorted(value_table, magnitude, side="left")
    upper = np.minimum(upper, len(value_table) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = value_table[upper] == magnitude
    lower = np.where(exact, upper, lower)
    if round_mode == "floor":
        chosen = np.where(np.signbit(scaled), upper, lower)
    else:
        low_distance = magnitude - value_table[lower]
        high_distance = value_table[upper] - magnitude
        if round_mode == "rint":
            use_upper = (high_distance < low_distance) | (
                (high_distance == low_distance) & ((raw_table[upper] & 1) == 0)
            )
        elif round_mode == "round":
            use_upper = high_distance <= low_distance
        else:
            raise ValueError(f"Unsupported round_mode: {round_mode}")
        chosen = np.where(use_upper, upper, lower)
    encoded = raw_table[chosen] | (np.signbit(scaled).astype(np.uint8) << sign_bit)
    quantized = (
        encoded.view(element_type)
        .reshape(rows, block_count * 32)[:, :columns]
        .reshape(a.shape)
    )
    scale_bytes = np.where(invalid, 255, exponent + 127).astype(np.uint8)
    if block_count % 2:
        scale_bytes = np.pad(scale_bytes, ((0, 0), (0, 1)))
    scales = scale_bytes.view(en_dtypes.float8_e8m0).reshape(
        a.shape[:-1] + ((block_count + 1) // 2, 2)
    )
    return (
        quantized,
        summed.astype(storage),
        scales,
        reciprocal if output_rstd else None,
    )


class IndependentMxReference:
    def __call__(
        self,
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
        return _independent_reference(
            x1,
            x2,
            gamma,
            beta,
            x3,
            epsilon,
            scale_alg,
            round_mode,
            dst_type,
            output_rstd,
        )


class IndependentMxReferenceV1:
    def __call__(
        self,
        x1,
        x2,
        gamma,
        beta=None,
        epsilon=1e-6,
        scaleAlg=0,
        roundMode="rint",
        dstType=40,
        outputRstd=False,
    ):
        return _independent_reference(
            x1, x2, gamma, beta, None, epsilon, scaleAlg, roundMode, dstType, outputRstd
        )


class IndependentMxReferenceV2:
    def __call__(
        self,
        x1,
        x2,
        gamma,
        beta=None,
        x3Optional=None,
        epsilon=1e-6,
        scaleAlg=0,
        roundMode="rint",
        dstType=40,
        outputRstd=False,
    ):
        return _independent_reference(
            x1,
            x2,
            gamma,
            beta,
            x3Optional,
            epsilon,
            scaleAlg,
            roundMode,
            dstType,
            outputRstd,
        )


class AddRmsNormDynamicMxQuantTestSpec:
    golden = staticmethod(add_rms_norm_dynamic_mx_quant_golden)
    third_party = {"torch": IndependentMxReference}
    tolerance = _TOLERANCE


class AclnnAddRmsNormDynamicMxQuantTestSpec:
    golden = staticmethod(aclnn_add_rms_norm_dynamic_mx_quant_golden)
    third_party = {"torch": IndependentMxReferenceV1}
    tolerance = _TOLERANCE


class AclnnAddRmsNormDynamicMxQuantV2TestSpec:
    golden = staticmethod(aclnn_add_rms_norm_dynamic_mx_quant_v2_golden)
    third_party = {"torch": IndependentMxReferenceV2}
    tolerance = _TOLERANCE


class AddRmsNormDynamicMxQuantE2ESpec:
    golden = staticmethod(add_rms_norm_dynamic_mx_quant_golden)
    third_party = {"torch": IndependentMxReference}
    # Retain the previously declared E2E tolerances for ordinary floats.
    tolerance = {
        **_TOLERANCE,
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

    @staticmethod
    def pre_compare(y, x, mxscale, rstd, g_y, g_x, g_mxscale, g_rstd):
        # torch has no native FP4 tensor. The extension exposes packed bytes,
        # low nibble first; reinterpret them as the declared semantic dtype.
        if y is not None and g_y is not None and y.dtype == np.uint8:
            if g_y.dtype.name in ("float4_e2m1", "float4_e1m2"):
                nibbles = np.empty(y.size * 2, dtype=np.uint8)
                raw = np.ascontiguousarray(y).reshape(-1)
                nibbles[0::2] = raw & 15
                nibbles[1::2] = raw >> 4
                y = nibbles.view(g_y.dtype).reshape(g_y.shape)
        # The graph converter exposes E8M0 bytes through a uint8 tensor.
        if mxscale is not None and g_mxscale is not None and mxscale.dtype == np.uint8:
            if g_mxscale.dtype.name == "float8_e8m0":
                mxscale = mxscale.view(g_mxscale.dtype).reshape(g_mxscale.shape)
        return [y, x, mxscale, rstd, g_y, g_x, g_mxscale, g_rstd]
