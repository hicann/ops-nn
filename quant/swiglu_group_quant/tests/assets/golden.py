# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from ttk.utilities.dtypes import (
    numpy_bfloat16,
    numpy_float4_e1m2,
    numpy_float4_e2m1,
    numpy_float8_e4m3fn,
    numpy_float8_e5m2,
    numpy_float8_e8m0,
    numpy_to_torch_tensor,
    torch_to_numpy_tensor,
)


__golden__ = {
    "aclnn": {
        "aclnnSwigluGroupQuant": "aclnn_swiglu_group_quant_golden",
    },
    "kernel": {
        "swiglu_group_quant": "swiglu_group_quant_golden",
    },
}

__input__ = {
    "aclnn": {
        "aclnnSwigluGroupQuant": "aclnn_swiglu_group_quant_input",
    },
    "kernel": {
        "swiglu_group_quant": "swiglu_group_quant_input",
    },
}


DT_FLOAT8_E5M2 = 35
DT_FLOAT8_E4M3FN = 36
DT_FLOAT4_E2M1 = 40
DT_FLOAT4_E1M2 = 41
DEFAULT_CLAMP_LIMIT = -1.0

FP8_MAX_VALUE = {
    DT_FLOAT8_E5M2: (2 - 2 ** -2) * 2 ** 15,
    DT_FLOAT8_E4M3FN: (2 - 2 ** -2) * 2 ** 8,
}
E1M2_STEP = 0.25
E1M2_MAX_CODE = 7


def _to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        return torch_to_numpy_tensor(tensor.detach().cpu())
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


def _write_tensor(tensor, value):
    if tensor is None:
        return
    if isinstance(tensor, np.ndarray):
        tensor[...] = value.astype(tensor.dtype, copy=False)
        return
    import torch

    src = torch.as_tensor(value, device=tensor.device)
    if src.dtype != tensor.dtype:
        src = src.to(tensor.dtype)
    tensor.copy_(src.reshape(tensor.shape))


def _dtype_to_int(dst_type):
    if isinstance(dst_type, int):
        return dst_type
    text = str(dst_type)
    if text.endswith("float8_e5m2"):
        return DT_FLOAT8_E5M2
    if text.endswith("float8_e4m3fn"):
        return DT_FLOAT8_E4M3FN
    if "float4_e2m1" in text:
        return DT_FLOAT4_E2M1
    if "float4_e1m2" in text:
        return DT_FLOAT4_E1M2
    raise ValueError(f"Unsupported dst_type: {dst_type}")


def _fp8_dtype(dst_type):
    if dst_type == DT_FLOAT8_E5M2:
        return numpy_float8_e5m2()
    if dst_type == DT_FLOAT8_E4M3FN:
        return numpy_float8_e4m3fn()
    raise ValueError(f"Unsupported fp8 dst_type: {dst_type}")


def _fp4_dtype(dst_type):
    if dst_type == DT_FLOAT4_E2M1:
        return numpy_float4_e2m1()
    if dst_type == DT_FLOAT4_E1M2:
        return numpy_float4_e1m2()
    raise ValueError(f"Unsupported fp4 dst_type: {dst_type}")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _ceil_div(x, y):
    return (x + y - 1) // y


def _use_clamp(clamp_limit):
    return clamp_limit is not None and float(clamp_limit) != DEFAULT_CLAMP_LIMIT


def _compute_swiglu(x, weight=None, clamp_limit=None):
    x = _to_numpy(x).astype(np.float32)
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    hidden = orig_shape[-1] // 2
    x0 = x[:, :hidden]
    x1 = x[:, hidden:]

    if _use_clamp(clamp_limit):
        limit = float(clamp_limit)
        x0 = np.minimum(x0, limit)
        x1 = np.minimum(limit, np.maximum(x1, -limit))

    y_origin = _silu(x0) * x1
    weight = _to_numpy(weight)
    if weight is not None:
        y_origin *= weight.reshape(-1, 1).astype(np.float32)
    return y_origin.reshape(*orig_shape[:-1], hidden)


def _get_scale_and_inv(amax, round_scale, use_e8m0, fp8_max):
    amax = np.maximum(amax.astype(np.float32), np.float32(1e-4))
    scale = amax / np.float32(fp8_max)
    if not round_scale:
        return scale.astype(np.float32), (np.float32(fp8_max) / amax).astype(np.float32)

    scale_bits = scale.view(np.uint32)
    exp = ((scale_bits >> 23) & 0xFF).astype(np.int32)
    mantissa = scale_bits & ((1 << 23) - 1)
    exp_scale = exp - 127 + (mantissa != 0)
    if use_e8m0:
        scale_out = np.clip(exp_scale + 127, 0, 255).astype(np.uint8)
    else:
        scale_out = ((exp_scale + 127).astype(np.uint32) << 23).view(np.float32)
    inv_scale = ((127 - exp_scale).astype(np.uint32) << 23).view(np.float32)
    return scale_out, inv_scale


def _quantize_fp8(y_origin, dst_type, quant_mode, round_scale):
    if quant_mode == 1 and not round_scale:
        raise ValueError("quant_mode 1 only supports round_scale=True")

    block_size = 128 if quant_mode == 0 else 32
    y_shape = y_origin.shape
    hidden = y_shape[-1]
    y_2d = y_origin.reshape(-1, hidden).astype(np.float32)
    num_tokens = y_2d.shape[0]
    num_blocks = _ceil_div(hidden, block_size)
    pad_hidden = num_blocks * block_size
    y_pad = np.pad(y_2d, ((0, 0), (0, pad_hidden - hidden)), mode="constant")
    y_block = y_pad.reshape(num_tokens, num_blocks, block_size)
    amax = np.max(np.abs(y_block), axis=-1)

    scale_out, inv_scale = _get_scale_and_inv(
        amax, round_scale, quant_mode == 1, FP8_MAX_VALUE[dst_type])
    y_scaled = y_block * inv_scale.reshape(num_tokens, num_blocks, 1)
    y_fp8 = y_scaled.reshape(num_tokens, pad_hidden)[:, :hidden].astype(_fp8_dtype(dst_type))

    if quant_mode == 0:
        scale = scale_out.astype(np.float32).reshape(*y_shape[:-1], num_blocks)
    else:
        if num_blocks % 2 != 0:
            scale_out = np.pad(scale_out, ((0, 0), (0, 1)), mode="constant")
        scale = scale_out.view(numpy_float8_e8m0()).reshape(*y_shape[:-1], _ceil_div(num_blocks, 2), 2)
    return y_fp8.reshape(y_shape), scale


def _float32_to_bf16_bits(value):
    return value.astype(numpy_bfloat16()).view(np.uint16)


def _quantize_e1m2_nibble(y_scaled):
    mag = np.rint(np.abs(y_scaled) / E1M2_STEP).astype(np.int32)
    mag = np.clip(mag, 0, E1M2_MAX_CODE).astype(np.uint8)
    return np.where(y_scaled < 0, mag | 0x8, mag).astype(np.uint8)


def _quantize_fp4(y_origin, dst_type, pack=True):
    block_size = 32
    y_shape = y_origin.shape
    hidden = y_shape[-1]
    y_2d = y_origin.reshape(-1, hidden).astype(np.float32)
    num_tokens = y_2d.shape[0]
    if hidden % block_size != 0:
        raise ValueError("MxFp4 golden only supports hidden divisible by 32")

    num_blocks = hidden // block_size
    y_block = y_2d.reshape(num_tokens, num_blocks, block_size)

    fp4_emax_exp = 2 if dst_type == DT_FLOAT4_E2M1 else 0
    amax = np.max(np.abs(y_block), axis=-1)
    e_amax = ((_float32_to_bf16_bits(amax) >> 7) & 0xFF).astype(np.int32)
    e8m0 = np.maximum(e_amax - fp4_emax_exp, 0).astype(np.uint8)
    inv_scale = np.exp2(127.0 - e8m0.astype(np.float32)).astype(np.float32)
    y_scaled = y_block * inv_scale.reshape(num_tokens, num_blocks, 1)

    if dst_type == DT_FLOAT4_E2M1:
        fp4_values = y_scaled.reshape(num_tokens, hidden).astype(np.float32).astype(_fp4_dtype(dst_type))
        nibble = fp4_values.view(np.uint8).reshape(num_tokens, hidden)
    elif dst_type == DT_FLOAT4_E1M2:
        nibble = _quantize_e1m2_nibble(y_scaled.astype(np.float32)).reshape(num_tokens, hidden)
        fp4_values = nibble.view(_fp4_dtype(dst_type))
    else:
        raise RuntimeError(f"MxFp4 golden only implements FLOAT4_E2M1/E1M2, got {dst_type}")

    if num_blocks % 2 != 0:
        e8m0 = np.pad(e8m0, ((0, 0), (0, 1)), mode="constant")
    scale = e8m0.view(numpy_float8_e8m0()).reshape(*y_shape[:-1], _ceil_div(num_blocks, 2), 2)
    if not pack:
        return fp4_values.reshape(y_shape), scale

    packed = np.bitwise_or(
        nibble[:, 0::2],
        nibble[:, 1::2] << np.uint8(4),
    ).astype(np.uint8)
    packed = packed.view(_fp4_dtype(dst_type))
    return packed.reshape(*y_shape[:-1], _ceil_div(hidden, 2)), scale


def _cast_y_origin(y_origin, x):
    x_np = _to_numpy(x)
    if x_np.dtype.name == "bfloat16":
        return y_origin.astype(numpy_bfloat16())
    return y_origin.astype(x_np.dtype, copy=False)


def _maybe_to_torch(value, use_torch):
    if not use_torch:
        return value
    if value is None or not isinstance(value, np.ndarray):
        return value
    if value.dtype.name in ("float4_e2m1", "float4_e1m2", "float8_e8m0"):
        return value
    return numpy_to_torch_tensor(value)


def aclnn_swiglu_group_quant_golden(
    x,
    weightOptional,
    groupIndexOptional,
    dstType,
    quantMode,
    blockSize,
    roundScale,
    clampLimit,
    outputOrigin,
    yOut,
    scaleOutOut,
    yOriginOut,
    **kwargs,
):
    del blockSize, outputOrigin
    use_torch = kwargs.get("use_torch", False)
    dst_type = _dtype_to_int(dstType)
    quant_mode = int(quantMode)
    y_origin = _compute_swiglu(x, weightOptional, clampLimit)

    if dst_type in (DT_FLOAT4_E2M1, DT_FLOAT4_E1M2):
        y, scale_out = _quantize_fp4(y_origin, dst_type)
    else:
        y, scale_out = _quantize_fp8(y_origin, dst_type, quant_mode, bool(roundScale))

    y_origin = _cast_y_origin(y_origin, x)
    return _maybe_to_torch(y, use_torch), _maybe_to_torch(scale_out, use_torch), _maybe_to_torch(y_origin, use_torch)


def aclnn_swiglu_group_quant_input(
    x,
    weightOptional,
    groupIndexOptional,
    dstType,
    quantMode,
    blockSize,
    roundScale,
    clampLimit,
    outputOrigin,
    yOut,
    scaleOutOut,
    yOriginOut,
    **kwargs,
):
    del quantMode, blockSize, roundScale
    del clampLimit, outputOrigin, yOut, scaleOutOut, yOriginOut
    dst_type = _dtype_to_int(dstType)
    if dst_type in (DT_FLOAT4_E2M1, DT_FLOAT4_E1M2):
        x_np = _to_numpy(x)
        hidden = x_np.shape[-1] // 2
        stable_x = np.ones(x_np.shape, dtype=np.float32)
        sign_pattern = np.where(np.arange(hidden) % 2 == 0, 1.0, -1.0).astype(np.float32)
        stable_x.reshape(-1, x_np.shape[-1])[:, hidden:] = sign_pattern
        _write_tensor(x, stable_x)
        if weightOptional is not None:
            _write_tensor(weightOptional, np.ones(_to_numpy(weightOptional).shape, dtype=np.float32))
    if groupIndexOptional is None:
        return
    token_num = int(np.prod(_to_numpy(x).shape[:-1]))
    group_index = _to_numpy(groupIndexOptional).reshape(-1)
    group_index[...] = 0
    if group_index.size == 1:
        group_index[0] = token_num
    else:
        group_index[0] = token_num // 2
        group_index[1] = token_num - group_index[0]
    if isinstance(groupIndexOptional, np.ndarray):
        return
    import torch

    groupIndexOptional.copy_(torch.as_tensor(group_index, device=groupIndexOptional.device))


def swiglu_group_quant_golden(
    x,
    weight,
    group_index,
    dst_type=DT_FLOAT8_E4M3FN,
    quant_mode=0,
    block_size=0,
    round_scale=False,
    clamp_limit=DEFAULT_CLAMP_LIMIT,
    output_origin=False,
    **kwargs,
):
    del group_index, block_size, kwargs
    dst_type = _dtype_to_int(dst_type)
    quant_mode = int(quant_mode)
    y_origin = _compute_swiglu(x, weight, clamp_limit)

    if dst_type in (DT_FLOAT4_E2M1, DT_FLOAT4_E1M2):
        y, scale_out = _quantize_fp4(y_origin, dst_type, pack=False)
    else:
        y, scale_out = _quantize_fp8(y_origin, dst_type, quant_mode, bool(round_scale))

    y_origin = _cast_y_origin(y_origin, x) if output_origin else None
    return y, scale_out, y_origin


def swiglu_group_quant_input(
    x,
    weight,
    group_index,
    dst_type=DT_FLOAT8_E4M3FN,
    quant_mode=0,
    block_size=0,
    round_scale=False,
    clamp_limit=DEFAULT_CLAMP_LIMIT,
    output_origin=False,
    **kwargs,
):
    del kwargs
    aclnn_swiglu_group_quant_input(
        x,
        weight,
        group_index,
        dst_type,
        quant_mode,
        block_size,
        round_scale,
        clamp_limit,
        output_origin,
        None,
        None,
        None,
    )
    return x, weight, group_index
