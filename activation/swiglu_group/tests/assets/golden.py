# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden / input plugin for the SwigluGroup operator (plain SwiGLU, no quant).

    y = silu(A) * B

where the last dim of ``x`` is evenly split into ``A`` and ``B``. ``weight`` is an
optional per-token scale; ``clamp_limit`` (> 0) clamps the SwiGLU inputs before
activation. ``group_index`` (count mode) only bounds how many rows are processed
(realBs = min(sum(group_index), bs)); the input plugin makes its sum equal the
token count so every row is produced and the all-rows golden matches.
"""

import numpy as np

from ttk.utilities.dtypes import (
    numpy_bfloat16,
    numpy_to_torch_tensor,
    torch_to_numpy_tensor,
)


__golden__ = {
    "kernel": {"swiglu_group": "swiglu_group_golden"},
    "aclnn": {"aclnnSwigluGroup": "aclnn_swiglu_group_golden"},
}

__input__ = {
    "kernel": {"swiglu_group": "swiglu_group_input"},
    "aclnn": {"aclnnSwigluGroup": "aclnn_swiglu_group_input"},
}

DEFAULT_CLAMP_LIMIT = -1.0

# inf / nan / boundary pool used when the testcase name contains "edge". 65504 = fp16 max,
# 6.10e-5 = fp16 smallest normal, 5.96e-8 = fp16 smallest denormal. The golden computes
# SwiGLU in numpy, which propagates inf/nan IEEE-consistently with the kernel.
EDGE_VALUES = np.array(
    [np.inf, -np.inf, np.nan, 0.0, -0.0, 1.0, -1.0, 2.0, -2.0,
     65504.0, -65504.0, 6.10e-5, -6.10e-5, 5.96e-8, 1.0e4, -1.0e4],
    dtype=np.float32,
)


def _is_edge_case(kwargs):
    return "edge" in str(kwargs.get("testcase_name", "")).lower()


def _fill_edge(x, weight):
    """Fill x entirely with the inf/nan/boundary pool; set weight (if any) to ones."""
    x_np = _to_numpy(x)
    _write_tensor(x, np.resize(EDGE_VALUES, x_np.size).reshape(x_np.shape))
    if weight is not None:
        _write_tensor(weight, np.ones(_to_numpy(weight).shape, dtype=np.float32))


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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


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

    y = _silu(x0) * x1
    weight = _to_numpy(weight)
    if weight is not None:
        y *= weight.reshape(-1, 1).astype(np.float32)
    return y.reshape(*orig_shape[:-1], hidden)


def _cast_like(y, x):
    x_np = _to_numpy(x)
    if x_np.dtype.name == "bfloat16":
        return y.astype(numpy_bfloat16())
    return y.astype(x_np.dtype, copy=False)


def _maybe_to_torch(value, use_torch):
    if not use_torch or value is None or not isinstance(value, np.ndarray):
        return value
    return numpy_to_torch_tensor(value)


def _fill_group_index(x, group_index):
    """Make sum(group_index) == token count so realBs == bs and all rows are produced."""
    if group_index is None:
        return
    token_num = int(np.prod(_to_numpy(x).shape[:-1]))
    gi = _to_numpy(group_index).reshape(-1).copy()
    gi[...] = 0
    n = gi.size
    if n == 1:
        gi[0] = token_num
    else:
        base = token_num // n
        gi[:] = base
        gi[-1] = token_num - base * (n - 1)
    _write_tensor(group_index, gi)


# --------------------------------------------------------------------------- #
# Kernel level (numpy)
# --------------------------------------------------------------------------- #
def swiglu_group_golden(x, weight, group_index, clamp_limit=DEFAULT_CLAMP_LIMIT, **kwargs):
    """Golden for swiglu_group. Parameters follow swiglu_group_def.cpp (no outputs)."""
    del group_index, kwargs
    return _cast_like(_compute_swiglu(x, weight, clamp_limit), x)


def swiglu_group_input(x, weight, group_index, clamp_limit=DEFAULT_CLAMP_LIMIT, **kwargs):
    del clamp_limit
    if _is_edge_case(kwargs):
        _fill_edge(x, weight)
    _fill_group_index(x, group_index)
    return [x, weight, group_index]


# --------------------------------------------------------------------------- #
# ACLNN level (torch)
# --------------------------------------------------------------------------- #
def aclnn_swiglu_group_golden(x, weightOptional, groupIndexOptional, clampLimit, yOut, **kwargs):
    """Golden for aclnnSwigluGroup. Parameters follow aclnn_swiglu_group.h."""
    del groupIndexOptional, yOut
    use_torch = kwargs.get("use_torch", False)
    return _maybe_to_torch(_cast_like(_compute_swiglu(x, weightOptional, clampLimit), x), use_torch)


def aclnn_swiglu_group_input(x, weightOptional, groupIndexOptional, clampLimit, yOut, **kwargs):
    del clampLimit, yOut
    if _is_edge_case(kwargs):
        _fill_edge(x, weightOptional)
    _fill_group_index(x, groupIndexOptional)
    return [x, weightOptional, groupIndexOptional]
