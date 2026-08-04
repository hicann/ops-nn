#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""SGD kernel golden.

Formula (aligned with docs/spec.yaml math_semantics.formula):
    grad     = gradient + parameters * weight_decay      (only when weight_decay != 0)
    accum_t  = accum * m + grad                          (UNCONDITIONAL)
    accum_t -= grad * (1 - stat) * dampening             (only when dampening != 0)
    p_out    = p - (grad*lr + accum_t*m*lr)  if nesterov else  p - accum_t*lr
    m != 0 : accum_out = accum_t ; stat_out = 0
    m == 0 : accum_out / stat_out keep the input bit pattern (no writeback)

The arithmetic runs on torch tensors in the float32 domain, matching what the NPU
does, and follows the repo convention of lifting bfloat16 / float16 to float32
before handing the buffer to torch and casting the result back afterwards.

Three rules that must not be violated:
  1. accum_t is computed UNCONDITIONALLY. Do not treat "m == 0" as "multiplying by
     zero can be skipped": when accum holds +-inf, 0 * inf = NaN and that NaN must
     propagate into parameters_out per IEEE 754.
  2. When weight_decay == 0 / dampening == 0 the corresponding step is genuinely
     skipped rather than written as "multiply by zero" (same 0 * inf problem).
  3. The m == 0 branch returns the input buffers themselves, not values recomputed
     to be equal. NaN payloads, -0.0 and +-inf must survive bit for bit, so that
     branch never round-trips through torch.

Note: learning_rate and momentum share the dtype of parameters, so under float16 /
bfloat16 those two scalars are already quantized (lr = 0.1 is not exactly 0.1 in
float16). This function consumes the tensor values as given and never rebuilds them
from Python float literals.
"""

import numpy as np

__golden__ = {"kernel": {"sgd": "sgd_golden"}}


def _to_f32_np(arr):
    """Lift an input to the float32 compute domain (the NPU works in float32 too).

    TTK hands over bfloat16 as a real ml_dtypes bfloat16 dtype, which supports
    astype(np.float32) directly. Only a raw uint16 bit pattern needs manual shifting.
    torch has no bfloat16 numpy bridge, so this widening also makes the buffer
    something torch.from_numpy can take.
    """
    a = np.asarray(arr)
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(a.astype(np.float32))


def _cast_back(t, ref):
    """Cast a torch float32 result back to the reference tensor dtype.

    Never return a uint16 view here: the harness clamps goldens with
    array[array < dtype_min] = -inf, which raises OverflowError on integer arrays.
    """
    return t.numpy().astype(np.asarray(ref).dtype)


def _scalar(arr):
    """Read a [1] / 0-d scalar tensor out as an exactly-representable float32 value.

    float() widens float32 to double losslessly, and torch casts the Python scalar
    back to the tensor dtype when it meets a float32 tensor, so the round trip adds
    no rounding of its own.
    """
    return float(_to_f32_np(arr).reshape(-1)[0])


def sgd_golden(
    parameters,
    gradient,
    learning_rate,
    accum,
    momentum,
    stat,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    **kwargs,
):
    """Golden function for SGD kernel.

    Supported dtypes: float32, float16, bfloat16 (all six inputs share one dtype).

    Args:
        parameters: weights to update (numpy.ndarray)
        gradient: gradient, same shape and dtype as parameters
        learning_rate: scalar tensor of shape [1]
        accum: momentum accumulator, same shape and dtype as parameters
        momentum: scalar tensor of shape [1]
        stat: per-element first-step flag, same shape and dtype as parameters
        dampening: float attribute, default 0.0
        weight_decay: float attribute, default 0.0
        nesterov: bool attribute, default False

    Returns:
        [parameters_out, accum_out, stat_out], each cast back to the input dtype.
        All three are in-place writeback slots; accum_out and stat_out return the
        input arrays unchanged when momentum == 0 (the same bits, not recomputed
        equal values).
    """
    import torch

    p32 = torch.from_numpy(_to_f32_np(parameters))
    g32 = torch.from_numpy(_to_f32_np(gradient))
    a32 = torch.from_numpy(_to_f32_np(accum))
    s32 = torch.from_numpy(_to_f32_np(stat))
    lr = _scalar(learning_rate)
    m = _scalar(momentum)

    d = float(dampening)
    wd = float(weight_decay)
    nest = bool(nesterov)

    # Step 1: genuinely skipped when weight_decay == 0
    grad = (g32 + p32 * wd) if wd != 0.0 else g32

    # Step 2: unconditional
    accum_t = a32 * m + grad

    # Step 3: genuinely skipped when dampening == 0
    if d != 0.0:
        accum_t = accum_t - grad * ((1.0 - s32) * d)

    # Step 4: unconditional writeback
    if nest:
        p_new = p32 - (grad * lr + accum_t * m * lr)
    else:
        p_new = p32 - accum_t * lr

    parameters_out = _cast_back(p_new, parameters)

    # Step 5: momentum != 0 mask. IEEE != is used, so -0.0 counts as zero while
    # 1e-8 / 1e-30 count as non-zero.
    if m != 0.0:
        accum_out = _cast_back(accum_t, accum)
        stat_out = _cast_back(torch.zeros_like(s32), stat)
    else:
        # Keep the input bits: return copies of the inputs, no numeric rebuild.
        accum_out = np.asarray(accum).copy()
        stat_out = np.asarray(stat).copy()

    return [parameters_out, accum_out, stat_out]
