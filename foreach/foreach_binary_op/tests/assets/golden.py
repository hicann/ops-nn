#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
TTK custom golden for foreach_binary_op.

Compute formula (foreach_binary_op_proto.h:21; op_kernel/arch35/foreach_binary_op_simt.h:52):
    y[t][i] = x1[t][i] <op> x2[t][i]   (t = 0..n-1 list elements, i over numel)
where <op> is selected by REQUIRED attr op_code:
    0 = add, 1 = sub, 2 = mul, 3 = div.

Non-inplace: output y is a separate TensorList of n sub-tensors, same per-tensor shape/dtype
as x1 (and x2). Output order == x1 sub-tensor order.

dtype handling mirrors the kernel (simt.h):
  * float32      : direct compute.
  * float16/bf16 : cast up to float32, compute, cast back (BinaryApply on T=fp32 after Cast*).
  * int32        : native integer ops with 2's-complement wraparound (NO saturation) for
                   add/sub/mul. For div: b == 0 -> 0 (device guard, simt.h:65), otherwise
                   integer division truncated toward zero (C semantics).
  * float div b == 0 -> IEEE inf/nan, left as-is (simt.h:67).

Positional args (TTK passes context.input_arrays unflattened, in CSV input_shapes order):
    x1_list : list of numpy arrays  (DYNAMIC TensorList x1, n sub-tensors)
    x2_list : list of numpy arrays  (DYNAMIC TensorList x2, sync per-tensor with x1)
op_code is delivered via **kwargs (TTK passes parsed `attributes` entries as kwargs).
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

OP_ADD, OP_SUB, OP_MUL, OP_DIV = 0, 1, 2, 3


def _resolve_op_code(kwargs):
    op_code = kwargs.get("op_code")
    if op_code is None:
        attrs = kwargs.get("attributes") or {}
        if isinstance(attrs, dict):
            op_code = attrs.get("op_code")
    if op_code is None:
        op_code = OP_ADD
    return int(op_code)


def _int_binary(a, b, op_code, dt):
    # 2's-complement wrap for add/sub/mul via wide int -> narrow cast; div guarded + truncated.
    wide = np.int64
    aw = np.asarray(a).astype(dt).astype(wide)
    bw = np.asarray(b).astype(dt).astype(wide)
    if op_code == OP_ADD:
        return (aw + bw).astype(dt)
    if op_code == OP_SUB:
        return (aw - bw).astype(dt)
    if op_code == OP_MUL:
        return (aw * bw).astype(dt)
    # OP_DIV: b == 0 -> 0, else truncate toward zero (integer-only, exact, no float rounding).
    out = np.zeros_like(aw)
    nz = bw != 0
    q = (np.abs(aw[nz]) // np.abs(bw[nz])) * (np.sign(aw[nz]) * np.sign(bw[nz]))
    out[nz] = q
    return out.astype(dt)


def __golden_foreach_binary_op(x1_list, x2_list, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    op_code = _resolve_op_code(kwargs)

    results = []
    for i, (a, b) in enumerate(zip(x1_list, x2_list)):
        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)

        is_int = (target != "bfloat16") and np.issubdtype(np.dtype(target), np.integer)

        if is_int:
            out = _int_binary(a, b, op_code, np.dtype(target))
        else:
            a32 = np.asarray(a).astype(np.float32)
            b32 = np.asarray(b).astype(np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                if op_code == OP_ADD:
                    out = a32 + b32
                elif op_code == OP_SUB:
                    out = a32 - b32
                elif op_code == OP_MUL:
                    out = a32 * b32
                else:  # OP_DIV: IEEE inf/nan on b == 0, matching the NPU float path
                    out = a32 / b32
            if target == "bfloat16":
                out = out.astype(_bf16) if _bf16 is not None else out
            else:
                out = out.astype(target)
        results.append(out)
    return results


__golden__ = {"kernel": {"foreach_binary_op": "__golden_foreach_binary_op"}}
