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

计算实现改用竞品 torch._foreach_* (红线 R3: golden 只能是竞品接口实现或竞品算子拼接实现,
禁 numpy 纯公式), numpy 仅保留 I/O 与 dtype 转换; 数值与改造前逐位一致。整型除法用
torch.div(rounding_mode="trunc") 并保留 b == 0 -> 0 的设备守卫(torch 整除 0 会抛异常)。
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

OP_ADD, OP_SUB, OP_MUL, OP_DIV = 0, 1, 2, 3


def _per_tensor_dtypes(od):
    """把 output_dtypes 拍平成 per-tensor 列表。

    TTK 归一化后，TensorList 输出的 output_dtypes 是"按输出分组"的嵌套形式
    (( 'float32', ... ) ,)；单 tensor 输出则是扁平的。老用例集写成扁平 N 项，
    当前 TTK 会以 CASE_FIELD_AMBIGUOUS 拒收。两种形式都要能收，否则
    output_dtypes[i] 取到的是元组，np.dtype(tuple) 直接抛 GOLDEN_FAILURE。
    """
    if od is None:
        return None
    flat = []
    for e in od:
        if isinstance(e, (tuple, list)):
            flat.extend(e)
        else:
            flat.append(e)
    return flat


def _to_fp32(a):
    """bf16 按 |V2 原始字节传入时先 view 回 bf16(torch 不收 bf16), 再升 fp32 计算。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    # dtype 已匹配时复用原缓冲: 大张量(单份 GB 级)无谓复制会把进程推向 OOM。
    # 下游 torch 算子均非原地，不会改写输入，故复用安全。
    return a.astype(np.float32, copy=False)


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
    # 宽中间量走 torch int64, 窄化回 dt 由 torch 的 C 截断完成(与 numpy 窄化同语义)。
    narrow = torch.from_numpy(np.empty(0, dtype=dt)).dtype
    aw = torch.from_numpy(np.asarray(a).astype(dt)).to(torch.int64)
    bw = torch.from_numpy(np.asarray(b).astype(dt)).to(torch.int64)
    if op_code == OP_ADD:
        return torch._foreach_add([aw], [bw])[0].to(narrow).numpy()
    if op_code == OP_SUB:
        return torch._foreach_sub([aw], [bw])[0].to(narrow).numpy()
    if op_code == OP_MUL:
        return torch._foreach_mul([aw], [bw])[0].to(narrow).numpy()
    # OP_DIV: b == 0 -> 0, else truncate toward zero (integer-only, exact, no float rounding).
    zero = bw == 0
    safe = torch.where(zero, torch.ones_like(bw), bw)
    q = torch.div(aw, safe, rounding_mode="trunc")
    return torch.where(zero, torch.zeros_like(q), q).to(narrow).numpy()


def __golden_foreach_binary_op(x1_list, x2_list, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
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
            ta = torch.from_numpy(_to_fp32(a))
            tb = torch.from_numpy(_to_fp32(b))
            if op_code == OP_ADD:
                out = torch._foreach_add([ta], [tb])[0]
            elif op_code == OP_SUB:
                out = torch._foreach_sub([ta], [tb])[0]
            elif op_code == OP_MUL:
                out = torch._foreach_mul([ta], [tb])[0]
            else:  # OP_DIV: IEEE inf/nan on b == 0, matching the NPU float path
                out = torch._foreach_div([ta], [tb])[0]
            out = out.numpy()
            if target == "bfloat16":
                out = out.astype(_bf16) if _bf16 is not None else out
            else:
                out = out.astype(target)
        results.append(out)
    return results


__golden__ = {"kernel": {"foreach_binary_op": "__golden_foreach_binary_op"}}
