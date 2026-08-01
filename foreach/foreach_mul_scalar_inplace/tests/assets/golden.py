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
TTK custom golden for foreach_mul_scalar_inplace.

Compute formula (docs/aclnnForeachMulScalarInplace.md:18-26):
    x_i = x_i * scalar   (i = 0, 1, ..., n-1)

Inplace output: x list (n sub-tensors). Output order == x sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened):
    x_list : list of numpy arrays   (the DYNAMIC TensorList x)
    scalar : numpy array, shape (1,)  (scalar multiplier)

计算实现改用竞品 torch._foreach_mul (红线 R3: golden 只能是竞品接口实现或竞品算子拼接
实现, 禁 numpy 纯公式), numpy 仅保留 I/O 与 dtype 转换; 数值与改造前逐位一致。
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


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


def __golden_foreach_mul_scalar_inplace(x_list, scalar, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))

    # scalar is a 1-element tensor; take the scalar value in float32 for stable compute.
    scalar_val = torch.from_numpy(
        np.asarray(scalar).astype(np.float32).reshape(-1)[:1]
    )[0]

    results = []
    for i, a in enumerate(x_list):
        # Cast back to the per-output dtype declared in the CSV so the golden carries
        # the same rounding semantics as the NPU output.
        if output_dtypes is not None and i < len(output_dtypes):
            target = str(output_dtypes[i])
        else:
            target = str(np.asarray(a).dtype)

        is_int = target != "bfloat16" and np.issubdtype(np.dtype(target), np.integer)

        if is_int:
            # NPU int path: pure 2's-complement wraparound on scalar*x -- NO saturation.
            # numpy float->int cast maps overflow/NaN to INT_MIN (diverges from NPU wrap),
            # so keep pure int (no float32). Same fix as foreach_add_list_inplace golden
            # (case00020_extreme: pure-wrap 100% vs saturate/float-cast 0%).
            # 回绕靠 int64 中间量 + 窄化回 dt 实现(torch 的窄化是 C 截断, 与 numpy 同)。
            dt = np.dtype(target)
            a_i = torch.from_numpy(np.asarray(a).astype(dt))
            scalar_i = int(np.asarray(scalar).astype(dt).reshape(-1)[0])
            prod = torch._foreach_mul([a_i.to(torch.int64)], scalar_i)[0]
            out = prod.to(a_i.dtype).numpy()
        else:
            ta = torch.from_numpy(_to_fp32(a))
            out32 = torch._foreach_mul([ta], scalar_val)[0].numpy()
            if target == "bfloat16":
                out = out32.astype(_bf16) if _bf16 is not None else out32
            else:
                out = out32.astype(target)
        results.append(out)
    return results


__golden__ = {
    "kernel": {"foreach_mul_scalar_inplace": "__golden_foreach_mul_scalar_inplace"}
}
