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
TTK custom golden for foreach_sub_list_inplace.

Compute formula (docs/aclnnForeachSubListInplace.md):
    x1_i = x1_i - alpha * x2_i   (i = 0, 1, ..., n-1)

Inplace output: x1 list (n sub-tensors). Output order == x1 sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened):
    x1_list : list of numpy arrays  (the DYNAMIC TensorList x1)
    x2_list : list of numpy arrays  (the DYNAMIC TensorList x2)
    alpha   : numpy array, shape (1,)  (scalar coefficient)

Mirrors the proven foreach_add_list_inplace golden (200/200), with - instead of +.

计算实现改用竞品 torch._foreach_* (红线 R3: golden 只能是竞品接口实现或竞品算子拼接实现,
禁 numpy 纯公式), numpy 仅保留 I/O 与 dtype 转换。alpha 用 _foreach_mul + _foreach_sub
两步拼接而非 _foreach_sub(alpha=) 的 FMA 单次舍入形式, 与内核 Muls 再 Sub 的两步舍入
一致, 也与改造前的 golden 逐位一致。
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


def __golden_foreach_sub_list_inplace(x1_list, x2_list, alpha, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))

    # alpha is a 1-element tensor; take the scalar value in float32 for stable compute.
    alpha_val = torch.from_numpy(np.asarray(alpha).astype(np.float32).reshape(-1)[:1])[
        0
    ]

    results = []
    for i, (a, b) in enumerate(zip(x1_list, x2_list)):
        # Cast back to the per-output dtype declared in the CSV so the golden carries
        # the same rounding semantics as the NPU output.
        if output_dtypes is not None and i < len(output_dtypes):
            target = str(output_dtypes[i])
        else:
            target = str(np.asarray(a).dtype)

        is_int = target != "bfloat16" and np.issubdtype(np.dtype(target), np.integer)

        if is_int:
            # NPU int path: pure 2's-complement wraparound on BOTH alpha*x2 (mul) and
            # x1- (sub) -- NO saturation. numpy float->int cast maps overflow/NaN to
            # INT_MIN, which diverges from NPU wrap. Mirror add_list's verified pure-wrap.
            # 回绕靠 int64 中间量 + 窄化回 dt 实现(torch 的窄化是 C 截断, 与 numpy 同),
            # 乘、减各回绕一次的顺序保持不变。
            dt = np.dtype(target)
            a_i = torch.from_numpy(np.asarray(a).astype(dt))
            b_i = torch.from_numpy(np.asarray(b).astype(dt))
            alpha_i = int(np.asarray(alpha).astype(dt).reshape(-1)[0])
            prod = torch._foreach_mul([b_i.to(torch.int64)], alpha_i)[0].to(a_i.dtype)
            acc = torch._foreach_sub([a_i.to(torch.int64)], [prod.to(torch.int64)])[0]
            out = acc.to(a_i.dtype).numpy()
        else:
            ta = torch.from_numpy(_to_fp32(a))
            tb = torch.from_numpy(_to_fp32(b))
            scaled = torch._foreach_mul([tb], alpha_val)
            out32 = torch._foreach_sub([ta], scaled)[0].numpy()
            if target == "bfloat16":
                out = out32.astype(_bf16) if _bf16 is not None else out32
            else:
                out = out32.astype(target)
        results.append(out)
    return results


__golden__ = {
    "kernel": {"foreach_sub_list_inplace": "__golden_foreach_sub_list_inplace"}
}
