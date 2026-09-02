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
TTK custom golden for foreach_sub_scalar_inplace.

Compute formula (docs/aclnnForeachSubScalarInplace.md):
    x_i = x_i - scalar   (i = 0, 1, ..., n-1)

Inplace output: x list (n sub-tensors). Output order == x sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened):
    x_list : list of numpy arrays   (the DYNAMIC TensorList x)
    scalar : numpy array, shape (1,)  (scalar subtrahend)

Mirrors the proven foreach_mul_scalar_inplace golden (200/200), with - instead of *.

计算实现改用竞品 torch._foreach_sub (红线 R3: golden 只能是竞品接口实现或竞品算子拼接
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


def __golden_foreach_sub_scalar_inplace(x_list, scalar, **kwargs):
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
            # NPU int path: pure 2's-complement wraparound on x-scalar -- NO saturation.
            # numpy float->int cast maps overflow/NaN to INT_MIN (diverges from NPU wrap),
            # so keep pure int. Same fix as foreach_mul_scalar_inplace golden.
            # 回绕靠 int64 中间量 + 窄化回 dt 实现(torch 的窄化是 C 截断, 与 numpy 同)。
            dt = np.dtype(target)
            a_i = torch.from_numpy(np.asarray(a).astype(dt))
            scalar_i = int(np.asarray(scalar).astype(dt).reshape(-1)[0])
            diff = torch._foreach_sub([a_i.to(torch.int64)], scalar_i)[0]
            out = diff.to(a_i.dtype).numpy()
        else:
            ta = torch.from_numpy(_to_fp32(a))
            out32 = torch._foreach_sub([ta], scalar_val)[0].numpy()
            if target == "bfloat16":
                out = out32.astype(_bf16) if _bf16 is not None else out32
            else:
                out = out32.astype(target)
        results.append(out)
    return results


__golden__ = {
    "kernel": {"foreach_sub_scalar_inplace": "__golden_foreach_sub_scalar_inplace"}
}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 在保留原 golden 的基础上补三方标杆能力。
# third_party 直接对标 torch 的 _foreach_* 竞品 API，在设备侧跑，供 cross_check 比对。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _tp_list(xs):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。"""
    out = []
    for a in xs:
        t = a if isinstance(a, torch.Tensor) else torch.as_tensor(_to_fp32(a))
        out.append(t.to(torch.float32))
    return out


def _tp_scalar(x):
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x, "float32"))
    return float(t.reshape(-1)[0])


_GOLDEN_FN = __golden_foreach_sub_scalar_inplace


class _ForeachSubScalarInplaceCompose:
    def __call__(self, x, scalar, **kwargs):
        return torch._foreach_sub(_tp_list(x), _tp_scalar(scalar))


class ForeachSubScalarInplaceKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ForeachSubScalarInplaceCompose}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_sub_scalar_inplace": "ForeachSubScalarInplaceKernelSpec",
    "aclnnForeachSubScalarInplace": "ForeachSubScalarInplaceAclnnSpec",
}


def _tp_one(t):
    """aclnn 通路: 框架传入的已是设备侧 torch.Tensor, 不经 numpy。"""
    t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


def _tp_num(v):
    if isinstance(v, torch.Tensor):
        return float(v.reshape(-1)[0])
    return float(v)


def _keep_dtype(res, ref):
    """golden 输出 dtype 必须与算子输出一致: 比对按 dtype 判定, fp16/bf16 提到 fp32
    算完必须还原, 否则 binary_equal 直接判 "dtype 不可比"(实测 GOLD 0%)。
    golden_mode=Promote 时入参本身已是 fp32, 此处是恒等操作。"""
    refs = ref if isinstance(ref, (list, tuple)) else [ref] * len(res)
    return [
        t.to(r.dtype)
        if isinstance(t, torch.Tensor) and isinstance(r, torch.Tensor)
        else t
        for t, r in zip(res, refs)
    ]


class ForeachSubScalarInplaceAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachSubScalarInplaceGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 复用 kernel 通路的竞品类
    ——其形参名即 def 注册名, 与头文件一致。"""

    @staticmethod
    def golden(x, scalar, **kwargs):
        return _keep_dtype(
            torch._foreach_sub([_tp_one(t) for t in x], _tp_num(scalar)), x
        )

    third_party = {"torch": _ForeachSubScalarInplaceCompose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付——算子目录下无 framework/ 插件、
# 无 graph pass, 也未发现 torch_npu eager/aten 绑定到该 aclnn 接口。
