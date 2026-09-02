#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK golden for foreach_a_cos_inplace: 竞品 torch._foreach_acos。
读法: 命名参数 x_list 直接迭代(同 add_list/div, 不用 *input_arrays/split)。
bf16 输入升 fp32 算(torch 不收 bf16, 同 kernel "bf16 computes in float")、输出 cast 回 bf16 对齐 NPU 的 bf16 输出。"""

import numpy as np
import torch

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
except ImportError:
    _BF16 = None


def _to_fp32(a):
    a = np.asarray(a)
    if (
        a.dtype.kind == "V" and _BF16 is not None
    ):  # bf16 按 |V2 原始字节传 -> view 成 bf16
        a = a.view(_BF16)
    # dtype 已匹配时复用原缓冲: 大张量(单份 GB 级)无谓复制会把进程推向 OOM。
    # 下游 torch 算子均非原地，不会改写输入，故复用安全。
    return a.astype(np.float32, copy=False)


def __golden_foreach_a_cos_inplace(x_list, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    tensors = [torch.from_numpy(_to_fp32(a)) for a in x_list]  # 竞品原生 foreach acos
    outs = torch._foreach_acos(tensors)
    if output_dtypes and isinstance(output_dtypes[0], (list, tuple)):
        dt_flat = list(output_dtypes[0])
    else:
        dt_flat = list(output_dtypes or [])
    results = []
    for i, t in enumerate(outs):
        r = t.numpy()
        tgt = dt_flat[i] if i < len(dt_flat) else "float32"
        results.append(r.astype(_BF16) if tgt == "bfloat16" else r.astype(tgt))
    return results


__golden__ = {"kernel": {"foreach_a_cos_inplace": "__golden_foreach_a_cos_inplace"}}

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


_GOLDEN_FN = __golden_foreach_a_cos_inplace


class _ForeachACosInplaceCompose:
    def __call__(self, x, **kwargs):
        return torch._foreach_acos(_tp_list(x))


class ForeachACosInplaceKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ForeachACosInplaceCompose}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_a_cos_inplace": "ForeachACosInplaceKernelSpec",
    "aclnnForeachACosInplace": "ForeachACosInplaceAclnnSpec",
}


def _tp_one(t):
    """aclnn 通路: 框架传入的已是设备侧 torch.Tensor, 不经 numpy。"""
    t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


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


class ForeachACosInplaceAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachACosInplaceGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 复用 kernel 通路的竞品类
    ——其形参名即 def 注册名, 与头文件一致。"""

    @staticmethod
    def golden(x, **kwargs):
        return _keep_dtype(torch._foreach_acos([_tp_one(t) for t in x]), x)

    third_party = {"torch": _ForeachACosInplaceCompose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付——算子目录下无 framework/ 插件、
# 无 graph pass, 也未发现 torch_npu eager/aten 绑定到该 aclnn 接口。
