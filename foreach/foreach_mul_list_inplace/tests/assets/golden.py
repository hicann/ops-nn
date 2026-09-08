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

"""TTK golden for foreach_mul_list_inplace: 竞品 torch._foreach_mul (x1_i *= x2_i)。
读法: 命名参数 x1_list/x2_list 直接迭代(原 *input_arrays+n_total//2 会把整个list当1个张量, 错)。
bf16 升 fp32 算、输出 cast 回 bf16; int 保持原 dtype。"""

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
except ImportError:
    _BF16 = None


def _to_compute(a):
    """bf16/void 升 fp32(torch 不收 bf16); 其它(int/fp16/fp32)保持。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _BF16 is not None:
        a = a.view(_BF16).astype(np.float32)
    return a


def __golden_foreach_mul_list_inplace(x1_list, x2_list, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes and isinstance(output_dtypes[0], (list, tuple)):
        dt_flat = list(output_dtypes[0])
    else:
        dt_flat = list(output_dtypes or [])
    results = []
    for i, (a, b) in enumerate(zip(x1_list, x2_list)):
        a = np.asarray(a)
        b = np.asarray(b)
        tgt = dt_flat[i] if i < len(dt_flat) else str(a.dtype)
        if np.issubdtype(a.dtype, np.integer):
            res = (a.astype(np.int64) * b.astype(np.int64)).astype(tgt)
        else:
            ta = torch.from_numpy(_to_compute(a))
            tb = torch.from_numpy(_to_compute(b))
            res = (ta * tb).numpy().astype(_BF16 if tgt == "bfloat16" else tgt)
        results.append(res)
    return results


__golden__ = {
    "kernel": {"foreach_mul_list_inplace": "__golden_foreach_mul_list_inplace"}
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


try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def _tp_bf16_carrier(a):
    """bf16 用 ml_dtypes 承载, torch 不认其 void 载体, 按位 view 回 bf16(同宽无损)。
    这是**载体还原**, 不是精度干预。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    if _bf16 is not None and a.dtype == _bf16:
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(a)


def _tp_list(xs):
    """third_party 入参(TensorList): 只做**载体还原**(bf16 用 ml_dtypes 承载, torch 不认其
    void 视图, 按位 view 回来, 同宽无损), 不在这里干预精度。

    ⚠️ 不能嵌套调用 golden 侧的 _to_compute/_to_fp32: 它们会把 bf16 抬成 fp32, 使三方腿
    与走 Promote(bf16->fp32) 的 golden 逐位相等 —— 双标杆塌成单标杆, 三比值分母被夹到
    精度标准 §4.5.1 的 err, 有量纲的 RMSE 比值随输出量级线性放大而假红。
    整型同理必须整型进整型出, 走一趟 fp32 会把 >2^24 的 int32 抹掉低位(见 #85 的成因)。
    计算精度统一由 _kf_widen/_kf_narrow 按内核算法处理。
    """
    return [
        a if isinstance(a, torch.Tensor) else _tp_bf16_carrier(np.asarray(a))
        for a in xs
    ]


def _tp_scalar(x, ref=None):
    """三方标量: **不转 Python float**(那是 fp64, 会触发 torch 类型提升把整条链抬档);
    整型标量过 float 还会把 >2^24 的 int32 抹掉低位。

    ⚠️ 必须落在 **CPU**: torch._foreach_* 的 scalars 形参要求在 CPU 上, 三方腿在 GPU 上跑,
    标量跟着上 cuda 会报 "Expected scalars to be on CPU, got cuda:0"。
    """
    t = x if isinstance(x, torch.Tensor) else _tp_bf16_carrier(np.asarray(x))
    if ref is not None:
        rd = (
            ref.dtype
            if isinstance(ref, torch.Tensor)
            else torch.as_tensor(np.asarray(ref)).dtype
        )
        t = t.to(rd)
    return t.reshape(-1).cpu()


_GOLDEN_FN = __golden_foreach_mul_list_inplace


class _ForeachMulListInplaceCompose:
    def __call__(self, x1, x2, **kwargs):
        return torch._foreach_mul(_tp_list(x1), _tp_list(x2))


# ---------------------------------------------------------------------------
# 三方(GPU)腿按内核算法转写: 入口复刻 Cast<U, T, 0>(窄类型加宽到内核计算类型),
# 出口复刻 Cast<T, U, 1> / NarrowStore(窄回算子输出 dtype)。两者**必须成对**:
#   * 缺入口加宽 -> 三方在窄类型上逐步截断, 与融合内核(中间量全程 fp32)不是同一算法;
#     A100 实测 fp16 下 300*300 直接得 inf, 而 (a*a)/a 真值 300 明明存得下。
#   * 缺出口窄回 -> 三方停在 fp32, 与走 TTK Promote(fp16->fp32) 的 golden 逐位相等,
#     双标杆塌成单标杆, 三比值分母夹到精度标准 §4.5.1 的 err, 有量纲的 RMSE 比值随
#     输出量级线性放大而假红(真机实测: 不窄回 rmse 比值 2348.6, 窄回后 1.0000)。
# 整型不动: 内核对 int 也是原生/int32 累加, 走一趟 fp32 会把 >2^24 抹掉低位。
# 只作用于 third_party(GPU); CPU golden 由 TTK 的 Promote 单独喂高精度输入, 不受影响。
# ---------------------------------------------------------------------------


def _kf_widen(a, seen):
    if isinstance(a, (list, tuple)):
        return type(a)(_kf_widen(x, seen) for x in a)
    if isinstance(a, torch.Tensor):
        if a.dtype in (torch.float16, torch.bfloat16):
            seen.append(a.dtype)
            return a.float()
        return a
    if isinstance(a, np.ndarray) or (_KF_BF16 is not None and hasattr(a, "dtype")):
        n = np.asarray(a)
        if n.dtype.kind == "V" and _KF_BF16 is not None:
            n = n.view(_KF_BF16)
        if _KF_BF16 is not None and n.dtype == _KF_BF16:
            seen.append(torch.bfloat16)
            return n.astype(np.float32)
        if n.dtype == np.float16:
            seen.append(torch.float16)
            return n.astype(np.float32)
        return a
    return a


def _kf_narrow(o, dt):
    if isinstance(o, (list, tuple)):
        return type(o)(_kf_narrow(x, dt) for x in o)
    if isinstance(o, torch.Tensor) and o.is_floating_point():
        return o.to(dt)
    return o


class _TpKernelFaithful:
    _INNER = _ForeachMulListInplaceCompose

    def __call__(self, *args, **kwargs):
        seen = []
        wa = [_kf_widen(a, seen) for a in args]
        wk = {k: _kf_widen(v, seen) for k, v in kwargs.items()}
        outs = self._INNER()(*wa, **wk)
        return outs if not seen else _kf_narrow(outs, seen[0])


# TTK 服务端按 __call__ 的**签名**做入参绑定(remote/server/executor.py::_bind ->
# _function_param_names)。包装类若只写 *args/**kwargs, 服务端取不到形参名, 会退化成
# 位置传参、同时又传同名关键字 -> "got multiple values for argument ..." 而三方腿整个不可用。
# 故把内层 compose 的签名透传出去。
try:
    _TpKernelFaithful.__call__.__signature__ = _kf_inspect.signature(
        _ForeachMulListInplaceCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class ForeachMulListInplaceKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_mul_list_inplace": "ForeachMulListInplaceKernelSpec",
    "aclnnForeachMulListInplace": "ForeachMulListInplaceAclnnSpec",
}


def _tp_one(t):
    """aclnn 通路三方腿入参: **不替 torch 决定精度**, 原样交给它(口径同 _tp_list)。

    aclnn 通路同样取用 third_party(按名绑定), 见下方 _TpAclnn。
    """
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)


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


class _TpAclnn:
    """aclnn 通路三方腿适配: 头文件形参名与 kernel 通路的 def 注册名**不同**。

    inplace 算子的 aclnn 头文件把首个(被原地改写的)形参写作 `x1Ref`, 而 def 注册名是
    `x1`。服务端对 third_party 按名绑定(remote/server/execution_container.py::
    bind_params), pool 的 key 取自头文件形参名; 直接复用 kernel 通路的竞品类会因
    形参 `x1` 不在 pool 中而抛 UnknownParamError, 三方腿整条起不来, cross_check
    随即 GOLDEN_FAILURE。故按头文件形参名另立适配类, 内部转调同一个竞品类。
    """

    def __call__(self, x1Ref, x2, **kwargs):
        return _TpKernelFaithful()(x1Ref, x2)


class ForeachMulListInplaceAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachMulListInplaceGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 因该名与 def 注册名不同,
    另由 _TpAclnn 适配后转调同一个竞品类。"""

    @staticmethod
    def golden(x1, x2, **kwargs):
        return _keep_dtype(
            torch._foreach_mul([_tp_one(t) for t in x1], [_tp_one(t) for t in x2]), x1
        )

    third_party = {"torch": _TpAclnn}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付——算子目录下无 framework/ 插件、
# 无 graph pass, 也未发现 torch_npu eager/aten 绑定到该 aclnn 接口。
