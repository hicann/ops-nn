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

"""TTK golden for foreach_log_inplace: 竞品 torch._foreach_log。
读法: 命名参数 x_list 直接迭代(同 add_list/div/a_cos)。bf16 升 fp32 算、输出 cast 回 bf16。有效域 (0,inf)。"""

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


def _to_fp32(a):
    a = np.asarray(a)
    if a.dtype.kind == "V" and _BF16 is not None:
        a = a.view(_BF16)
    # dtype 已匹配时复用原缓冲: 大张量(单份 GB 级)无谓复制会把进程推向 OOM。
    # 下游 torch 算子均非原地，不会改写输入，故复用安全。
    # Promote 后的 float64 是高精度真值, **不能砍回 fp32**: 砍了就与三方腿(fp32)
    # 逐位相等, cross_check 三比值分母被夹到精度标准 §4.5.1 的 err ——
    # 无量纲的 mare/mere 恒等于地板值(实测 fp32 档 mare 恒为 0.0019531 = 1ULP/2^-14),
    # 有量纲的 RMSE 随输出量级放大而假红。fp32 档等于从未被真正验证。
    if a.dtype == np.float64:
        return a
    return a.astype(np.float32, copy=False)


def __golden_foreach_log_inplace(x_list, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    tensors = [torch.from_numpy(_to_fp32(a)) for a in x_list]
    outs = torch._foreach_log(tensors)
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


__golden__ = {"kernel": {"foreach_log_inplace": "__golden_foreach_log_inplace"}}

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


_GOLDEN_FN = __golden_foreach_log_inplace


class _ForeachLogInplaceCompose:
    def __call__(self, x, **kwargs):
        return torch._foreach_log(_tp_list(x))


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
    _INNER = _ForeachLogInplaceCompose

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
        _ForeachLogInplaceCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class ForeachLogInplaceKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_log_inplace": "ForeachLogInplaceKernelSpec",
    "aclnnForeachLogInplace": "ForeachLogInplaceAclnnSpec",
}


def _tp_one(t):
    """aclnn 通路三方腿入参: **不替 torch 决定精度**, 原样交给它(口径同 _tp_list)。

    【预留】TTK 的 aclnn 通路当前不取用 third_party(仅 kernel/GEIR 取用), 写在此处不生效
    也无副作用; 待该通路支持三方后自动接上。
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


class ForeachLogInplaceAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachLogInplaceGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 复用 kernel 通路的竞品类
    ——其形参名即 def 注册名, 与头文件一致。"""

    @staticmethod
    def golden(x, **kwargs):
        return _keep_dtype(torch._foreach_log([_tp_one(t) for t in x]), x)

    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付——算子目录下无 framework/ 插件、
# 无 graph pass, 也未发现 torch_npu eager/aten 绑定到该 aclnn 接口。
