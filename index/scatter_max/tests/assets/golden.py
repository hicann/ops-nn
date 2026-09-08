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

"""scatter_max TTK 自定义 golden plugin（kernel 模式）。

计算公式（aclnnScatterMax.md 计算公式节 / proto.h:36）:
  varRef[indices[i], ...] = max(varRef[indices[i], ...], updates[i, ...])
  - 多个 updates 作用到 var 同一切片时依次取最大值（顺序无关）。
  - shape 约束: updates.shape = indices.shape + var.shape[1:]。
  - indices 越界值 (idx < 0 或 idx >= var.shape[0]) 被算子跳过（kernel simt.h:112）。
  - var 原地更新，输出即更新后的 var。

输入顺序: (var, indices, updates)  输出: [var]

实现说明: 计算由 numpy 逐条 np.maximum 改为 torch 竞品算子 Tensor.index_reduce_(reduce="amax",
include_self=True)——纯 numpy 公式实现与被测 kernel 易犯同类错误, 会掩盖 kernel 精度短板。
numpy 仅保留 I/O 与 dtype 转换。语义(重复索引取最大、越界跳过、工作 dtype、返回结构) 未变。
"""

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None


def __golden_scatter_max(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out = var.copy()
    var_first_dim = out.shape[0] if out.ndim >= 1 else 0

    idx_flat = indices.reshape(-1)
    n_idx = idx_flat.shape[0]

    if n_idx == 0 or var_first_dim == 0 or updates.size == 0:
        return [out.astype(var.dtype)]

    # updates 展平为 (n_idx, *slice_shape)：slice_shape = var.shape[1:]
    slice_shape = out.shape[1:]
    upd = updates.reshape((n_idx,) + tuple(slice_shape))

    # 浮点用 float32 中间计算（fp16 numpy 不自动提升）；整型保持原类型精确比对。
    # 注意：max 不累加、永不溢出，整型绝不能转 float32——大 int32(>2^24) 经 float32
    # round-trip 会丢精度，导致与 kernel 的精确 int32 max 不符（曾误报 inf/大值用例失败）。
    is_float = np.issubdtype(var.dtype, np.floating)
    work_dtype = np.float32 if is_float else var.dtype
    work = torch.from_numpy(out.astype(work_dtype))
    upd_w = torch.from_numpy(upd.astype(work_dtype))

    # 越界索引先剔除（与 kernel 的 skip 一致），剩下的交给 torch 竞品算子；
    # include_self=True 即 max(var 原值, 命中该行的所有 updates)，重复索引顺序无关。
    idx_t = torch.from_numpy(idx_flat.astype(np.int64))
    valid = (idx_t >= 0) & (idx_t < var_first_dim)
    idx_t = idx_t[valid]
    if idx_t.numel() > 0:
        work.index_reduce_(0, idx_t, upd_w[valid], "amax", include_self=True)

    return [work.numpy().astype(var.dtype)]


__golden__ = {"kernel": {"scatter_max": "__golden_scatter_max"}}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 保留原 golden，补三方标杆与自定义输入。
# third_party 用 torch 竞品算子在设备侧跑，供 cross_check 比对；
# customize_inputs 即原 input.py 的合法索引重采样（原文件保留，不影响旧机制）。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "binary_equal"},
    "float16": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}


def _scatter_numel(t):
    """张量元素数; 对 torch / numpy / tf 的 eager 张量都成立(只读 shape)。"""
    shape = getattr(t, "shape", None)
    if shape is None:
        return None
    n = 1
    for d in tuple(shape):
        n *= int(d)
    return n


def _scatter_noop(var, updates):
    """空张量短路判据: var 或 updates 无元素时 scatter 是 no-op, 原样返回 var。

    必须在 updates 展平之前判。展平写的是 reshape((-1,) + var.shape[1:]), 当 var 的
    **非首维含 0**(切片宽度为 0)时, 0 个元素铺进 [-1, 0] 的 -1 无法唯一推断, torch 抛
    "cannot reshape tensor of 0 elements into shape [-1, 0] ... is ambiguous", tf 同理。
    合法入参下 updates 为空 <=> indices 为空(shape 约束 updates = indices + var[1:],
    且此时 var[1:] 全非 0), 两种情形都是 no-op, 故本判据既充分也不会误伤。

    kernel 通路那份 golden 早有这道短路(sliceSize==0 直接返回原 var), 是 aclnn spec
    与三方腿新增时漏带。算子侧同样是 no-op: scatter_reduce_common_tiling.cpp 对空张量
    按结构合法放行, scatter_reduce_common_simt.h 的 Init/Process 在 tiling_.sliceSize == 0
    时直接 return。
    """
    nv = _scatter_numel(var)
    nu = _scatter_numel(updates)
    return nv == 0 or nu == 0


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。"""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    # 不抬精度: 三方标杆按算子自身 dtype 计算。此前 fp16/bf16 统一升 fp32 会让三方与走
    # Promote(fp32) 的 golden 逐位相等, cross_check 分母塌到 safe_div 的 small_value 地板,
    # 判据退化成"NPU 与 fp32 参照的绝对误差", 随输出量级线性放大而必红。
    return t.clone()


def scatter_max_input(var, indices, updates, **kwargs):
    """
    Input function for scatter_max.
    All the parameters (names and order) follow scatter_max_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Default random indices may fall out of [0, var.shape[0]); resample them into
    the legal first-dim range so var[indices[i]] is always addressable (kernel
    silently skips out-of-range indices, but golden/kernel agree only on legal ones).

    Args:
        **kwargs: input_dtypes, full_soc_version, short_soc_version, testcase_name

    Returns:
        Input tensors
    """
    shape_indices, dtype_indices, size_indices = (
        indices.shape,
        indices.dtype,
        indices.size,
    )
    max_indices = var.shape[0]

    if var.size * indices.size * updates.size == 0:
        return [var, indices, updates]

    replace = size_indices > max_indices
    indices = np.random.choice(max_indices, size_indices, replace=replace).astype(
        dtype_indices
    )
    indices = np.reshape(indices, shape_indices)
    return [var, indices, updates]


class _ScatterMaxCompose:
    def __call__(self, var, indices, updates, **kwargs):
        work = _tp_t(var)
        if _scatter_noop(work, updates):
            return [work]
        upd = _tp_t(updates).reshape((-1,) + tuple(work.shape[1:]))
        it = (
            indices
            if isinstance(indices, torch.Tensor)
            else torch.as_tensor(np.asarray(indices))
        )
        idx = it.reshape(-1).to(torch.int64)
        valid = (idx >= 0) & (idx < work.shape[0])
        idx = idx[valid]
        if idx.numel() > 0:
            work = work.index_reduce(0, idx, upd[valid], "amax", include_self=True)
        return [work]


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
    _INNER = _ScatterMaxCompose

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
        _ScatterMaxCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class _ScatterMaxTfCompose:
    """tf 腿适配类。TF 侧对标算子是 ScatterMax(tf_plugin 的 OriginOpType), 但它是
    **ref-variable 语义**: tf.raw_ops.ScatterMax 首参名叫 ref 且必须是可变引用, 与 def.cpp
    的 var 既不同名、也不能直接喂 eager tensor, 故不能用 API 直调形式(那种写法只适用于
    参数名与 def.cpp 一致的普通算子), 要在这里把 var 包成 tf.Variable 再调。

    与 torch 腿的两处一致性(不一致就会系统性假红):
    1) 非法下标(越界/负)按算子语义静默跳过——TF 自己会抛 InvalidArgumentError;
    2) **按内核算法转写**: 内核对 fp16 是 LoadWiden 到 fp32 累加、NarrowStore 窄回
       (scatter_reduce_common_simt.h: "ACC: float for fp16"), 故此处同样先加宽再窄回,
       与 torch 腿同口径。只在窄类型上算会逐步截断/溢出, 与被测内核不是同一个算法。
    """

    def __call__(self, var, indices, updates, **kwargs):
        import tensorflow as tf

        work = tf.convert_to_tensor(var)
        if _scatter_noop(work, updates):
            return [work]
        _tf_dt = work.dtype  # 算子输出 dtype, 出口窄回用
        if _tf_dt in (tf.float16, tf.bfloat16):
            work = tf.cast(work, tf.float32)  # 复刻内核 LoadWiden
        upd = tf.reshape(
            tf.convert_to_tensor(updates), tf.concat([[-1], tf.shape(work)[1:]], axis=0)
        )
        idx = tf.reshape(tf.cast(tf.convert_to_tensor(indices), tf.int64), [-1])
        keep = tf.logical_and(idx >= 0, idx < tf.cast(tf.shape(work)[0], tf.int64))
        idx, upd = tf.boolean_mask(idx, keep), tf.boolean_mask(upd, keep)
        ref = tf.Variable(work)
        if tf.size(idx) > 0:
            tf.compat.v1.scatter_max(
                ref, tf.cast(idx, tf.int32), tf.cast(upd, ref.dtype)
            )
        out = tf.convert_to_tensor(ref)
        return [
            tf.cast(out, _tf_dt) if out.dtype != _tf_dt else out
        ]  # 复刻 NarrowStore


_GOLDEN_FN = __golden_scatter_max


class ScatterMaxKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful, "tf": _ScatterMaxTfCompose}
    customize_inputs = scatter_max_input
    tolerance = _TOL_KERNEL


__spec__ = {
    "scatter_max": "ScatterMaxKernelSpec",
    "aclnnScatterMax": "ScatterMaxAclnnSpec",
}


def _tp_one(t):
    """aclnn 通路三方腿入参: **不替 torch 决定精度**, 原样交给它。

    三方腿的输入 dtype 与 NPU 一致, torch 算完自然就是同一 dtype, 无需人为抬档或回 cast;
    是否在内部抬到 fp32 由 torch 的算子实现自行决定。此前无条件把 fp16/bf16 抬到 fp32,
    会让三方与走 Promote(fp32) 的 golden 逐位相等 —— 双标杆塌成单标杆, 三比值分母夹到
    §4.5.1 的 err, 有量纲的 RMSE 比值随输出量级线性放大而假红。

    【预留】TTK 的 aclnn 通路当前不取用 third_party(仅 kernel/GEIR 取用), 此处写法不生效
    也无副作用; 待该通路支持三方后自动接上, 口径与 kernel/GEIR 腿保持一致。
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


class ScatterMaxAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterMaxGetWorkspaceSize 的
    形参 varRef/indices/updates/useLocking;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 用适配类把头文件的
    varRef 接到 kernel 通路竞品类的 def 注册名 var 上。"""

    @staticmethod
    def golden(varRef, indices, updates, useLocking=None, **kwargs):
        work = _tp_one(varRef).clone()
        if _scatter_noop(work, updates):
            return _keep_dtype([work], varRef)
        upd = _tp_one(updates).reshape((-1,) + tuple(work.shape[1:]))
        it = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices)
        idx = it.reshape(-1).to(torch.int64)
        valid = (idx >= 0) & (idx < work.shape[0])
        idx = idx[valid]
        if idx.numel() > 0:
            work = work.index_reduce(0, idx, upd[valid], "amax", include_self=True)
        return _keep_dtype([work], varRef)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterMaxCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# TensorFlow: 算子目录下有 framework 的 tf_plugin(OriginOpType "ScatterMax")。
#   三方标杆(精度/性能)已补: third_party["tf"] = _ScatterMaxTfCompose, 跑 --provider tf。
#   通路连通(ⓐ)未注册: TTK 的 tf 通路是 e2e 前端(api_name 写 TF API), NPU 侧需要
#   Ascend TF adapter(npu_device/tfplugin)才能把 TF 图下沉到本算子; 当前环境
#   (cann-9.2.0)未装该组件, 装不上就跑不出 invoke_path 证据, 故不注册空壳键
#   (规范: __spec__ 注册集合必须等于 01 §3.3 的 ✅ 集合与 invoke_path 的通路取值)。
#   该组件到位前, tf 通路连通性仍按预生成 .pb + aclgrphParseTensorFlow 验证。
# e2e / ONNX / 融合 pass: 均未交付。
