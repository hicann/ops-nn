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

"""TTK custom golden plugin for scatter_mul (kernel mode).

Compute formula (docs/aclnnScatterMul.md, 计算公式节):
    varRef[indices[i], ...] = varRef[indices[i], ...] * updates[i, ...]
若多个 updates 作用到同一切片，则在该切片上连乘。
索引越界（idx < 0 或 idx >= var.shape[0]）按 kernel 语义跳过。

实现说明: 计算由 numpy 逐条连乘改为 torch 竞品算子 Tensor.index_reduce_(reduce="prod",
include_self=True)——纯 numpy 公式实现与被测 kernel 易犯同类错误，会掩盖 kernel 精度短板。
numpy 仅保留 I/O 与 dtype 转换。语义（重复索引连乘、越界跳过、累加 dtype、返回结构）未变。
"""

import numpy as np
import torch

_NP_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "int32": np.int32,
    "int8": np.int8,
    "uint8": np.uint8,
}


def __golden_scatter_mul(*input_arrays, **kwargs):
    # input order matches CSV input_shapes: var, indices, updates
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    # dtype 规则(两档分开):
    #  * 浮点: **完全不 cast**, 由 TTK 的 golden_mode=Promote 保障(cross_check 时 fp16/bf16
    #    ->fp32、fp32->fp64 下发), 出口也不窄回, TTK 负责。自行 cast 会撤销 Promote, 真值塌
    #    到与三方腿同精度, 三比值分母被夹, RMSE 比值假红。bf16 由 TTK 自行桥接(numpy 无原生
    #    bf16), 与 Promote 正交, golden 无需处理。
    #  * 整型: 没有三方腿(判据 binary_equal -> need_3party=False), TTK 也从不 Promote 整型,
    #    故按 **NPU 的实现逻辑** 决定是否 cast —— 内核 AccT 对 int32 原生, 对 int8/uint8 经
    #    SubwordWidenToI32 提到 int32 做整条链, 末尾 NarrowStore 窄回, golden 同步复刻。
    # 计算 dtype 一律**跟随 NPU 的 AccT**, 唯一例外是"浮点 + 三方": 那时 TTK 已按
    # golden_mode=Promote 把入参抬档下发(fp16/bf16->fp32、fp32->fp64), golden 零 cast 直接算
    # 即为高精度真值。其余情形(整型恒是两方; 浮点在两方泛化下) TTK 不提升, 必须由 golden
    # 自己复刻内核: AccT 对 fp16/bf16 是 float、对 int8/uint8 是 int32、对 fp32/int32 原生,
    # 出口再复刻 NarrowStore 窄回。判 Promote 用 TTK 下发的 golden_mode, 不靠 dtype 猜。
    promoted = kwargs.get("golden_mode") == "Promote"
    sub_int = np.issubdtype(var.dtype, np.integer) and var.dtype.itemsize < 4
    narrow_fp = (not promoted) and var.dtype == np.float16
    acc_dtype = np.int32 if sub_int else (np.float32 if narrow_fp else var.dtype)
    need_narrow = acc_dtype != var.dtype

    result = var.astype(acc_dtype, copy=True)
    upd = updates.astype(acc_dtype, copy=False)

    var_first = result.shape[0] if result.ndim >= 1 else 1
    idx_flat = indices.reshape(-1).astype(np.int64)
    # updates leading dims correspond to indices entries; trailing = var.shape[1:]
    n_idx = idx_flat.shape[0]
    slice_shape = result.shape[1:]
    upd_slices = (
        upd.reshape((n_idx,) + tuple(slice_shape))
        if n_idx > 0
        else upd.reshape((0,) + tuple(slice_shape))
    )

    # out-of-bound indices dropped first (scatter_reduce_common_simt.h:112), the rest
    # go through the torch reference op; include_self=True == var * all matching updates.
    result_t = torch.from_numpy(result)
    idx_t = torch.from_numpy(idx_flat)
    valid = (idx_t >= 0) & (idx_t < var_first)
    if int(valid.sum()) > 0:
        upd_t = torch.from_numpy(upd_slices)
        result_t.index_reduce_(0, idx_t[valid], upd_t[valid], "prod", include_self=True)

    # 出口不 cast —— TTK 负责窄回。自行 astype 会把 Promote 出来的高精度真值砍回去。
    out = result_t.numpy()
    if need_narrow:
        out = out.astype(var.dtype)  # 复刻 NarrowStore
    return [out]


def __golden_scatter_mul_e2e(var, indices, updates, use_locking=None, **kwargs):
    """e2e(TF 前端)通路 golden: 入参为框架张量, 还原为 numpy 后复用 kernel 档实现。"""

    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    return __golden_scatter_mul(_np(var), _np(indices), _np(updates))


__golden__ = {
    "kernel": {"scatter_mul": "__golden_scatter_mul"},
    "e2e": {"tf.compat.v1.scatter_mul": "__golden_scatter_mul_e2e"},
}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 保留原 golden，补三方标杆与自定义输入。
# third_party 用 torch 竞品算子在设备侧跑，供 cross_check 比对；
# customize_inputs 即原 input.py 的合法索引重采样（原文件保留，不影响旧机制）。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
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
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备(GPU)。
    这里只做载体还原/拷贝, 精度由 _tp_widen/_tp_narrow 按内核算法统一处理。"""
    if isinstance(x, torch.Tensor):
        return x.clone()
    return torch.as_tensor(np.asarray(x)).clone()  # 仅本地自测兜底: 框架侧不会走到


def _tp_widen(t):
    """加宽到**内核的累加类型**, 复刻 scatter_reduce_common_simt.h 的 LoadWiden
    (该文件注释: "ACC: float for fp16, native for fp32/int32, int32 for int8/uint8",
    "load srcGm into accUb, widening subword/fp16 to the ACC type")。

    本算子对重复索引是链式规约, 内核**整条链都在 fp32 accUb 上做**, 只在 NarrowStore
    时窄一次。三方若在 fp16 上逐步截断, 就不是同一个算法: 实测 A100 上中间量会溢出成
    inf(链长 257 的用例 13 个位置), 而内核算得出有限值。整型不涉及: 整型判据是 binary_equal, need_3party=False, TTK 不执行 GPU 腿。
    """
    if t.dtype in (torch.float16, torch.bfloat16):
        return t.float()
    if t.dtype in (torch.int8, torch.uint8):
        return t.to(
            torch.int32
        )  # 内核 AccT 对 1 字节类型提升到 int32(SubwordWidenToI32)
    return t


def _tp_narrow(outs, dt):
    """复刻 NarrowStore: 算完窄回算子输出 dtype。**必须与 _tp_widen 成对**——
    少了它三方停在 fp32, 与走 Promote(fp16->fp32) 的 golden 逐位相等, 双标杆塌成单标杆,
    三比值分母夹到 §4.5.1 的 err, 有量纲的 RMSE 比值随输出量级放大而假红
    (真机实测: 不窄回 rmse 比值 2348.6, 窄回后 1.0000)。"""
    # 整型也要窄回: torch 在整型路径上会把中间量升到 int32, 只窄浮点会让 int8/uint8
    # 出参漏成 int32。三方腿出口必须与算子 dtype 一致(复刻内核 NarrowStore)。
    return [o.to(dt) for o in outs]


def scatter_mul_input(var, indices, updates, **kwargs):
    """
    Input function for scatter_mul.
    All the parameters (names and order) follow scatter_mul_def.cpp without outputs.
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


class _ScatterMulCompose:
    def __call__(self, var, indices, updates, **kwargs):
        work0 = _tp_t(var)
        if _scatter_noop(work0, updates):
            return [work0]
        _dt = work0.dtype
        work = _tp_widen(work0)
        upd = _tp_widen(_tp_t(updates)).reshape((-1,) + tuple(work.shape[1:]))
        it = (
            indices
            if isinstance(indices, torch.Tensor)
            else torch.as_tensor(np.asarray(indices))
        )
        idx = it.reshape(-1).to(torch.int64)
        valid = (idx >= 0) & (idx < work.shape[0])
        idx = idx[valid]
        if idx.numel() > 0:
            work = work.index_reduce(0, idx, upd[valid], "prod", include_self=True)
        return _tp_narrow([work], _dt)


class _ScatterMulTfCompose:
    """tf 腿适配类。TF 侧对标算子是 ScatterMul(tf_plugin 的 OriginOpType), 但它是
    **ref-variable 语义**: tf.raw_ops.ScatterMul 首参名叫 ref 且必须是可变引用, 与 def.cpp
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
            tf.compat.v1.scatter_mul(
                ref, tf.cast(idx, tf.int32), tf.cast(upd, ref.dtype)
            )
        out = tf.convert_to_tensor(ref)
        return [
            tf.cast(out, _tf_dt) if out.dtype != _tf_dt else out
        ]  # 复刻 NarrowStore


_GOLDEN_FN = __golden_scatter_mul


class ScatterMulKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _ScatterMulCompose, "tf": _ScatterMulTfCompose}
    customize_inputs = scatter_mul_input
    tolerance = _TOL_KERNEL


# e2e(TF 前端)通路。不声明则判据回落 TTK 默认 mix_tolerance —— 其 max_abs_error
# 绝对硬上限(fp16 为 1e-1)对大值域不适配: 数据量级达 65504 时 1 ULP 即 32, 末位
# 差异必然超限假红。与 kernel 腿同口径走 cross_check L1。
_TOL_E2E = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


class _TpE2eMul:
    """e2e 通路三方腿适配: 池的 key 取自 TF API 形参名(ref/indices/updates), 与 def
    注册名(var/...)不同 —— 直接复用 kernel 腿竞品类会报 "parameter 'var' is not a
    known input or attribute name", 三方腿整条起不来。按 TF 形参名另立适配类, 内部
    按位置转调同一竞品类, 不改变竞品语义。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterMulCompose()(ref, indices, updates)


class _TpE2eMulTf:
    """同上, tf provider 腿。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterMulTfCompose()(ref, indices, updates)


class ScatterMulE2eSpec:
    """e2e 通路 spec: 三方腿与判据。"""

    third_party = {"torch": _TpE2eMul, "tf": _TpE2eMulTf}
    tolerance = _TOL_E2E


__spec__ = {
    "scatter_mul": "ScatterMulKernelSpec",
    "tf.compat.v1.scatter_mul": "ScatterMulE2eSpec",
    "aclnnScatterMul": "ScatterMulAclnnSpec",
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


class ScatterMulAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterMulGetWorkspaceSize 的
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
            work = work.index_reduce(0, idx, upd[valid], "prod", include_self=True)
        return _keep_dtype([work], varRef)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterMulCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# TensorFlow: 算子目录下有 framework 的 tf_plugin(OriginOpType "ScatterMul")。
#   三方标杆(精度/性能)已补: third_party["tf"] = _ScatterMulTfCompose, 跑 --provider tf。
#   通路连通(ⓐ)未注册: TTK 的 tf 通路是 e2e 前端(api_name 写 TF API), NPU 侧需要
#   Ascend TF adapter(npu_device/tfplugin)才能把 TF 图下沉到本算子; 当前环境
#   (cann-9.2.0)未装该组件, 装不上就跑不出 invoke_path 证据, 故不注册空壳键
#   (规范: __spec__ 注册集合必须等于 01 §3.3 的 ✅ 集合与 invoke_path 的通路取值)。
#   该组件到位前, tf 通路连通性仍按预生成 .pb + aclgrphParseTensorFlow 验证。
# e2e / ONNX / 融合 pass: 均未交付。
