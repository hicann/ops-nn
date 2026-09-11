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

"""TTK custom golden for ScatterDiv (kernel mode).

计算公式 (docs/aclnnScatterDiv.md):
    varRef[indices[i], ...] = varRef[indices[i], ...] / updates[i, ...]
多个 updates 命中同一切片时连除（顺序无关，与 TF scatter_div 一致）。
越界索引 (idx < 0 或 idx >= var.shape[0]) 跳过 (scatter_reduce_common_simt.h:112)。
整型 dtype (int32/int8/uint8) 走 C++ 整数除法（向零截断）。
in-place: 输出 var 与输入 var 同缓冲（output_inplace_indexes=(0,)）。

实现说明: 计算由 numpy/python 公式改为 torch 竞品算子拼接（index_select + torch.div +
index_copy_，整型走 rounding_mode="trunc" 即 C++ 向零截断）——纯 numpy 公式实现与被测
kernel 易犯同类错误，会掩盖 kernel 精度短板。numpy 仅保留 I/O 与 dtype 转换。
除法不满足交换律（a/b1/b2 != a/(b1*b2)，逐位不同），故仍按索引顺序逐条下发，
不能折叠成一次 index_reduce；语义（连除顺序、越界跳过、工作 dtype、返回结构）未变。
"""

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None

_INT_DTYPES = {"int32", "int8", "uint8"}


def __golden_scatter_div(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtypes = kwargs.get("output_dtypes", None)
    if out_dtypes:
        out_dt_str = out_dtypes[0]
    else:
        out_dt_str = str(var.dtype)
    is_int = out_dt_str in _INT_DTYPES

    var_first = var.shape[0]
    slice_shape = var.shape[1:]
    slice_size = int(np.prod(slice_shape)) if slice_shape else 1

    # 工作 dtype: 整型走 int64 防连除溢出; 浮点**跟随 TTK 下发的 dtype, 只向上兜底不向下砍**。
    # golden 在 cross_check 下收到的是 Promote 后的入参(fp16/bf16 -> fp32, fp32 -> float64),
    # 若无条件 astype(np.float32) 就把 fp32 档的 float64 真值砍回 fp32 —— 等于撤销 Promote,
    # 使 golden 与三方腿逼近逐位相等, 三比值分母被夹到精度标准 §4.5.1 的 err, RMSE 比值假红。
    # bf16 无 numpy 原生类型, 只有它需要向上桥接到 fp32。
    def _wd(a):
        return (
            np.float32 if a.dtype.kind == "V" or a.dtype.name == "bfloat16" else a.dtype
        )

    if is_int:
        work = torch.from_numpy(var.astype(np.int64).reshape(var_first, slice_size))
        upd = torch.from_numpy(updates.astype(np.int64).reshape(-1, slice_size))
    else:
        wdt = np.promote_types(_wd(var), _wd(updates))
        work = torch.from_numpy(var.astype(wdt).reshape(var_first, slice_size))
        upd = torch.from_numpy(updates.astype(wdt).reshape(-1, slice_size))

    idx_flat = indices.reshape(-1).astype(np.int64)
    n = idx_flat.shape[0]

    for m in range(n):
        idv = int(idx_flat[m])
        if idv < 0 or idv >= var_first:
            continue  # out-of-bound skip
        row = work.index_select(0, torch.tensor([idv]))
        if is_int:
            # C++ integer division (truncation toward zero)
            denom = upd[m : m + 1]
            nonzero = denom != 0
            # match C++ UB conservatively: leave unchanged is not defined;
            # use trunc-toward-zero with denom guarded to avoid a division by zero.
            safe = torch.where(nonzero, denom, torch.ones_like(denom))
            res = torch.where(nonzero, torch.div(row, safe, rounding_mode="trunc"), row)
        else:
            res = torch.div(row, upd[m : m + 1])
        work.index_copy_(0, torch.tensor([idv]), res)

    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": np.float32,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }.get(out_dt_str, var.dtype)

    out = work.numpy().reshape(var.shape).astype(np_dtype)
    return [out]


def __golden_scatter_div_e2e(var, indices, updates, use_locking=None, **kwargs):
    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    return __golden_scatter_div(_np(var), _np(indices), _np(updates))


__golden__ = {
    "kernel": {"scatter_div": "__golden_scatter_div"},
    "e2e": {"tf.compat.v1.scatter_div": "__golden_scatter_div_e2e"},
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
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。

    ⚠️ 不能把 fp16/bf16 升 fp32 再算。本算子对重复索引是**链式**运算, 误差随链长累积;
    三方若用更高精度跑, cross_check 拿到的就是"内核误差 / 一个 fp16 实现物理上够不到的
    参照"之比, 会系统性判红——实测 scatter_div fp16 链长中位 8 的用例 mare_ratio 12.78
    (限值 5), 改回与算子同 dtype 的语义后 0.65 通过, 内核本身的误差 99.99% 落在 fp16
    链式误差预算内。三方必须与算子同 dtype 语义, 内核误差才有可比对象。
    golden 反过来要保持高精度(fp32 链), 它是两条腿共同的参照点。
    """
    if isinstance(x, torch.Tensor):
        return x.clone()
    t = torch.as_tensor(np.asarray(x))  # 仅本地自测兜底: 框架侧不会走到
    return t.to(torch.float32) if t.dtype == torch.bfloat16 else t.clone()


def scatter_div_input(var, indices, updates, **kwargs):
    """
    Input function for scatter_div.
    All the parameters (names and order) follow scatter_div_def.cpp without outputs.
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


def _tp_widen(t):
    """加宽到内核的累加类型(fp16/bf16 按 fp32 累算), 与内核同算法: 链式相除整条链
    都在加宽类型上做, 只在出口窄一次。整型保持整型(截断除须在整型域内逐步做)。"""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def _tp_narrow(outs, dt):
    """算完窄回算子输出 dtype, 必须与 _tp_widen 成对。"""
    return [o.to(dt) if o.is_floating_point() else o for o in outs]


class _ScatterDivCompose:
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
        idx, upd = idx[valid], upd[valid]
        # div 无 index_reduce 归约模式: 按索引顺序逐条相除(重复索引累除, 与算子语义一致)。
        # 整型必须走截断除且除数为 0 时保持原值, 不能升 fp32——大 int32 经 float32 会静默丢精度。
        # 整型判定与 golden 同源(_INT_DTYPES = int32/int8/uint8, 即 def 注册的整型面),
        # 不用 dtype.is_floating_point 泛判, 避免与 golden 的分支口径分叉。
        # _INT_DTYPES = {int32, int8, uint8}（def 注册的整型面）的 torch 对应;
        # 不能用 work.numpy() 反查——CUDA 张量转不了 numpy。
        is_int = _dt in (torch.int32, torch.int8, torch.uint8)
        # 分层向量化: 串行依赖只在同一行内部, 不同行互相独立。按作用次序分层、层内
        # 批量相除, 循环次数由索引数降为单行最大重复次数; stable 排序保证同行内仍按
        # 原顺序, 与逐条相除逐位一致。
        n = int(idx.numel())
        if n:
            ix = idx.cpu().numpy()
            order = np.argsort(ix, kind="stable")
            s = ix[order]
            rank = np.arange(n) - np.searchsorted(s, s, side="left")
            lay = np.argsort(rank, kind="stable")
            sel_all = torch.as_tensor(order[lay], device=idx.device)
            row_all = torch.as_tensor(ix[order[lay]], device=idx.device).to(torch.int64)
            rk = rank[lay]
            bounds = np.searchsorted(rk, np.arange(int(rk.max()) + 2))
            for r in range(len(bounds) - 1):
                a, b = int(bounds[r]), int(bounds[r + 1])
                if a == b:
                    continue
                sel, rows = sel_all[a:b], row_all[a:b]
                d = upd.index_select(0, sel)
                cur = work.index_select(0, rows)
                if is_int:
                    nonzero = d != 0
                    safe = torch.where(nonzero, d, torch.ones_like(d))
                    q = torch.div(cur, safe, rounding_mode="trunc")
                    work[rows] = torch.where(nonzero, q, cur)
                else:
                    work[rows] = torch.div(cur, d)
        return _tp_narrow([work], _dt)


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
    _INNER = _ScatterDivCompose

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
        _ScatterDivCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class _ScatterDivTfCompose:
    """tf 腿适配类。TF 侧对标算子是 ScatterDiv(tf_plugin 的 OriginOpType), 但它是
    **ref-variable 语义**: tf.raw_ops.ScatterDiv 首参名叫 ref 且必须是可变引用, 与
    def.cpp 的 var 既不同名、也不能直接喂 eager tensor, 故用适配类包 tf.Variable 再调。

    与 torch 腿逐条对齐的三处语义(不一致就会系统性假红):
    1) 非法下标(越界/负)按算子语义静默跳过——TF 自己会抛 InvalidArgumentError;
    2) 除数含 0 时不能直接调 TF 的 ScatterDiv: 它对**所有 dtype**硬性拒收零除数
       (实测 tf 2.21: InvalidArgumentError "updates must not contain 0", 浮点也拒),
       即"除数为 0"落在 TF 该算子的定义域之外。而本算子对零除数是有定义的:
       整型保持原值、浮点按普通除法给 ±inf(见 golden 与 torch 腿)。故按 dtype 分流——
         · 整型: 把 0 除数换成 1(x/1 == x, 等价于保持原值), 仍走单次 scatter_div;
         · 浮点且含 0 除数: 退化成按索引顺序的链式 tf.divide(TF 自己的除法核, ±inf
           与 nan 的产生方式与 TF 一致), 只有这条罕见分支走循环, 常规用例仍是单次调用;
    3) **不升精度**: 一律用算子自身 dtype 算(见 _tp_t 的说明——本算子对重复索引是链式
       运算, 三方升精度会让 cross_check 拿 fp16 实现够不到的参照去比, 系统性判红)。
    TF 的整型除法本身就是截断除, 与算子/golden 的 trunc 语义一致, 无需额外处理。
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
        if tf.size(idx) == 0:
            return [tf.cast(work, _tf_dt) if work.dtype != _tf_dt else work]
        upd = tf.cast(upd, work.dtype)
        has_zero = bool(tf.reduce_any(tf.equal(upd, tf.zeros_like(upd))))
        # 整型判定与 golden/torch 腿同源(def 注册的整型面), 不用 is_floating 泛判。
        is_int = work.dtype in (tf.int32, tf.int8, tf.uint8)

        if is_int and has_zero:
            upd = tf.where(tf.equal(upd, tf.zeros_like(upd)), tf.ones_like(upd), upd)
            has_zero = False

        if not has_zero:
            ref = tf.Variable(work)
            tf.compat.v1.scatter_div(ref, tf.cast(idx, tf.int32), upd)
            out = tf.convert_to_tensor(ref)
            return [tf.cast(out, _tf_dt) if out.dtype != _tf_dt else out]

        # 浮点 + 零除数: TF 的 ScatterDiv 拒收, 按索引顺序链式除(与算子/golden 同序)。
        rows = idx.numpy().tolist()
        for k, i in enumerate(rows):
            work = tf.tensor_scatter_nd_update(
                work, [[i]], tf.expand_dims(tf.divide(work[i], upd[k]), 0)
            )
        # 链式除同样要窄回: 上面的加宽只为复刻 LoadWiden, 不能泄漏到出参
        return [tf.cast(work, _tf_dt) if work.dtype != _tf_dt else work]


_GOLDEN_FN = __golden_scatter_div


class ScatterDivKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful, "tf": _ScatterDivTfCompose}
    customize_inputs = scatter_div_input
    tolerance = _TOL_KERNEL


# e2e(TF 前端)通路判据: 与 kernel 腿同口径。不声明则回落默认的绝对容差判据,
# 输出量级接近 dtype 上限时 1 ULP 即超限。
_TOL_E2E = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


class _TpE2eDiv:
    """e2e 三方腿适配: 该通路按框架 API 形参名下发, 与 def 注册名不同, 故另立适配类
    按位置转调同一竞品类, 不改变竞品语义。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterDivCompose()(ref, indices, updates)


class _TpE2eDivTf:
    """同上, tf provider 腿。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterDivTfCompose()(ref, indices, updates)


class ScatterDivE2eSpec:
    """e2e 通路 spec: 三方腿与判据。"""

    third_party = {"torch": _TpE2eDiv, "tf": _TpE2eDivTf}
    tolerance = _TOL_E2E


__spec__ = {
    "scatter_div": "ScatterDivKernelSpec",
    "tf.compat.v1.scatter_div": "ScatterDivE2eSpec",
    "aclnnScatterDiv": "ScatterDivAclnnSpec",
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


class ScatterDivAclnnSpec:
    """aclnn 通路 spec。golden 由 TTK 按 aclnn 头文件形参**位置**下发
    (AclnnParamPlan.build_args), 故签名逐项对齐 aclnnScatterDivGetWorkspaceSize 的
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
        idx, upd = idx[valid], upd[valid]
        is_int = work.dtype in (torch.int32, torch.int8, torch.uint8)
        for k in range(idx.numel()):
            i = int(idx[k])
            if is_int:
                denom = upd[k]
                nz = denom != 0
                safe = torch.where(nz, denom, torch.ones_like(denom))
                q = torch.div(work[i], safe, rounding_mode="trunc")
                work[i] = torch.where(nz, q, work[i])
            else:
                work[i] = torch.div(work[i], upd[k])
        return _keep_dtype([work], varRef)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterDivCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + aclnn
# 未在 __spec__ 中注册:
# TensorFlow: 算子目录下有 framework 的 tf_plugin(OriginOpType "ScatterDiv")。
#   三方标杆(精度/性能)已补: third_party["tf"] = _ScatterDivTfCompose, 跑 --provider tf。
#   通路连通(ⓐ)未注册: TTK 的 tf 通路是 e2e 前端(api_name 写 TF API), NPU 侧需要
#   Ascend TF adapter(npu_device/tfplugin)才能把 TF 图下沉到本算子; 当前环境
#   (cann-9.2.0)未装该组件, 装不上就跑不出 invoke_path 证据, 故不注册空壳键
#   (规范: __spec__ 注册集合必须等于 01 §3.3 的 ✅ 集合与 invoke_path 的通路取值)。
#   该组件到位前, tf 通路连通性仍按预生成 .pb + aclgrphParseTensorFlow 验证。
# e2e / ONNX / 融合 pass: 均未交付。
