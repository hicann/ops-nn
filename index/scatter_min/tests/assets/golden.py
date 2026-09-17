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

"""TTK custom golden for scatter_min (kernel mode).

计算公式 (依据 aclnnScatterMin.md 「功能说明」节):
    varRef[indices[i], ...] = min(varRef[indices[i], ...], updates[i, ...])
  - indices 为扁平的索引条目序列; 对第 i 个索引条目, 取 var 的第 indices[i] 行
    (slice, 即 var[indices[i]]) 与 updates 的第 i 个 slice 逐元素取 min.
  - 越界索引被跳过 (scatter_reduce_common_simt.h:112: idxVal<0 或 >=var.shape[0] -> skip),
    与 kernel 语义一致.
  - 重复索引依次取 min (顺序无关, min 满足交换律/结合律).
  - 输出为原地更新后的 var (单输出).

输入顺序 (scatter_min_def.cpp): var, indices, updates
输出顺序 (scatter_min_def.cpp / infershape.cpp): var (inplace)

实现说明: 计算由 numpy 逐条 np.minimum 改为 torch 竞品算子 Tensor.index_reduce_(reduce="amin",
include_self=True)——纯 numpy 公式实现与被测 kernel 易犯同类错误, 会掩盖 kernel 精度短板。
numpy 仅保留 I/O 与 dtype 转换。语义(重复索引取最小、越界跳过、工作 dtype、返回结构) 未变。
"""

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None


def __golden_scatter_min(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtype = var.dtype
    var_shape = var.shape

    # var.shape[0] = 索引上界; slice = var.shape[1:]
    var_first_dim = var_shape[0] if var_shape else 0
    slice_shape = tuple(var_shape[1:])
    slice_size = int(np.prod(slice_shape)) if slice_shape else 1

    # 空切片 (var.shape 含 0 维 -> slice_size==0): 每个 slice 0 宽, scatter 是 no-op,
    # var 原样返回 (kernel 同样在 sliceSize==0 时直接 return)。不加这道会让下面
    # upd_flat=(0,1) 在循环里 upd_flat[i] 越界 -> IndexError(golden 侧 GOLDEN_FAILURE)。
    if slice_size == 0:
        return [var]  # 出口不 cast, TTK 负责

    # 中间计算用 float32, 整型保持原类型精确比对
    # 浮点跟随 TTK 下发的 dtype(理由同 scatter_mul 的注释): 写死 float32 会撤销 Promote,
    # 使 golden 与三方腿逼近逐位相等, 三比值分母被夹, RMSE 比值假红。
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
    sub_int = np.issubdtype(out_dtype, np.integer) and out_dtype.itemsize < 4
    narrow_fp = (not promoted) and out_dtype == np.float16
    acc_dtype = np.int32 if sub_int else (np.float32 if narrow_fp else out_dtype)
    need_narrow = acc_dtype != out_dtype
    work_dtype = acc_dtype

    # 不在此处做全尺寸 astype —— int8->int32 会把整个 var 放大 4 倍(dim0=2^30 时 4.3GB)。
    # 只保留二维视图, 真正的 work_dtype 转换推迟到"只取命中行"之后。
    var_2d = var.reshape(var_first_dim if var_first_dim else 0, slice_size)

    idx_flat = indices.reshape(-1).astype(np.int64)
    # updates 扁平化为 (indices_num, slice_size)
    # 同理: astype 推迟到取完命中条目之后, 避免对整个 updates 放大。
    upd_2d = updates.reshape(-1, slice_size) if slice_size else updates.reshape(-1, 1)

    n = idx_flat.shape[0]
    # 越界索引先剔除 (与 kernel 的 skip 一致), 再把命中的行压缩成小张量交给 torch 竞品算子;
    # include_self=True 即 min(var 原值, 命中该行的所有 updates), 重复索引顺序无关。
    # 只对被索引命中的行做归约: 未命中行按 scatter 语义原值输出。为整个 var 开 acc 缓冲时,
    # var 首维 2^30(int8 入参、int32 acc)要 4.3GB、2^32 要 17GB —— golden 会先于算子 OOM,
    # 宽档用例一例也验不了。逐元素结果与全量计算相同。
    out_arr = var.copy()
    valid = (idx_flat >= 0) & (idx_flat < var_first_dim)
    pos = np.nonzero(valid)[0]
    if n == 0 or pos.size == 0:
        return [out_arr]  # 无有效索引: 与 kernel 一致, var 原值输出
    # touched 为去重后的真实行号(有序), inverse 是各条目在 touched 中的下标;
    # 行号重标号不改变 amin 结果。
    touched, inverse = np.unique(idx_flat[pos], return_inverse=True)
    result_t = torch.from_numpy(
        np.ascontiguousarray(var_2d[touched]).astype(work_dtype, copy=False)
    )
    upd_t = torch.from_numpy(
        np.ascontiguousarray(upd_2d[:n][pos]).astype(work_dtype, copy=False)
    )
    result_t.index_reduce_(
        0, torch.from_numpy(inverse.astype(np.int64)), upd_t, "amin", include_self=True
    )

    # 出口不 cast —— TTK 负责窄回。自行 astype 会把 Promote 出来的高精度真值砍回去。
    res = result_t.numpy()
    if need_narrow:
        res = res.astype(out_dtype)  # 复刻 NarrowStore
    out_arr.reshape(var_first_dim, slice_size)[touched] = res
    return [out_arr]


def __golden_scatter_min_e2e(var, indices, updates, use_locking=None, **kwargs):
    """e2e(TF 前端)通路 golden: 入参为框架张量, 还原为 numpy 后复用 kernel 档实现。"""

    def _np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    return __golden_scatter_min(_np(var), _np(indices), _np(updates))


__golden__ = {
    "kernel": {"scatter_min": "__golden_scatter_min"},
    "e2e": {"tf.compat.v1.scatter_min": "__golden_scatter_min_e2e"},
}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册: 保留原 golden，补三方标杆。
# third_party 用 torch / tf 竞品算子在设备侧跑，供 cross_check 比对。
# 不注册 customize_inputs —— 索引取值(含跨桶同低位、超 int32、INT64_MAX 等)直接写在
# 用例集的 input_data_ranges 多元组里，由 TTK 保证这些值必然出现在生成数据中。
# golden 只负责算参照值，不承担输入构造。
# ----------------------------------------------------------------------------
# 浮点走 cross_check: 本算子挂了 third_party, 而 TTK 仅在 standard == cross_check 时
# 才取用三方腿 —— 写成 binary_equal 会让三方腿永不被调用, golden 成为唯一裁判。
# min/max 遇 ±0 是 IEEE 未指定的 tie, 逐位判据下会把数值相同的结果判为不等。
# 整型是精确选择语义, 保持逐位相等。
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
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。"""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    # 不抬精度: 三方标杆按算子自身 dtype 计算。此前 fp16/bf16 统一升 fp32 会让三方与走
    # Promote(fp32) 的 golden 逐位相等, cross_check 分母塌到 safe_div 的 small_value 地板,
    # 判据退化成"NPU 与 fp32 参照的绝对误差", 随输出量级线性放大而必红。
    return t.clone()


class _ScatterMinCompose:
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
            work = work.index_reduce(0, idx, upd[valid], "amin", include_self=True)
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
    _INNER = _ScatterMinCompose

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
        _ScatterMinCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class _ScatterMinTfCompose:
    """tf 腿适配类。TF 侧对标算子是 ScatterMin(tf_plugin 的 OriginOpType), 但它是
    **ref-variable 语义**: tf.raw_ops.ScatterMin 首参名叫 ref 且必须是可变引用, 与 def.cpp
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
            tf.compat.v1.scatter_min(
                ref, tf.cast(idx, tf.int32), tf.cast(upd, ref.dtype)
            )
        out = tf.convert_to_tensor(ref)
        return [
            tf.cast(out, _tf_dt) if out.dtype != _tf_dt else out
        ]  # 复刻 NarrowStore


_GOLDEN_FN = __golden_scatter_min


class ScatterMinKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful, "tf": _ScatterMinTfCompose}
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


class _TpE2eMin:
    """e2e 三方腿适配: 该通路按框架 API 形参名下发, 与 def 注册名不同, 故另立适配类
    按位置转调同一竞品类, 不改变竞品语义。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterMinCompose()(ref, indices, updates)


class _TpE2eMinTf:
    """同上, tf provider 腿。"""

    def __call__(self, ref, indices, updates, use_locking=None, **kwargs):
        return _ScatterMinTfCompose()(ref, indices, updates)


class ScatterMinE2eSpec:
    """e2e 通路 spec: 三方腿与判据。"""

    third_party = {"torch": _TpE2eMin, "tf": _TpE2eMinTf}
    tolerance = _TOL_E2E


class ScatterMinAclnnSpec:
    """aclnn 通路 spec。

    golden 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名对齐 aclnnScatterMinGetWorkspaceSize 的 varRef/indices/updates/useLocking;
    third_party 按**形参名**绑定, 用适配类把头文件的 varRef 接到竞品类的 def 注册名 var。
    """

    @staticmethod
    def golden(varRef, indices, updates, useLocking=None, **kwargs):
        """入参为 torch 张量/numpy 视图, 还原成 numpy 后转接 kernel 档 golden(单一真源)。

        类体内不能直接写 __golden_xxx —— 会被改写成 _ScatterMinAclnnSpec__golden_xxx, 故用 _GOLDEN_FN。
        """

        def _np(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

        return _GOLDEN_FN(_np(varRef), _np(indices), _np(updates), **kwargs)

    class _Compose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterMinCompose()(varRef, indices, updates, **kwargs)

    class _TfCompose:
        def __call__(self, varRef, indices, updates, **kwargs):
            return _ScatterMinTfCompose()(varRef, indices, updates, **kwargs)

    third_party = {"torch": _Compose, "tf": _TfCompose}
    tolerance = _TOL_KERNEL


# 通路交付情况
# __spec__ 已注册: kernel + GEIR(复用 kernel spec) + aclnn + e2e(TF 前端);
#   三个 spec 的 third_party 均为 {"torch", "tf"}, tf 腿跑 --provider tf。
# TF 通路连通性: 算子目录下有 framework 的 tf_plugin(OriginOpType "ScatterMin"), 但 NPU 侧要把
#   TF 图下沉到本算子需要 Ascend TF adapter(npu_device/tfplugin); 该组件未装时按
#   预生成 .pb + aclgrphParseTensorFlow 验证连通性。
# ONNX / 融合 pass: 均未交付。


__spec__ = {
    "scatter_min": "ScatterMinKernelSpec",
    "tf.compat.v1.scatter_min": "ScatterMinE2eSpec",
    "aclnnScatterMin": "ScatterMinAclnnSpec",
}
