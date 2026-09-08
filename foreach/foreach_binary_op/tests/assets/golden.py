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
TTK custom golden for foreach_binary_op.

Compute formula (foreach_binary_op_proto.h:21; op_kernel/arch35/foreach_binary_op_simt.h:52):
    y[t][i] = x1[t][i] <op> x2[t][i]   (t = 0..n-1 list elements, i over numel)
where <op> is selected by REQUIRED attr op_code:
    0 = add, 1 = sub, 2 = mul, 3 = div.

Non-inplace: output y is a separate TensorList of n sub-tensors, same per-tensor shape/dtype
as x1 (and x2). Output order == x1 sub-tensor order.

dtype handling mirrors the kernel (simt.h):
  * float32      : direct compute.
  * float16/bf16 : cast up to float32, compute, cast back (BinaryApply on T=fp32 after Cast*).
  * int32        : native integer ops with 2's-complement wraparound (NO saturation) for
                   add/sub/mul. For div: b == 0 -> 0 (device guard, simt.h:65), otherwise
                   integer division truncated toward zero (C semantics).
  * float div b == 0 -> IEEE inf/nan, left as-is (simt.h:67).

Positional args (TTK passes context.input_arrays unflattened, in CSV input_shapes order):
    x1_list : list of numpy arrays  (DYNAMIC TensorList x1, n sub-tensors)
    x2_list : list of numpy arrays  (DYNAMIC TensorList x2, sync per-tensor with x1)
op_code is delivered via **kwargs (TTK passes parsed `attributes` entries as kwargs).

计算实现改用竞品 torch._foreach_* (红线 R3: golden 只能是竞品接口实现或竞品算子拼接实现,
禁 numpy 纯公式), numpy 仅保留 I/O 与 dtype 转换; 数值与改造前逐位一致。整型除法用
torch.div(rounding_mode="trunc") 并保留 b == 0 -> 0 的设备守卫(torch 整除 0 会抛异常)。
"""

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

OP_ADD, OP_SUB, OP_MUL, OP_DIV = 0, 1, 2, 3


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
    # Promote 后的 float64 是高精度真值, **不能砍回 fp32**: 砍了就与三方腿(fp32)
    # 逐位相等, cross_check 三比值分母被夹到精度标准 §4.5.1 的 err ——
    # 无量纲的 mare/mere 恒等于地板值(实测 fp32 档 mare 恒为 0.0019531 = 1ULP/2^-14),
    # 有量纲的 RMSE 随输出量级放大而假红。fp32 档等于从未被真正验证。
    if a.dtype == np.float64:
        return a
    return a.astype(np.float32, copy=False)


def _resolve_op_code(kwargs):
    op_code = kwargs.get("op_code")
    if op_code is None:
        attrs = kwargs.get("attributes") or {}
        if isinstance(attrs, dict):
            op_code = attrs.get("op_code")
    if op_code is None:
        op_code = OP_ADD
    return int(op_code)


def _int_binary(a, b, op_code, dt):
    # 2's-complement wrap for add/sub/mul via wide int -> narrow cast; div guarded + truncated.
    # 宽中间量走 torch int64, 窄化回 dt 由 torch 的 C 截断完成(与 numpy 窄化同语义)。
    narrow = torch.from_numpy(np.empty(0, dtype=dt)).dtype
    aw = torch.from_numpy(np.asarray(a).astype(dt)).to(torch.int64)
    bw = torch.from_numpy(np.asarray(b).astype(dt)).to(torch.int64)
    if op_code == OP_ADD:
        return torch._foreach_add([aw], [bw])[0].to(narrow).numpy()
    if op_code == OP_SUB:
        return torch._foreach_sub([aw], [bw])[0].to(narrow).numpy()
    if op_code == OP_MUL:
        return torch._foreach_mul([aw], [bw])[0].to(narrow).numpy()
    # OP_DIV: b == 0 -> 0, else truncate toward zero (integer-only, exact, no float rounding).
    zero = bw == 0
    safe = torch.where(zero, torch.ones_like(bw), bw)
    q = torch.div(aw, safe, rounding_mode="trunc")
    return torch.where(zero, torch.zeros_like(q), q).to(narrow).numpy()


def __golden_foreach_binary_op(x1_list, x2_list, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
    op_code = _resolve_op_code(kwargs)

    results = []
    for i, (a, b) in enumerate(zip(x1_list, x2_list)):
        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)

        is_int = (target != "bfloat16") and np.issubdtype(np.dtype(target), np.integer)

        if is_int:
            out = _int_binary(a, b, op_code, np.dtype(target))
        else:
            ta = torch.from_numpy(_to_fp32(a))
            tb = torch.from_numpy(_to_fp32(b))
            if op_code == OP_ADD:
                out = torch._foreach_add([ta], [tb])[0]
            elif op_code == OP_SUB:
                out = torch._foreach_sub([ta], [tb])[0]
            elif op_code == OP_MUL:
                out = torch._foreach_mul([ta], [tb])[0]
            else:  # OP_DIV: IEEE inf/nan on b == 0, matching the NPU float path
                out = torch._foreach_div([ta], [tb])[0]
            out = out.numpy()
            if target == "bfloat16":
                out = out.astype(_bf16) if _bf16 is not None else out
            else:
                out = out.astype(target)
        results.append(out)
    return results


__golden__ = {"kernel": {"foreach_binary_op": "__golden_foreach_binary_op"}}

# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 在保留原 golden 的基础上补三方标杆能力。
# third_party 直接对标 torch 的 _foreach_* 竞品 API，在设备侧跑，供 cross_check 比对。
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


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


_GOLDEN_FN = __golden_foreach_binary_op


def _tp_int64(a):
    """整型入参提到 int64 做中间运算; 入参已是设备张量时不经 numpy。"""
    t = a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))
    return t.to(torch.int64)


class _ForeachBinaryOpCompose:
    """三方标杆必须跟随 op_code 分派: 该算子按属性 op_code 分解为 add/sub/mul/div,
    写死单一 API 只对 add 成立。整型走与 golden 同款的宽中间量+窄化(2's-complement
    wrap), 不能升 fp32——大 int32 经 float32 会静默丢精度。"""

    def __call__(self, x1, x2, **kwargs):
        op_code = _resolve_op_code(kwargs)
        fn = {
            OP_ADD: torch._foreach_add,
            OP_SUB: torch._foreach_sub,
            OP_MUL: torch._foreach_mul,
            OP_DIV: torch._foreach_div,
        }[op_code]
        first = x1[0] if len(x1) else None
        ft = None
        if first is not None:
            ft = (
                first
                if isinstance(first, torch.Tensor)
                else torch.as_tensor(np.asarray(first))
            )
        # 用 torch dtype 判整型: 入参是 CUDA 张量时 np.asarray 会崩
        is_int = ft is not None and not ft.is_floating_point()
        if is_int:
            narrow = ft.dtype
            aw = [_tp_int64(a) for a in x1]
            bw = [_tp_int64(b) for b in x2]
            if op_code == OP_DIV:
                # 整型除法: b == 0 取 0, 否则向零截断(纯整数, 不经浮点)
                outs = []
                for a_, b_ in zip(aw, bw):
                    zero = b_ == 0
                    safe = torch.where(zero, torch.ones_like(b_), b_)
                    q = torch.div(a_, safe, rounding_mode="trunc")
                    outs.append(torch.where(zero, torch.zeros_like(q), q).to(narrow))
                return outs
            return [t.to(narrow) for t in fn(aw, bw)]
        return fn(_tp_list(x1), _tp_list(x2))


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
    _INNER = _ForeachBinaryOpCompose

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
        _ForeachBinaryOpCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class ForeachBinaryOpKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {"foreach_binary_op": "ForeachBinaryOpKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md, 该算子只对内分解使用。
# e2e / TensorFlow / ONNX / 融合 pass: 均未交付。
