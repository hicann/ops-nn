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

"""TTK kernel 模式自定义 golden：foreach_addcmul_list。

编写依据：docs/aclnnForeachAddcmulList.md「功能说明 / 计算公式」节：
    y_i = x1_i + scalars * x2_i * x3_i   (i = 0 .. n-1)

⚠️ 结合序取 (x2*x3) 先乘、再乘标量。公式里三个因子的乘积在数学上与结合序无关，但在
fp32 中间量下不等价：scalars 取极值(如 3.35e38)时 (x2*scalars) 这一步就会冲破 fp32 上限
变成 inf，而 x2*x3 通常是小量、乘上 scalars 后并不溢出。实测某 bf16 用例 8192 个元素中有
7361 个的 |x2*scalars| 超限，按该结合序算出的是一片 inf，与 fp64 参照给出的有限真值相差
甚远。golden 的职责是给出最接近真值的参照，故取不引入伪溢出的那一种。整型分支同理。

输入顺序（与 CSV input_shapes 一致）：x1, x2, x3, scalars
输出顺序（与 CSV output_dtypes 一致）：y

说明：shape_mapping 将每个张量列表映射为单张量（列表长度 1），scalars 为
shape [1] 的单元素张量，对应 totalTensorCount_ == 1。golden 直接对单张量计算。

计算实现改用竞品 torch._foreach_*（红线 R3：golden 只能是竞品接口实现或竞品算子
拼接实现，禁 numpy 纯公式），numpy 仅保留 I/O 与 dtype 转换；浮点在 fp32 中间量、
整型在 int64 中间量按原结合次序计算，与算子的计算精度一致。
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


def _to_compute(a):
    """**跟随 TTK 下发的 dtype 计算，不要自己强制降档。**

    判据是 cross_check 时 TTK 会把 golden 切 golden_mode=Promote，按 DTYPE_PROMOTE_MAP
    抬一档下发(fp32->float64, fp16/bf16->float32)，就是为了让 golden 成为精度标准 §4.5
    「双标杆比对」要求的「**更高精度**的 CPU 实现为真值」。

    原先无条件降回 fp32，导致 golden 与三方腿(fp32 GPU、同一串 torch._foreach_*、同一
    结合序)在 IEEE-754 下**逐位相等**，双标杆塌成单标杆：GPU 误差恒 0，三比值分母全部
    夹到 §4.5.1 的 err。夹底后 MARE/MERE 分子是无量纲相对误差、仍然小，唯独 RMSE 分子是
    **有量纲**绝对误差，比值随输出量级线性放大 —— 「rmse 比 mare/mere 大 30 个数量级」由此而来。

    fp64 保持 fp64(Promote 档)，其余一律 fp32 —— 与改动前一致，非 cross_check 判据行为不变。
    bf16 必须转换：numpy/torch 不能直接在 |V2 视图或 ml_dtypes.bfloat16 上运算。
    """
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    if a.dtype == np.float64:
        return a
    # dtype 已匹配时复用原缓冲: 大张量(单份 GB 级)无谓复制会把进程推向 OOM。
    # 下游 torch 算子均非原地，不会改写输入，故复用安全。
    return a.astype(np.float32, copy=False)


def _to_fp32(a):
    """三方腿沿用的 fp32 口径：三方必须按**算子自身精度**算，不跟着 golden 抬档。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    return a.astype(np.float32, copy=False)


def __golden_foreach_addcmul_list(x1_list, x2_list, x3_list, scalars, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
    # 保留 scalars 原始 dtype：整数分支要用精确的整数标量，先过一道 float32 会把
    # 超过 2^24 的 int32 抹掉低位（1564714939 -> 1564714880），使整个整数结果偏掉。
    scalars_raw = np.asarray(scalars).reshape(-1)
    scalars_arr = scalars_raw.astype(np.float32)
    results = []
    for i, (a, b, c) in enumerate(zip(x1_list, x2_list, x3_list)):
        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)
        s = scalars_arr[i] if i < scalars_arr.size else scalars_arr[-1]
        if target != "bfloat16" and np.issubdtype(np.dtype(target), np.integer):
            dt = np.dtype(target)
            narrow = torch.from_numpy(np.empty(0, dtype=dt)).dtype
            a_i = torch.from_numpy(np.asarray(a).astype(dt)).to(torch.int64)
            b_i = torch.from_numpy(np.asarray(b).astype(dt)).to(torch.int64)
            c_i = torch.from_numpy(np.asarray(c).astype(dt)).to(torch.int64)
            s_raw = scalars_raw[i] if i < scalars_raw.size else scalars_raw[-1]
            s_i = int(np.asarray(s_raw).astype(dt))
            prod = torch._foreach_mul(torch._foreach_mul([b_i], [c_i]), s_i)
            y = torch._foreach_add([a_i], prod)[0].to(narrow).numpy()
        else:
            # 浮点路径跟随下发的计算精度(Promote 档为 fp64 真值,其余为 fp32),再 cast 回目标 dtype。
            ta = torch.from_numpy(_to_compute(a))
            tb = torch.from_numpy(_to_compute(b))
            tc = torch.from_numpy(_to_compute(c))
            # 标量落成 0 维张量: torch 类型提升里 0 维张量不抬 dim>0 张量的档,
            # 故数据 fp32 时结果仍 fp32,数据 fp64 时标量自动跟到 fp64。
            ts = torch.as_tensor(np.asarray(s, dtype=np.float64))
            prod = torch._foreach_mul(torch._foreach_mul([tb], [tc]), ts)
            out = torch._foreach_add([ta], prod)[0].numpy()
            if target == "bfloat16":
                y = out.astype(_bf16) if _bf16 is not None else out.astype(np.float32)
            else:
                y = out.astype(target)
        results.append(y)
    return results


__golden__ = {"kernel": {"foreach_addcmul_list": "__golden_foreach_addcmul_list"}}

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
    """bf16 用 ml_dtypes 承载, torch.from_numpy 不认, 须按位 view 成 torch.bfloat16(同宽无损)。
    这是**载体还原**, 不是精度干预。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    if _bf16 is not None and a.dtype == _bf16:
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(a)


def _tp_list(xs):
    """third_party 入参: **不替 torch 决定精度**, 原样交给它。

    是否在内部抬到 fp32 由 torch 的算子实现自行决定(CUDA 上不少算子对 fp16 输入会用
    fp32 累加)。三方标杆的价值在于 CPU/NPU/GPU 三套**独立实现**互相制约, 一旦按内核的
    算法去复刻, 它就不再独立、发现不了算法层面的缺陷。

    ⚠️ 曾经在这里嵌套调用 _to_fp32(golden 侧的转换器), numpy 入参一进来就被转成 fp32,
    下面那个"保持原 dtype"的判断**永远命中 fp32 分支、等于没生效** —— fp16 用例的三方腿
    因此仍被抬到 fp32、与 Promote 后的 golden 逐位相等, 分母照塌。改为只做 bf16 载体还原。
    """
    return [a if isinstance(a, torch.Tensor) else _tp_bf16_carrier(a) for a in xs]


def _tp_scalars_t(v, ref):
    """三方标量: **不转 Python float**(那是 fp64, 会触发 torch 类型提升、把整条链抬档);
    整型标量尤其不能过 float —— 超过 2^24 的 int32 会被抹掉低位(同 #85 的成因)。

    torch._foreach_addc{mul,div} 的 scalars 形参收 **Tensor** 或 Number 序列, 不收
    0 维张量的列表, 因此这里返回一维 Tensor 并对齐数据 dtype。"""
    t = v if isinstance(v, torch.Tensor) else _tp_bf16_carrier(np.asarray(v))
    rd = (
        ref.dtype
        if isinstance(ref, torch.Tensor)
        else torch.as_tensor(np.asarray(ref)).dtype
    )
    # ⚠️ 必须落在 **CPU**: torch._foreach_addc{mul,div} 的 scalars 形参要求在 CPU 上,
    # 三方腿在 GPU 上跑, 标量跟着上 cuda 会报
    # "RuntimeError: Expected scalars to be on CPU, got cuda:0"。
    # dtype 仍对齐数据(不转 Python float —— 那是 fp64, 会触发类型提升;
    # 整型标量过 float 还会把 >2^24 的 int32 抹掉低位, 见 #85)。
    return t.reshape(-1).to(rd).cpu()


_GOLDEN_FN = __golden_foreach_addcmul_list


class _ForeachAddcmulListCompose:
    def __call__(self, x1, x2, x3, scalars, **kwargs):
        st = (
            scalars
            if isinstance(scalars, torch.Tensor)
            else torch.as_tensor(np.asarray(scalars))
        )
        a, b, c = _tp_list(x1), _tp_list(x2), _tp_list(x3)
        sc = _tp_scalars_t(st, a[0])
        # 与 CPU golden 同口径: mul -> mul -> add **三步拼接**, 不用 addcmul 的
        # 融合形式。融合是一次 FMA 舍入, 与 golden 的逐步舍入序列不是同一个算法,
        # 两条腿会恒差 1 ULP, cross_check 的 mare 比值随之假红。
        # sc 是一维 packed scalars(addcmul 的 scalars 形参要求如此), 但 _foreach_mul
        # 只收 0 维 Tensor 或 Python 数值列表, 直接传会抛
        # "scalar tensor expected to be 0 dim"。tolist() 按 dtype 还原为 int/float,
        # 整型不过 float 故不抹低位; Python 数值是 weak-typed, 不会抬高结果 dtype。
        sl = sc.tolist()
        return torch._foreach_add(a, torch._foreach_mul(torch._foreach_mul(b, c), sl))


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
    _INNER = _ForeachAddcmulListCompose

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
        _ForeachAddcmulListCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class ForeachAddcmulListKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_addcmul_list": "ForeachAddcmulListKernelSpec",
    "aclnnForeachAddcmulList": "ForeachAddcmulListAclnnSpec",
    "torch.ops.aten._foreach_addcmul.Tensor": "ForeachAddcmulListTorchSpec",  # e2e：算子无原生支持，需 torch_npu meta 补丁，详见下方说明
}


def _tp_one(t):
    """aclnn 通路三方腿入参: **不替 torch 决定精度**, 原样交给它。

    三方腿的输入 dtype 与 NPU 一致, torch 算完自然就是同一 dtype, 无需人为抬档或回 cast;
    是否在内部抬到 fp32 由 torch 的算子实现自行决定。此前无条件把 fp16/bf16 抬到 fp32,
    会让三方与走 Promote(fp32) 的 golden 逐位相等 —— 双标杆塌成单标杆, 三比值分母夹到
    §4.5.1 的 err, 有量纲的 RMSE 比值随输出量级线性放大而假红。

    kernel / GEIR / aclnn / e2e 四条通路的三方腿共用此口径。
    """
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)


def _golden_one(t):
    """CPU golden: 低精度浮点使用 fp32 中间量，整型和 fp64 保持原精度。"""
    t = _tp_one(t).detach().cpu()
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


def _tp_scalars(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().reshape(-1).tolist()
    return list(v)


def _tp_int_scalars(v, dtype):
    """整型分支的标量: 保持精确整数。走 float() 会把超过 2^24 的 int32 抹掉低位
    (1564714939 -> 1564714880), 整个整数结果随之偏掉。"""
    t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    return [int(x) for x in t.detach().cpu().reshape(-1).to(dtype)]


def _bcast_scalars(sc, n):
    """scalars 逐张量对齐; 数量不足时沿用最后一个(与 kernel 通路 golden 同规则)。"""
    if not sc:
        return [0] * n
    return [sc[i] if i < len(sc) else sc[-1] for i in range(n)]


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


class ForeachAddcmulListAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachAddcmulListGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 复用 kernel 通路的竞品类
    ——其形参名即 def 注册名, 与头文件一致。"""

    @staticmethod
    def golden(x1, x2, x3, scalars, out=None, **kwargs):
        a = [_golden_one(t) for t in x1]
        b = [_golden_one(t) for t in x2]
        c = [_golden_one(t) for t in x3]
        # 整型(def 支持 int32)必须走 int64 中间量 + 精确整数标量, 与 kernel 通路 golden
        # (__golden_foreach_addcmul_list 的整型分支)和算子实现同口径: 算子的 ComputeIntPath
        # 全程在整型 RegTensor 上 Mul/Muls/Add, 即 int32 回绕算术。
        # 若沿用浮点路径, _tp_scalars 的 float() 标量会把整条链提升到 float32,
        # 乘积冲破 2^31 后 cast 回 int32 全部饱和成 -2147483648。
        narrow = a[0].dtype if a else torch.int32
        if not narrow.is_floating_point:
            sc_i = _tp_int_scalars(scalars, narrow)
            ai = [t.to(torch.int64) for t in a]
            bi = [t.to(torch.int64) for t in b]
            ci = [t.to(torch.int64) for t in c]
            prod = [
                torch.mul(bb, cc) * int(ss)
                for bb, cc, ss in zip(bi, ci, _bcast_scalars(sc_i, len(bi)))
            ]
            return _keep_dtype(
                [torch.add(aa, pp).to(narrow) for aa, pp in zip(ai, prod)], x1
            )
        sc = _tp_scalars(scalars)
        return _keep_dtype(
            torch._foreach_add(a, torch._foreach_mul(torch._foreach_mul(b, c), sc)), x1
        )

    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


class _TpE2e:
    """e2e 通路三方腿适配: 池的 key 取自 torch 重载的形参名
    (self / tensor1 / tensor2 / scalars), 与 def 注册名 (x1 / x2 / x3 / scalars) 不同;
    直接复用 kernel 腿的竞品类会因形参 x1 不在 pool 中而抛 UnknownParamError,
    三方腿整条起不来。故按 torch 形参名另立适配类, 内部转调同一个竞品类, 不改变竞品语义。
    """

    def __call__(self, *args, **kwargs):
        # TTK 按名下发(self/tensor1/tensor2/scalars), 内层竞品类的形参是 def 注册名
        # (x1/x2/x3/scalars), 此处做一次名字映射后按位置转调。
        # 位置/关键字混合下发都要兜住; 同时兼容 def 注册名(x1/x2/x3)。
        aliases = (
            ("self", "x1"),
            ("tensor1", "x2"),
            ("tensor2", "x3"),
            ("scalars",),
        )
        vals = list(args)
        for group in aliases[len(vals) :]:
            for n in group:
                if n in kwargs:
                    vals.append(kwargs[n])
                    break
        if len(vals) != 4:
            raise TypeError(
                f"_TpE2e 收到 args={len(args)} kwargs={sorted(kwargs)} -> vals={len(vals)}"
            )
        return _TpKernelFaithful()(*vals)


# 绑定方法的签名会丢掉首个形参, 故首位用占位名, 其后才是 torch 的形参名。
try:
    _TpE2e.__call__.__signature__ = _kf_inspect.Signature(
        [
            _kf_inspect.Parameter(n, _kf_inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for n in ("_inst", "self", "tensor1", "tensor2", "scalars")
        ]
    )
except (ValueError, TypeError):
    pass


# ===========================================================================
# ⚠️ e2e 通路：算子**实际不具备 e2e 支持**，下方 ForeachAddcmulListTorchSpec 仅在
# 打过本地补丁的环境里才跑得起来，不代表产品能力。
#
# 原因：aten::_foreach_addcmul.Tensor 这个重载把 scalars 当**输入张量**传，torch 的
# CompositeExplicitAutograd 实现第一步就调 convert_tensor_to_scalar_list() 解引用
# scalars 的数据。追踪期 meta 张量没有数据，直接抛
#     "Expected scalars to be on CPU, got meta instead."
# → torch.compile 建不了图 → torchair 的 ge.ForeachAddcmulList converter 永远走不到。
#
# 2026-09-09 之所以能跑出 e2e 结果，是因为**手工给已安装的 torch_npu 打了补丁**：
#     site-packages/torch_npu/op_plugin/meta/_meta_registrations.py
#     追加 @impl(m_aten, "_foreach_addcmul.Tensor") 的 meta 实现
# 那是装好的包内改动，不在本仓、不是交付件，torch_npu 一重装就失效。
#
# 因此：**不要把 e2e 当作本算子的已交付通路**。若要正式交付 e2e，前置条件是上述
# meta 注册进入 torch_npu 正式版本，届时再把这里的说明去掉。
# ===========================================================================
class ForeachAddcmulListTorchSpec:
    """E2E Tensor/ScalarList overload: 无 ACLNN out 参数，返回 TensorList。"""

    @staticmethod
    def golden(self, tensor1, tensor2, scalars, **kwargs):
        return ForeachAddcmulListAclnnSpec.golden(
            self, tensor1, tensor2, scalars, **kwargs
        )

    # e2e 通路与 kernel / aclnn 同口径: 两条腿都要声明, 否则 cross_check 缺三方腿,
    # 判据会退化成 GOLDEN_FAILURE。
    #   * CPU golden(_golden_one): 低精度浮点用 fp32 中间量, 整型与 fp64 保持原精度,
    #     不做强制降档 —— 与 TTK 的 golden_mode=Promote 口径一致。
    #   * GPU 三方腿(_tp_one): 不替 torch 决定精度, 原样交给它, 是否内部抬到 fp32
    #     由 torch 的算子实现按需决定 —— 保证三方与 golden 是两套独立实现。
    third_party = {"torch": _TpE2e}
    tolerance = _TOL_KERNEL
