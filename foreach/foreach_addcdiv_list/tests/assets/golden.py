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
TTK custom golden for foreach_addcdiv_list.

Compute formula (docs/aclnnForeachAddcdivList.md:28; regbase.h:13):
    y[t][i] = x1[t][i] + scalars[t] * (x2[t][i] / x3[t][i])   (t = 0..n-1, per list element)

Non-inplace: output y is a separate TensorList of n sub-tensors, same shape/dtype as x1.
Output order == x1 sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened, in CSV input_shapes order):
    x1_list : list of numpy arrays  (DYNAMIC TensorList x1, n sub-tensors)
    x2_list : list of numpy arrays  (DYNAMIC TensorList x2, sync with x1)
    x3_list : list of numpy arrays  (DYNAMIC TensorList x3, sync with x1)
    scalars : numpy array, shape (n,)  (one scalar coefficient per list element, docs:113)

计算实现改用竞品 torch._foreach_* (红线 R3: golden 只能是竞品接口实现或竞品算子拼接实现,
禁 numpy 纯公式), numpy 仅保留 I/O 与 dtype 转换。这里用 _foreach_div + _foreach_mul +
_foreach_add 拼接而不是 _foreach_addcdiv: 后者按 (s*x2)/x3 结合, 与公式的
s*(x2/x3) 结合次序不同(fp32 下实测有 1ULP 级偏差); 拼接式与改造前的 golden 逐位一致。
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

    TTK 在判据是 cross_check 时会把 golden 切 golden_mode=Promote，按 DTYPE_PROMOTE_MAP
    把输入抬一档下发(fp32->float64, fp16/bf16->float32)，就是为了让 golden 成为
    精度标准 §4.5「双标杆比对」要求的「**更高精度**的 CPU 实现为真值」。

    此处原先无条件 `astype(float32)`，把 Promote 抬上来的 fp64 又降回 fp32，
    于是 golden 与三方腿(fp32 GPU，同一串 torch._foreach_* 、同一结合序)在 IEEE-754 下
    **逐位相等** —— 实测 ttk-compare.log 3200/3200 条 `|b-g| = 0.000000e+00`。
    双标杆塌成单标杆后 GPU 误差恒 0，三个比值的分母全部夹到 err(§4.5.1)，
    其中 MARE/MERE 分子是无量纲相对误差、仍然很小，唯独 RMSE 分子是**有量纲**绝对误差，
    比值随输出量级线性放大(x2_extreme 档实测 rmse 比值 5.8e34，而 NPU 自身误差只有 1.07 ULP)。

    非 Promote 的判据(stat_rel_err 等)下发的仍是原 dtype，golden 照旧贴着算子计算精度，
    行为不变。唯一必须转换的是 bf16：numpy/torch 不能直接在 |V2 视图或 ml_dtypes.bfloat16
    上做运算，先 view 回 bf16 再抬 fp32。
    """
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    if _bf16 is not None and a.dtype == _bf16:
        return a.astype(np.float32)
    # dtype 无需转换时复用原缓冲: 大张量(单份 GB 级)无谓复制会把进程推向 OOM。
    # 下游 torch 算子均非原地，不会改写输入，故复用安全。
    return a


def _to_fp32(a):
    """三方腿/标量沿用的 fp32 口径（三方必须按**算子自身精度**算，不能跟着 golden 抬档）。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _bf16 is not None:
        a = a.view(_bf16)
    return a.astype(np.float32, copy=False)


_F32_TINY = torch.tensor(float(np.finfo(np.float32).tiny), dtype=torch.float32)


def _ftz(x):
    """把落入 fp32 Subnormal 区间的结果清零，对齐算子两代共同的行为。

    A2(910B) 的 Div 没有 config 参数(asc-devkit Div.md 里带 config 的原型对 Atlas A2 标注
    "不支持")，只有单指令一条路，Subnormal 必然 FTZ；arch35 的 Vec::Div 用默认
    DivConfig{DivAlgo::INTRINSIC}，文档写明该档"Subnormal 均被 FTZ"。CPU 默认保留
    Subnormal，故显式补这一步。"""
    return torch.where((x != 0) & (x.abs() < _F32_TINY), torch.zeros_like(x), x)


def __golden_foreach_addcdiv_list(x1_list, x2_list, x3_list, scalars, **kwargs):
    output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))

    # scalars 同样跟随计算精度: Promote 下 fp64 真值不能被一个先降到 fp32 的标量污染。
    scalars_arr = np.asarray(_to_compute(scalars)).reshape(-1)

    results = []
    for i, (a, b, c) in enumerate(zip(x1_list, x2_list, x3_list)):
        ta = torch.from_numpy(_to_compute(a))
        tb = torch.from_numpy(_to_compute(b))
        tc = torch.from_numpy(_to_compute(c))
        s = scalars_arr[i] if i < scalars_arr.size else scalars_arr[-1]
        ts = torch.as_tensor(s).to(ta.dtype)

        # y_i = x1_i + scalars[i] * (x2_i / x3_i)
        quot = [_ftz(q) for q in torch._foreach_div([tb], [tc])]
        out = torch._foreach_add([ta], torch._foreach_mul(quot, ts))[0].numpy()

        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)
        if target == "bfloat16":
            out = out.astype(_bf16) if _bf16 is not None else out
        else:
            out = out.astype(target)
        results.append(out)
    return results


__golden__ = {"kernel": {"foreach_addcdiv_list": "__golden_foreach_addcdiv_list"}}

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


_GOLDEN_FN = __golden_foreach_addcdiv_list


class _ForeachAddcdivListCompose:
    def __call__(self, x1, x2, x3, scalars, **kwargs):
        st = (
            scalars
            if isinstance(scalars, torch.Tensor)
            else torch.as_tensor(np.asarray(scalars))
        )
        a, b, c = _tp_list(x1), _tp_list(x2), _tp_list(x3)
        sc = _tp_scalars_t(st, a[0])
        return torch._foreach_addcdiv(a, b, c, sc)


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
    _INNER = _ForeachAddcdivListCompose

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
        _ForeachAddcdivListCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class ForeachAddcdivListKernelSpec:
    golden = _GOLDEN_FN
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {
    "foreach_addcdiv_list": "ForeachAddcdivListKernelSpec",
    "aclnnForeachAddcdivList": "ForeachAddcdivListAclnnSpec",
    "torch._foreach_addcdiv": "ForeachAddcdivListTorchSpec",
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


def _golden_one(t):
    """CPU golden: 低精度浮点使用 fp32 中间量，整型和 fp64 保持原精度。"""
    t = _tp_one(t).detach().cpu()
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


def _tp_scalars(v):
    if isinstance(v, torch.Tensor):
        return [float(x) for x in v.detach().cpu().reshape(-1)]
    return [float(x) for x in v]


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


class ForeachAddcdivListAclnnSpec:
    """aclnn 通路 spec。golden 收设备侧 torch.Tensor(README: ACLNN 传入已 H2D 的
    torch.Tensor), 由 TTK 按 aclnn 头文件形参**位置**下发(AclnnParamPlan.build_args),
    故签名逐项对齐 aclnnForeachAddcdivListGetWorkspaceSize 的形参;
    third_party 走按名绑定(pool 的 key 取自头文件形参名), 复用 kernel 通路的竞品类
    ——其形参名即 def 注册名, 与头文件一致。"""

    @staticmethod
    def golden(x1, x2, x3, scalars, out=None, **kwargs):
        a = [_golden_one(t) for t in x1]
        b = [_golden_one(t) for t in x2]
        c = [_golden_one(t) for t in x3]
        sc = _tp_scalars(scalars)
        # 两步拼接: 先 div 再按标量乘, 最后相加, 不用 addcdiv 的融合舍入
        return _keep_dtype(
            torch._foreach_add(a, torch._foreach_mul(torch._foreach_div(b, c), sc)), x1
        )

    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


class ForeachAddcdivListTorchSpec:
    """E2E Tensor/ScalarList overload: 无 ACLNN out 参数，返回 TensorList。"""

    @staticmethod
    def golden(self, tensor1, tensor2, scalars, **kwargs):
        return ForeachAddcdivListAclnnSpec.golden(
            self, tensor1, tensor2, scalars, **kwargs
        )
