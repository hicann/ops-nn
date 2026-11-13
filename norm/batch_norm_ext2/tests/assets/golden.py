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
"""
TTK golden plugin for batch_norm_ext2 in the TestSpec multi-path format.

**Implemented with torch tensor ops** (torch.mean / torch.rsqrt / torch.mul + torch.add two-step),
not hand-written numpy formulas — the semantics were validated against tf.raw_ops.FusedBatchNormV2 on GPU:
  training : output_mean = batch mean (biased), output_variance = unbiased var (Bessel),
             reserve_space_1 = batch mean, reserve_space_2 = biased var E[(x-mean)^2]
  inference: y = scale*(x-mean)*rsqrt(var+eps)+offset; the 4 stats outputs equal the
             input mean/variance (CANN compatibility behavior).

Reference for IO/attr semantics:
    ops-nn/norm/batch_norm_ext2/op_host/batch_norm_ext2_def.cpp
    ops-nn/norm/batch_norm_ext2/op_graph/batch_norm_ext2_proto.h

Canonical IO order:
    inputs : input_x(fp16|fp32, NCHW/NHWC 4D), input_scale(fp32, C), input_offset(fp32, C),
             input_mean(fp32, C, optional), input_variance(fp32, C, optional)
    outputs: output_y(like input_x), output_mean(fp32,C), output_variance(fp32,C),
             output_reserve_space_1(fp32,C), output_reserve_space_2(fp32,C)
    attrs  : epsilon(float=1e-4), data_format(str="NHWC"), is_training(bool=True)

TTK passes input arrays positionally in CSV input_shapes order; attributes via **kwargs.
In training mode input_mean/input_variance are ignored (may be None or dummy data).
"""

# Kernel golden — the legacy kernel golden loader resolves the snake-case operator key here.
__golden__ = {
    "kernel": {"batch_norm_ext2": "_golden_batch_norm_ext2"},
}

# TestSpec multi-path registration: kernel/GEIR share one Spec.
__spec__ = {"batch_norm_ext2": "BatchNormExt2KernelSpec"}

import numpy as np
import torch


_TOL = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def _as_bool(v, default):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return bool(v)


def _reference_dtype(*arrays):
    """Compute dtype: at least fp32; keep any wider float (fp64 under cross_check Promote)."""
    dtype = np.float32
    for arr in arrays:
        if arr is not None:
            dtype = np.promote_types(dtype, arr.dtype)
    return dtype


def _normalize_dtype(dt):
    if dt is None:
        return None
    name = str(dt).lower().split(".")[-1].rstrip("'>\"")
    return name


# ---------------------------------------------------------------------------
# 出口 dtype：把 ttk 下发的 dtype 还原成用例**声明**的存储 dtype
# ---------------------------------------------------------------------------
# ttk 在 golden_mode=Promote（cross_check 判据）下按 DTYPE_PROMOTE_MAP 把声明 dtype
# 各抬一档后才下发，所以 kernel/geir 通路 kwargs["output_dtypes"] 里是**抬升后**的
# dtype，不是用例声明值（kwargs 里没有 ori_dtypes / tensor_dtypes 可直取）。
# 本算子 input_x / output_y 的声明域只有 {float16, float32}
# （op_graph/batch_norm_ext2_proto.h:60、:65），Promote 后：
#   声明 float16 -> 下发 float32；声明 float32 -> 下发 float64
# 反推唯一（f16 与 bf16 会同为 float32 的歧义在此算子不存在：proto 不接受 bfloat16）。
_PROMOTED_TO_DECLARED = {"float32": "float16", "float64": "float32"}
# 只有比真值计算精度更窄的声明类型才需要在出口取整（fp16 量化台阶）。
# 声明 float32 的用例在 Promote 下真值是 fp64，一律保持原样（与改动前逐位一致）。
_NARROW_OUT_DTYPES = ("float16", "bfloat16")


def _first_dtype(dt):
    """取 TensorList 嵌套声明里的首个 dtype；标量/字符串走 _normalize_dtype。"""
    while isinstance(dt, (tuple, list)):
        if not dt:
            return None
        dt = dt[0]
    return _normalize_dtype(dt)


def _declared_output_dtype(dt, kwargs):
    """把 ttk 下发的输出 dtype 还原成用例声明的存储 dtype。

    Promote 下按 _PROMOTED_TO_DECLARED 反推；非 Promote 下下发值本就是声明值，
    原样返回（该分支与改动前行为完全一致）。
    """
    name = _first_dtype(dt)
    if name is None:
        return None
    if str(kwargs.get("golden_mode", "")).lower() == "promote":
        return _PROMOTED_TO_DECLARED.get(name, name)
    return name


def _spatial_mean_var(xt, data_format):
    """Per-channel mean/biased-var over the spatial dims via a single flattened axis.

    torch/numpy multi-axis reductions (e.g. dim=(0,2,3)) accumulate the strided leading
    axis without the accurate pairwise path, losing fp32 precision on extreme shapes
    (huge N + tiny C/HW: N=2^20, C=2, H*W=2 observed ~2.5e-4 relative error on
    output_variance). Collapsing the reduced dims into one contiguous axis keeps the
    reference accurate (matches fp64 to ~1e-7) for every shape.
    """
    if data_format == "NHWC":
        c = xt.shape[3]
        flat = xt.permute(3, 0, 1, 2).reshape(c, -1)  # (C, N*H*W)
        out_shape = (1, 1, 1, c)
    else:  # NCHW
        c = xt.shape[1]
        flat = xt.permute(1, 0, 2, 3).reshape(c, -1)  # (C, N*H*W)
        out_shape = (1, c, 1, 1)
    mean = flat.mean(dim=1, keepdim=True)
    var = ((flat - mean) ** 2).mean(dim=1, keepdim=True)
    return mean.reshape(out_shape), var.reshape(out_shape)


def _golden_batch_norm_ext2(
    input_x, input_scale, input_offset, input_mean=None, input_variance=None, **kwargs
):
    """Kernel/GEIR adapter: NumPy inputs and a NumPy output list.

    精度策略:golden 按**输入实际精度**计算,不硬编码降回 fp32。
    cross_check 三方场景 ttk 走 golden_mode=Promote,把 fp32 输入抬到 fp64
    (fp16/bf16→fp32)再喂给 golden——若这里再 astype(float32) 会把框架抬上去的
    精度砍回来,使标杆自身带 fp32 误差、污染三方 mare 比值(大归约下 ~1e-3 相对误差)。
    最后按 ttk 下发的 output_dtypes 返回(Enable 模式→原精度,Promote 模式→抬升后精度)。
    """
    eps = float(_attr(kwargs, "eps", _attr(kwargs, "epsilon", 1e-4)))
    is_training = _as_bool(_attr(kwargs, "is_training", True), True)

    x = np.asarray(input_x)
    scale = np.asarray(input_scale).reshape(-1)
    offset = np.asarray(input_offset).reshape(-1)
    mean_in = np.asarray(input_mean).reshape(-1) if input_mean is not None else None
    var_in = (
        np.asarray(input_variance).reshape(-1) if input_variance is not None else None
    )

    data_format = str(_attr(kwargs, "data_format", "NHWC")).upper()
    # 严格校验(与 tiling 一致,不做启发式 C 轴推断):data_format 必须为 NCHW/NHWC。
    # 默认值取 def 的 data_format 默认 "NHWC",使 golden 可被 TTK 直接调用(不显式传该属性时也能跑)。
    if data_format not in ("NCHW", "NHWC"):
        raise ValueError(f"data_format must be NCHW/NHWC, got {data_format!r}")

    # 确认格式后,C 轴长度必须与其余输入一致(scale/offset 必选,mean/var 可选)。
    if x.ndim != 4:
        raise ValueError(f"input_x must be 4D, got ndim={x.ndim}")
    c_len = x.shape[1] if data_format == "NCHW" else x.shape[3]
    for name, arr in (
        ("input_scale", scale),
        ("input_offset", offset),
        ("input_mean", mean_in),
        ("input_variance", var_in),
    ):
        if arr is not None and arr.size != c_len:
            raise ValueError(
                f"{name} length {arr.size} is inconsistent with data_format={data_format} "
                f"C axis length {c_len}"
            )

    compute_dtype = _reference_dtype(x, scale, offset, mean_in, var_in)

    if data_format == "NHWC":
        num = int(np.prod(x.shape[:3]))
    else:  # NCHW
        num = int(np.prod([x.shape[0], x.shape[2], x.shape[3]]))

    xt = torch.from_numpy(x.astype(compute_dtype))
    st = torch.from_numpy(scale.astype(compute_dtype))
    ot = torch.from_numpy(offset.astype(compute_dtype))

    if is_training:
        # batch mean / biased var over spatial dims, per channel (single flattened
        # reduce axis; multi-axis dim=(0,2,3) loses fp32 precision on extreme shapes)
        mean_t, var_t = _spatial_mean_var(xt, data_format)
        rstd_t = torch.rsqrt(var_t + eps)
        # y = scale * (x - mean) * rstd + offset  (competing-op composition, same algorithm)
        # 两步拼接(禁 addcmul 这类 FMA 融合算子):先逐元素乘、再逐个加,
        # 保持运算序列可分离(golden_standard.md §三 / 旧模板「禁用融合算子」条款)。
        bias_t = (
            ot.reshape(1, 1, 1, -1)
            if data_format == "NHWC"
            else ot.reshape(1, -1, 1, 1)
        )
        scale_t = (
            st.reshape(1, 1, 1, -1)
            if data_format == "NHWC"
            else st.reshape(1, -1, 1, 1)
        )
        # 逐元素乘 + 逐个加两步拼接(禁 addcmul 这类 FMA 融合算子)。
        # 逐元素乘之前先把按通道的两个因子合成一次:y = (x-mean)*(rstd*scale)+offset,
        # 与 (x-mean)*rstd*scale 数学等价(fp64 下差异 <= 数个 ulp),但逐元素乘法
        # 由两次并为一次——scale 极小时 (x-mean)*rstd 的假 inf 不再出现。
        k_t = rstd_t * scale_t
        y_t = torch.add(torch.mul(xt - mean_t, k_t), bias_t)
        batch_mean = mean_t.reshape(-1).numpy()
        batch_var_t = var_t.reshape(-1)
        batch_var = batch_var_t.numpy()  # 有偏方差，给 out_rs2
        # output_variance is unbiased (Bessel correction) — same as TF FusedBatchNormV2
        unbiased_var = (
            batch_var_t * (num / (num - 1)) if num > 1 else batch_var_t * float("inf")
        ).numpy()
        out_mean = batch_mean
        out_variance = unbiased_var
        out_rs1 = batch_mean
        # reserve_space_2 是**有偏方差** E[(x-mean)^2]（= batch_var，已经是无贝塞尔校正的 M2/n），
        # 与 output_variance（无偏）不同；不能用 rstd。
        out_rs2 = batch_var
    else:
        mt = torch.from_numpy(mean_in.astype(compute_dtype))
        vt = torch.from_numpy(var_in.astype(compute_dtype))
        rstd_t = torch.rsqrt(vt + eps)
        if data_format == "NHWC":
            bias_t = ot.reshape(1, 1, 1, -1)
            scale_t = st.reshape(1, 1, 1, -1)
            centered = (xt - mt.reshape(1, 1, 1, -1)) * rstd_t.reshape(1, 1, 1, -1)
        else:  # NCHW
            bias_t = ot.reshape(1, -1, 1, 1)
            scale_t = st.reshape(1, -1, 1, 1)
            centered = (xt - mt.reshape(1, -1, 1, 1)) * rstd_t.reshape(1, -1, 1, 1)
        # 两步拼接(禁 addcmul/FMA),同训练分支。
        y_t = torch.add(torch.mul(centered, scale_t), bias_t)
        out_mean = mean_in
        out_variance = var_in
        out_rs1 = mean_in
        out_rs2 = var_in

    # Cast outputs to ttk-expected dtypes (Enable→原精度, Promote→抬升后精度)。
    # 唯一例外：output_y（out0）再按**用例声明**的存储精度取整——真值计算过程
    # （含 Promote 抬上来的 fp64）一律不动，只把最后这一步舍入落到算子真正交付的
    # dtype 上。否则 fp16 用例的 golden 停在 fp32，与任何 fp16 值都不相等，
    # fp16 量化台阶会被整份记成被测误差（cross_check 的 mare 分子）。
    out_dtypes = kwargs.get("output_dtypes")
    if not out_dtypes:
        out_dtypes = (
            _normalize_dtype(x.dtype),
            "float32",
            "float32",
            "float32",
            "float32",
        )
    declared_y_dtype = _declared_output_dtype(
        out_dtypes[0] if out_dtypes else None, kwargs
    )
    raw_outs = [y_t.numpy(), out_mean, out_variance, out_rs1, out_rs2]
    result = []
    for i, arr in enumerate(raw_outs):
        dt = _first_dtype(out_dtypes[i]) if i < len(out_dtypes) else None
        if i == 0 and declared_y_dtype in _NARROW_OUT_DTYPES:
            dt = declared_y_dtype
            # 先把真值落到算子**实际计算精度** fp32（Promote 下真值是 fp64），再按声明
            # dtype 取整：保证 golden 的 fp16 输出 ==「原有 fp32 出口结果.astype(声明
            # dtype)」，即算子按 fp32 计算 + 按存储精度舍入能达到的最优输出。
            # fp64 直接舍到 fp16 会在 fp16 中点处落到只有更高精度才能得到的值，与任何
            # fp32 实现都对不上（实测约 1/200 的用例存在这种中点，1 ulp 差）。
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
        result.append(
            np.ascontiguousarray(arr.astype(np.dtype(dt)))
            if dt is not None
            else np.ascontiguousarray(arr)
        )
    return result


# ---------------------------------------------------------------------------
# 内核 rstd 段（Newton-Raphson）的 fp32 逐指令复刻
#   来源：ops-nn/norm/batch_norm_ext2/op_kernel/arch35/batch_norm_ext2_base.h:164-215
#         ComputeRstdNewtonRaphsonRegExt2
#   注意：内核用的是两次 NR 迭代的**近似** rstd，不是精确 rsqrt。三方 compose 若仍用
#         torch.rsqrt，大 var 区间会与内核差出若干 ULP（内核自己也承认这是近似）。
#   训练路径的 6 处调用全部是 ComputeRstdNewtonRaphsonRegExt2<false> / ...Ext2<false>
#   （无 Maxs），推理路径的 4 处用默认模板参数 <true>（有 Maxs(v, -99.99)）。
# ---------------------------------------------------------------------------
_POS_INF_LITERAL = 3.40282366920938e38  # 内核 POS_INF 的十进制字面量（见下）
# 该字面量超出 fp32 范围（FLT_MAX=3.402823466e38 < literal < 2^128），按 IEEE 就近舍入
# 上溢。用 CANN 自带编译前端（tools/bisheng_compiler/bin/bisheng）与 g++/gcc 实测，
# 三者都把该字面量折成 0x7F800000 = +inf，所以内核里 POS_INF 的运行时值就是 +inf：
#   v == +inf -> rstd = 0        （var+eps 溢出到 inf 时的兜底）
#   v == 0    -> rstd = +inf     （1/sqrt(0) 的真实极限，与数学一致）
_POS_INF = float("inf")

# 两个 Mula 的单次舍入实现方式，可在三种之间切换（离线实测见 report.md）：
#   "addcmul" : torch.addcmul(c, a, b)（设备侧单算子，NPU 上大概率融合成一条 FMA）
#   "float64" : a.double()*b.double()+c.double() 再降回 fp32（数学上等价于单次舍入，
#               但依赖 fp64，NPU 不支持，仅作离线参考）
#   "split"   : c + a*b（两次舍入，与内核的 Mula 语义不一致）
_FMA_MODE = "addcmul"


def _fma32(a, b, c):
    """fp32 融合乘加 a*b+c（单次舍入），复刻内核 Mula(dst, src0, src1)。"""
    if _FMA_MODE == "float64":
        return (a.double() * b.double() + c.double()).to(torch.float32)
    if _FMA_MODE == "split":
        return c + a * b
    return torch.addcmul(c, a, b)


def _rstd_nr_torch(var, epsilon, need_max):
    """逐指令复刻 ComputeRstdNewtonRaphsonRegExt2 的 rstd 序列（纯 torch fp32）。

    var      : fp32 张量（逐通道方差），只占 C 个元素
    epsilon  : 已按 tiling 习惯转成 fp32 的 eps（python float）
    need_max : 内核模板参数；训练路径为 False，推理路径为 True

    eager 逐算子执行（不放进 compiled 区）：既保证不会被 inductor 代数化简回 rsqrt，
    也保证每一步都是独立的 fp32 舍入点。
    """
    dev = var.device

    def _s(x):
        return torch.full((), x, dtype=torch.float32, device=dev)

    one = _s(1.0)
    v = var + _s(epsilon)  # Adds(var, var, epsilon)
    if need_max:
        v = torch.maximum(v, _s(-99.99))  # Maxs(var, var, -99.99)
    y = torch.sqrt(v)  # Sqrt(y, var)
    r = torch.div(one, y)  # Div(r, one, y)
    t = v * _s(-0.5)  # Muls(t, var, -0.5)
    t = t * r  # Mul(t, t, r)
    t1 = _fma32(t, r, _s(1.5))  # Mula(t1, t, r)
    rstd = r * t1  # Mul(rstd, r, t1)
    t4 = y * rstd  # Mul(t4, y, rstd)
    t4 = t4 * t4  # Mul(t4, t4, t4)
    t4 = t4 * _s(-1.0)  # Muls(t4, t4, -1)
    s = t4 + one  # Adds(s, t4, 1)
    s = s * rstd  # Mul(s, s, rstd)
    rstd = _fma32(s, _s(0.5), rstd)  # Mula(rstd, s, 0.5)
    # 顺序与内核一致：先 Compares(var, POS_INF) -> 0，再 Compares(var, 0) -> POS_INF
    rstd = torch.where(v == _POS_INF, _s(0.0), rstd)
    rstd = torch.where(v == 0.0, _s(_POS_INF), rstd)
    return rstd


class _BatchNormExt2Compose:
    """[probe V3] Legacy original graph for stats + NR-rstd norm for y.

    The pre-alignment golden's mean/var numerics are reproduced by running the
    **exact original `_impl` graph** (`torch.rsqrt(var+eps)` inside): inductor's
    reduction codegen depends on that graph, and every rewrite (splitting stats
    out, or swapping rsqrt for the NR sequence) perturbs it on extreme shapes
    (049/230). The `y` produced by the legacy graph is discarded; the returned
    `y` is recomputed by `_norm` with the eager NR rstd.
    """

    def __init__(self, epsilon=1e-4, **kwargs):
        eps = float(_attr(kwargs, "epsilon", epsilon))
        self.epsilon = float(torch.tensor(eps, dtype=torch.float32).item())
        self.data_format = str(_attr(kwargs, "data_format", "NHWC")).upper()
        self.is_training = _as_bool(_attr(kwargs, "is_training", True), True)
        self._compiled_legacy = None
        self._compiled_norm = None

    def _legacy_impl(
        self, input_x, input_scale, input_offset, input_mean=None, input_variance=None
    ):
        if self.data_format == "NHWC":
            axis = (0, 1, 2)
            bshape = (1, 1, 1, -1)
        else:  # NCHW
            axis = (0, 2, 3)
            bshape = (1, -1, 1, 1)
        num = 1
        for dim in (input_x.shape[i] for i in axis):
            num *= int(dim)

        # The kernel computes every arithmetic operand in fp32.
        x_f32 = input_x.to(dtype=torch.float32)
        s_f32 = input_scale.to(dtype=torch.float32).reshape(-1)
        o_f32 = input_offset.to(dtype=torch.float32).reshape(-1)
        eps_t = torch.tensor(self.epsilon, dtype=torch.float32, device=input_x.device)

        if self.is_training:
            # single flattened reduce axis — multi-axis dim=(0,2,3) loses fp32
            # precision on extreme shapes (huge N + tiny C/HW), see _spatial_mean_var
            if self.data_format == "NHWC":
                flat = x_f32.permute(3, 0, 1, 2).reshape(x_f32.shape[3], -1)
            else:  # NCHW
                flat = x_f32.permute(1, 0, 2, 3).reshape(x_f32.shape[1], -1)
            mean_flat = flat.mean(dim=1, keepdim=True)  # (C, 1)
            var_flat = ((flat - mean_flat) ** 2).mean(dim=1, keepdim=True)
            mean_t = mean_flat.reshape(bshape)
            var_t = var_flat.reshape(bshape)
            rstd_t = torch.rsqrt(var_t + eps_t)
            # 训练分支与训练 kernel 的归一化段逐项对齐:Sub→Mul(rstd)→Mul(gamma)→Add(beta),
            # 即 ((x-mean)*rstd)*scale+offset(见 batch_norm_ext2_welford.h /
            # batch_norm_ext2_rar_block_split_r.h 的 CalculateNormalizeVF)。
            # 推理分支是 scale 先乘,见下方注释;两个分支各自对齐对应的 kernel 分支。
            y = (x_f32 - mean_t) * rstd_t * s_f32.reshape(bshape) + o_f32.reshape(
                bshape
            )
            out_mean = mean_flat.reshape(-1)
            if num > 1:
                out_var = var_flat.reshape(-1) * (float(num) / (num - 1))
            else:
                out_var = var_flat.reshape(-1) * float("inf")
            out_rs1 = mean_flat.reshape(-1)
            # reserve_space_2 = 有偏方差 E[(x-mean)^2]（var_flat 本身即 M2/n），不是 rstd。
            out_rs2 = var_flat.reshape(-1)
        else:
            m_f32 = input_mean.to(dtype=torch.float32).reshape(-1)
            v_f32 = input_variance.to(dtype=torch.float32).reshape(-1)
            rstd_t = torch.rsqrt(v_f32 + eps_t)
            # 推理分支与推理 kernel 对齐(Sub→Mul(gamma)→Mul(rstd)→Add(beta),
            # 见 batch_norm_ext2_infer.h):|mean| 可达 FLT_MAX 时,先乘 rstd 会让中间量
            # ((x-m)*rstd) 先溢出为 inf,即便随后乘 <1 的 scale 本可回落 -> 竞品多出假 inf。
            y = (x_f32 - m_f32.reshape(bshape)) * s_f32.reshape(
                bshape
            ) * rstd_t.reshape(bshape) + o_f32.reshape(bshape)
            out_mean = m_f32
            out_var = v_f32
            out_rs1 = m_f32
            out_rs2 = v_f32

        # The first output is stored in x dtype; the four stat outputs are fp32.
        return [
            y.to(dtype=input_x.dtype),
            out_mean.clone(),
            out_var.clone(),
            out_rs1.clone(),
            out_rs2.clone(),
        ]

    def _norm(self, x_f32, mean_t, rstd_flat, s_f32, o_f32):
        """Normalization段：两张分支各自对齐对应 kernel 的操作顺序。"""
        if self.data_format == "NHWC":
            bshape = (1, 1, 1, -1)
        else:  # NCHW
            bshape = (1, -1, 1, 1)
        rstd_t = rstd_flat.reshape(bshape)
        if self.is_training:
            # 训练分支与训练 kernel 的归一化段逐项对齐:Sub→Mul(rstd)→Mul(gamma)→Add(beta),
            # 即 ((x-mean)*rstd)*scale+offset(见 batch_norm_ext2_welford.h /
            # batch_norm_ext2_rar_block_split_r.h 的 CalculateNormalizeVF)。
            # 推理分支是 scale 先乘,见下方注释;两个分支各自对齐对应的 kernel 分支。
            return (x_f32 - mean_t) * rstd_t * s_f32.reshape(bshape) + o_f32.reshape(
                bshape
            )
        # 推理分支与推理 kernel 对齐(Sub→Mul(gamma)→Mul(rstd)→Add(beta),
        # 见 batch_norm_ext2_infer.h):|mean| 可达 FLT_MAX 时,先乘 rstd 会让中间量
        # ((x-m)*rstd) 先溢出为 inf,即便随后乘 <1 的 scale 本可回落 -> 竞品多出假 inf。
        return (x_f32 - mean_t) * s_f32.reshape(bshape) * rstd_t + o_f32.reshape(bshape)

    def __call__(
        self,
        input_x,
        input_scale,
        input_offset,
        input_mean=None,
        input_variance=None,
        **kwargs,
    ):
        del kwargs
        bshape = (1, 1, 1, -1) if self.data_format == "NHWC" else (1, -1, 1, 1)
        if self.is_training:
            if self._compiled_legacy is None:
                try:
                    self._compiled_legacy = torch.compile(
                        self._legacy_impl, dynamic=True
                    )
                except Exception:
                    self._compiled_legacy = self._legacy_impl
            try:
                _y_rsqrt, out_mean, out_var, out_rs1, out_rs2 = self._compiled_legacy(
                    input_x, input_scale, input_offset, None, None
                )
            except Exception:
                self._compiled_legacy = self._legacy_impl
                _y_rsqrt, out_mean, out_var, out_rs1, out_rs2 = self._legacy_impl(
                    input_x, input_scale, input_offset, None, None
                )
            mean_t = out_rs1.reshape(bshape)
            rstd_flat = _rstd_nr_torch(
                out_rs2.to(dtype=torch.float32), self.epsilon, need_max=False
            )
        else:
            mean_t = input_mean.to(dtype=torch.float32).reshape(bshape)
            v_f32 = input_variance.to(dtype=torch.float32).reshape(-1)
            rstd_flat = _rstd_nr_torch(v_f32, self.epsilon, need_max=True)
            out_mean = mean_t.reshape(-1)
            out_var = v_f32
            out_rs1 = mean_t.reshape(-1)
            out_rs2 = v_f32

        x_f32 = input_x.to(dtype=torch.float32)
        s_f32 = input_scale.to(dtype=torch.float32).reshape(-1)
        o_f32 = input_offset.to(dtype=torch.float32).reshape(-1)
        if self._compiled_norm is None:
            try:
                self._compiled_norm = torch.compile(self._norm, dynamic=True)
            except Exception:
                self._compiled_norm = self._norm
        try:
            y = self._compiled_norm(x_f32, mean_t, rstd_flat, s_f32, o_f32)
        except Exception:
            self._compiled_norm = self._norm
            y = self._norm(x_f32, mean_t, rstd_flat, s_f32, o_f32)

        return [
            y.to(dtype=input_x.dtype),
            out_mean.clone(),
            out_var.clone(),
            out_rs1.clone(),
            out_rs2.clone(),
        ]


class BatchNormExt2KernelSpec:
    """Shared kernel TestSpec; parameters follow batch_norm_ext2_def.cpp."""

    golden = _golden_batch_norm_ext2
    # third_party 直接引用类名(非字符串),compose 类因此必须在本 Spec 之前定义。
    third_party = {"torch": _BatchNormExt2Compose}
    tolerance = _TOL


# 【支持的通路】(TestSpec 多通路格式;与 01_requirement.md §3.3 / 02_design.md §6 一致)
#   kernel = ✅ __golden__ 兼容旧式 kernel golden 加载;__spec__/TestSpec 供 kernel/GEIR 共用
#   geir   = ✅ 与 kernel 共用 BatchNormExt2KernelSpec(golden 收 numpy.ndarray,签名同 def.cpp)
# 【不存在的通路】
#   aclnn = ❌ op_host/CMakeLists.txt 声明 ACLNNTYPE aclnn_exclude,无 aclnnBatchNormExt2 接口(与 A2 一致)
#   e2e   = ❌ torch_npu 无 aclnnBatchNormExt2 符号,FusedBatchNorm* 走 BatchNorm(非本算子)
#   tf    = ❌ canndev tf_plugin FusedBatchNorm/FusedBatchNormV2/V3 → BatchNorm,非本算子
#   onnx  = ❌ onnx BatchNormalization → BatchNorm,非本算子
