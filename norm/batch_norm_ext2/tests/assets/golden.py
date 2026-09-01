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

**Implemented with torch tensor ops** (torch.mean / torch.rsqrt / addcmul), not hand-written
numpy formulas — the semantics were validated against tf.raw_ops.FusedBatchNormV2 on GPU:
  training : output_mean = batch mean (biased), output_variance = unbiased var (Bessel),
             reserve_space_1 = batch mean, reserve_space_2 = rstd = 1/sqrt(var+eps)
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
        y_t = torch.addcmul(
            ot.reshape(1, 1, 1, -1)
            if data_format == "NHWC"
            else ot.reshape(1, -1, 1, 1),
            (xt - mean_t) * rstd_t,
            st.reshape(1, 1, 1, -1)
            if data_format == "NHWC"
            else st.reshape(1, -1, 1, 1),
        )
        batch_mean = mean_t.reshape(-1).numpy()
        batch_var = var_t.reshape(-1).numpy()
        rstd = rstd_t.reshape(-1).numpy()
        # output_variance is unbiased (Bessel correction) — same as TF FusedBatchNormV2
        unbiased_var = (
            batch_var * (num / (num - 1)) if num > 1 else batch_var * float("inf")
        )
        out_mean = batch_mean
        out_variance = unbiased_var
        out_rs1 = batch_mean
        out_rs2 = rstd
    else:
        mt = torch.from_numpy(mean_in.astype(compute_dtype))
        vt = torch.from_numpy(var_in.astype(compute_dtype))
        rstd_t = torch.rsqrt(vt + eps)
        if data_format == "NHWC":
            y_t = torch.addcmul(
                ot.reshape(1, 1, 1, -1),
                (xt - mt.reshape(1, 1, 1, -1)) * rstd_t.reshape(1, 1, 1, -1),
                st.reshape(1, 1, 1, -1),
            )
        else:  # NCHW
            y_t = torch.addcmul(
                ot.reshape(1, -1, 1, 1),
                (xt - mt.reshape(1, -1, 1, 1)) * rstd_t.reshape(1, -1, 1, 1),
                st.reshape(1, -1, 1, 1),
            )
        out_mean = mean_in
        out_variance = var_in
        out_rs1 = mean_in
        out_rs2 = var_in

    # Cast outputs to ttk-expected dtypes (Enable→原精度, Promote→抬升后精度)。
    out_dtypes = kwargs.get("output_dtypes")
    if not out_dtypes:
        out_dtypes = (
            _normalize_dtype(x.dtype),
            "float32",
            "float32",
            "float32",
            "float32",
        )
    raw_outs = [y_t.numpy(), out_mean, out_variance, out_rs1, out_rs2]
    result = []
    for i, arr in enumerate(raw_outs):
        dt = _normalize_dtype(out_dtypes[i]) if i < len(out_dtypes) else None
        result.append(
            np.ascontiguousarray(arr.astype(np.dtype(dt)))
            if dt is not None
            else np.ascontiguousarray(arr)
        )
    return result


class _BatchNormExt2Compose:
    """Independent Torch composition matching the arch35 kernel's fp32 arithmetic.

    性能腿按竞品最优形态执行:torch.compile(dynamic=True) 融合,编译失败自动回落 eager
    (三方性能倍数不虚高的关键)。
    """

    def __init__(self, epsilon=1e-4, **kwargs):
        eps = float(_attr(kwargs, "epsilon", epsilon))
        # Tiling stores epsilon as float32 before kernel launch.
        self.epsilon = float(torch.tensor(eps, dtype=torch.float32).item())
        self.data_format = str(_attr(kwargs, "data_format", "NHWC")).upper()
        self.is_training = _as_bool(_attr(kwargs, "is_training", True), True)
        self._compiled = None

    def _impl(
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
            y = (x_f32 - mean_t) * rstd_t * s_f32.reshape(bshape) + o_f32.reshape(
                bshape
            )
            out_mean = mean_flat.reshape(-1)
            if num > 1:
                out_var = var_flat.reshape(-1) * (float(num) / (num - 1))
            else:
                out_var = var_flat.reshape(-1) * float("inf")
            out_rs1 = mean_flat.reshape(-1)
            out_rs2 = rstd_t.reshape(-1)
        else:
            m_f32 = input_mean.to(dtype=torch.float32).reshape(-1)
            v_f32 = input_variance.to(dtype=torch.float32).reshape(-1)
            rstd_t = torch.rsqrt(v_f32 + eps_t)
            y = (x_f32 - m_f32.reshape(bshape)) * rstd_t.reshape(
                bshape
            ) * s_f32.reshape(bshape) + o_f32.reshape(bshape)
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
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl, dynamic=True)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(
                input_x, input_scale, input_offset, input_mean, input_variance
            )
        except Exception:
            self._compiled = self._impl
            return self._impl(
                input_x, input_scale, input_offset, input_mean, input_variance
            )


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
