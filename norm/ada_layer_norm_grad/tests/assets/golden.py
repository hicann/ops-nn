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

"""AdaLayerNormGrad golden（TTK kernel / GEIR 模式）。

按算法语义在 CPU 上用 torch 拼接真值（与 04_golden/AdaLayerNormGrad_golden.py 的
`_compute` 同源），不照抄内核的步骤划分。
全部输入先统一 cast 到同一计算精度再计算（cast 后整体升档，不混合）：
默认 fp32（与算子内部 fp32 累加一致）；开启 --golden-mode Promote 时
（框架把 fp32 入参升为 fp64，据此探测）整体用 fp64 算高精度真值。
输出再按 kwargs["output_dtypes"] 逐输出 cast 回目标 dtype。

注册关系（GEIR 模式按 op_name 复用 kernel 条目，框架无独立 geir 键）：
- kernel/GEIR : op_name = ada_layer_norm_grad （框架规定 numpy 入/出参）

参数名与顺序取自 op_host/ada_layer_norm_grad_def.cpp，本算子无属性：
  输入（7）: dy, x, rstd, mean, scale, gamma, beta
  输出（5）: pd_x, pd_scale, pd_shift, pd_gamma, pd_beta

shape 约定（见 README）：
  dy / x        : [B..., S, H]
  rstd / mean   : [B..., S, 1]（恒 fp32）
  scale         : [B..., H] 或 [B..., 1, H]
  gamma / beta  : [H]（F16F32/BF16F32 时 gamma/beta 为 fp32）
  pd_x          : [B..., S, H]
  pd_scale/shift: 同 scale shape
  pd_gamma/beta : [H]

计算公式（与 README、01_requirement.md §2 一致，N = H）：
    y        = (x - mean) * rstd
    dy_g     = dy * gamma * (1 + scale_b)            # scale_b = scale 广播为 [B..., 1, H]
    temp1    = sum_H(dy_g) / N
    temp2    = sum_H(dy_g * y) / N
    pd_x     = (dy_g - temp1 - y * temp2) * rstd
    pd_scale = sum_S(dy * (gamma * y + beta))
    pd_shift = sum_S(dy)
    pd_gamma = sum_{B...,S}(dy * (1 + scale_b) * y)
    pd_beta  = sum_{B...,S}(dy * (1 + scale_b))
"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401  导入后 numpy 才识别 "bfloat16" dtype 名
except ImportError:  # pragma: no cover
    ml_dtypes = None

__golden__ = {
    "kernel": {"ada_layer_norm_grad": "ada_layer_norm_grad_golden"},
}


# ---------------------------------------------------------------------------
# 小工具（每个只做一件事）
# ---------------------------------------------------------------------------
def _np_to_torch(array):
    """kernel/GEIR 的 numpy 入参 → torch CPU 张量。

    bf16 特殊处理：ttk 用 ml_dtypes.bfloat16 表示 bf16 numpy 数组，
    torch.from_numpy 不认识它，需经 uint16 位视图无损转换。
    """
    array = np.ascontiguousarray(array)
    if ml_dtypes is not None and array.dtype == np.dtype(ml_dtypes.bfloat16):
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _compute_dtype(*tensors):
    """计算精度：任一入参为 fp64 即 Promote 模式（框架把 fp32 升到了 fp64），
    整体用 fp64 算高精度真值；否则整体 fp32（对齐算子内部 fp32 累加）。"""
    for tensor in tensors:
        if tensor.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _torch_to_numpy(tensor, dtype_name):
    """计算结果 → numpy 并 cast 到期望输出 dtype（bf16 依赖 ml_dtypes 注册的 dtype 名）。"""
    array = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(array.astype(dtype_name, copy=False))


def _output_dtype_names(kwargs):
    """用例声明的输出 dtype 名列表，兼容 ["float16", ...] 与 [("float16", "ND"), ...] 两种形态。"""
    names = []
    for dtype in kwargs.get("output_dtypes") or ():
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        names.append(str(dtype))
    return names


# ---------------------------------------------------------------------------
# kernel / GEIR golden（numpy 入/出参，框架规定）
# ---------------------------------------------------------------------------
def ada_layer_norm_grad_golden(dy, x, rstd, mean, scale, gamma, beta, **kwargs):
    """
    Kernel/GEIR golden for ada_layer_norm_grad.
    参数名与顺序跟随 ada_layer_norm_grad_def.cpp 输入（不含输出），本算子无属性。

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [pd_x, pd_scale, pd_shift, pd_gamma, pd_beta]（numpy），五个输出恒必需（无属性）。
    """
    # ---- 1. 异常校验 ----
    for name, array in (
        ("dy", dy),
        ("x", x),
        ("rstd", rstd),
        ("mean", mean),
        ("scale", scale),
        ("gamma", gamma),
        ("beta", beta),
    ):
        if array is None:
            raise ValueError(f"input {name} must not be None (all inputs are REQUIRED)")

    # ---- 2. 前处理：numpy → torch，统一 cast 到计算精度 ----
    dy_raw = _np_to_torch(dy)
    x_raw = _np_to_torch(x)
    rstd_raw = _np_to_torch(rstd)
    mean_raw = _np_to_torch(mean)
    scale_raw = _np_to_torch(scale)
    gamma_raw = _np_to_torch(gamma)
    beta_raw = _np_to_torch(beta)

    compute = _compute_dtype(
        dy_raw, x_raw, rstd_raw, mean_raw, scale_raw, gamma_raw, beta_raw
    )
    dy_t = dy_raw.to(compute)
    x_t = x_raw.to(compute)
    rstd_t = rstd_raw.to(compute)
    mean_t = mean_raw.to(compute)
    scale_t = scale_raw.to(compute)
    gamma_t = gamma_raw.to(compute)
    beta_t = beta_raw.to(compute)

    # ---- 3. 计算：按算法语义拼接（N = H） ----
    n = x_t.shape[-1]
    # scale 形如 [B..., H] 或 [B..., 1, H]：统一 reshape 成 [B..., 1, H] 与 dy 沿 S 广播
    scale_b = scale_t.reshape(x_t.shape[:-2] + (1, n))
    y = (x_t - mean_t) * rstd_t
    dy_g = dy_t * gamma_t * (1 + scale_b)
    temp1 = dy_g.sum(-1, keepdim=True) / n
    temp2 = (dy_g * y).sum(-1, keepdim=True) / n
    pd_x = ((dy_g - temp1 - y * temp2) * rstd_t).reshape(x_t.shape)

    # sum_S（axis=-2）→ [B..., H]，再 reshape 回 scale 形状
    pd_scale = (dy_t * (gamma_t * y + beta_t)).sum(-2).reshape(scale_t.shape)
    pd_shift = dy_t.sum(-2).reshape(scale_t.shape)
    # sum_{B..., S} → [H]，reshape 回 gamma/beta 形状
    reduce_dims = tuple(range(x_t.ndim - 1))
    pd_gamma = (dy_t * (1 + scale_b) * y).sum(reduce_dims).reshape(gamma_t.shape)
    pd_beta = (dy_t * (1 + scale_b)).sum(reduce_dims).reshape(beta_t.shape)

    # ---- 4. 后处理：cast 到用例声明的输出 dtype → numpy ----
    out_dtypes = _output_dtype_names(kwargs)
    results = (pd_x, pd_scale, pd_shift, pd_gamma, pd_beta)
    outputs = []
    for i, result in enumerate(results):
        dtype_name = out_dtypes[i] if i < len(out_dtypes) else "float32"
        outputs.append(_torch_to_numpy(result, dtype_name))
    return outputs
