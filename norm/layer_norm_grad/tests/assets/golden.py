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

"""LayerNormGrad golden（TTK kernel / GEIR 模式）。

真值通路为 CPU 上的 torch.ops.aten.native_layer_norm_backward。
全部输入先统一 cast 到同一计算精度再计算（cast 后整体升档，不混合）：
默认 fp32（与算子内部 fp32 累加一致）；开启 --golden-mode Promote 时
（框架把 fp32 入参升为 fp64，据此探测）整体用 fp64 算高精度真值。
输出再 cast 回目标 dtype。

注册关系（GEIR 模式按 op_name 复用 kernel 条目，框架无独立 geir 键）：
- kernel/GEIR : op_name = layer_norm_grad （框架规定 numpy 入/出参）

本算子无 aclnn API（仓内无 op_api），e2e 的 aten 接口在 NPU 侧由
LayerNormGradV3 通路承接，故仅提供 kernel/GEIR 真值。

计算公式（与 README 一致）：
    rstd     = 1 / sqrt(variance + epsilon)
             epsilon 与 tiling 保持一致（layer_norm_grad_tiling_base.cpp）：
             原始 x 为 FLOAT 时 1e-12，否则 1e-5
    x_hat    = (x - mean) * rstd
    dy_gamma = dy * gamma
    pd_x     = (dy_gamma - (mean_N(dy_gamma) + x_hat * mean_N(dy_gamma * x_hat))) * rstd
    pd_gamma = sum_M(dy * x_hat)
    pd_beta  = sum_M(dy)
"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401  导入后 numpy 才识别 "bfloat16" dtype 名
except ImportError:  # pragma: no cover
    ml_dtypes = None

__golden__ = {
    "kernel": {"layer_norm_grad": "layer_norm_grad_golden"},
}

# epsilon 取值与 tiling 一致（layer_norm_grad_tiling_base.cpp）：x 为 FLOAT 时 1e-12，否则 1e-5
_EPSILON_FP32 = 1e-12
_EPSILON_HALF = 1e-5


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


# ---------------------------------------------------------------------------
# kernel / GEIR golden（numpy 入/出参，框架规定）
# ---------------------------------------------------------------------------
def layer_norm_grad_golden(dy, x, variance, mean, gamma, **kwargs):
    """
    Kernel/GEIR golden for layer_norm_grad.
    参数名与顺序跟随 layer_norm_grad_def.cpp 输入（不含输出），本算子无属性。

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [pd_x, pd_gamma, pd_beta]（numpy），三个输出恒必需（无 output_mask 属性）。
    """
    # ---- 1. 异常校验 ----
    for name, array in (
        ("dy", dy),
        ("x", x),
        ("variance", variance),
        ("mean", mean),
        ("gamma", gamma),
    ):
        if array is None:
            raise ValueError(f"input {name} must not be None (all inputs are REQUIRED)")

    # ---- 2. 前处理：numpy → torch，variance → rstd，统一 cast 到计算精度 ----
    dy_raw = _np_to_torch(dy)
    x_raw = _np_to_torch(x)
    variance_raw = _np_to_torch(variance)
    mean_raw = _np_to_torch(mean)
    gamma_raw = _np_to_torch(gamma)
    # epsilon 与 tiling 一致：原始 x 为 FLOAT 取 1e-12，否则 1e-5。须在 cast 前判断：
    # Promote 下 x/variance 同步升档，两者 dtype 相等即原始 x 为 FLOAT
    epsilon = _EPSILON_FP32 if x_raw.dtype == variance_raw.dtype else _EPSILON_HALF

    compute = _compute_dtype(dy_raw, x_raw, variance_raw, mean_raw, gamma_raw)
    dy_t = dy_raw.to(compute)
    x_t = x_raw.to(compute)
    variance_t = variance_raw.to(compute)
    mean_t = mean_raw.to(compute)
    gamma_t = gamma_raw.to(compute)
    rstd_t = 1.0 / torch.sqrt(variance_t + epsilon)
    # normalized_shape 即 gamma 的 shape（gamma 恒为 norm 轴 [R1..Rj]）
    normalized_shape = list(gamma_t.shape)

    # ---- 3. 计算：aten 真值 ----
    # aten 不接受 None bias，传零张量
    bias_t = torch.zeros_like(gamma_t)
    pd_x, pd_gamma, pd_beta = torch.ops.aten.native_layer_norm_backward(
        dy_t, x_t, normalized_shape, mean_t, rstd_t, gamma_t, bias_t, [True, True, True]
    )

    # ---- 4. 后处理：cast 到用例声明的输出 dtype → numpy ----
    out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
    outputs = []
    for i, result in enumerate((pd_x, pd_gamma, pd_beta)):
        if i < len(out_dtypes):
            outputs.append(_torch_to_numpy(result, out_dtypes[i]))
        else:
            outputs.append(_torch_to_numpy(result, "float32"))
    return outputs
