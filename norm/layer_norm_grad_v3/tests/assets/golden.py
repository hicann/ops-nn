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

"""LayerNormGradV3 golden（TTK 四模式：kernel / GEIR / aclnn / e2e）。

真值通路统一为 CPU 上的 torch.ops.aten.native_layer_norm_backward。
全部输入先统一 cast 到同一计算精度再计算（cast 后整体升档，不混合）：
默认 fp32（与算子内部 fp32 累加一致）；开启 --golden-mode Promote 时
（框架把 fp32 入参升为 fp64，据此探测）整体用 fp64 算高精度真值。
输出再 cast 回目标 dtype。

注册关系（GEIR 模式按 op_name 复用 kernel 条目，框架无独立 geir 键）：
- kernel/GEIR : op_name  = layer_norm_grad_v3                 （框架规定 numpy 入/出参）
- aclnn       : api_name = aclnnLayerNormBackward             （torch CPU 张量）
- e2e         : api_name = torch.ops.aten.native_layer_norm_backward（torch 张量）

计算公式（与 README 一致）：
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
    "kernel": {"layer_norm_grad_v3": "layer_norm_grad_v3_golden"},
    "aclnn": {"aclnnLayerNormBackward": "aclnn_layer_norm_backward_golden"},
    "e2e": {
        "torch.ops.aten.native_layer_norm_backward": "aten_native_layer_norm_backward_golden"
    },
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


def _check_mask(mask):
    """output_mask 校验并归一化为 3 个 bool。"""
    if mask is None:
        return [True, True, True]
    mask = [bool(v) for v in mask]
    if len(mask) != 3:
        raise ValueError(f"output_mask length must be 3, got {len(mask)}")
    return mask


# ---------------------------------------------------------------------------
# kernel / GEIR golden（numpy 入/出参，框架规定）
# ---------------------------------------------------------------------------
def layer_norm_grad_v3_golden(dy, x, rstd, mean, gamma, output_mask=None, **kwargs):
    """
    Kernel/GEIR golden for layer_norm_grad_v3.
    参数名与顺序跟随 layer_norm_grad_v3_def.cpp 输入（不含输出），output_mask 为可选属性。

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [pd_x, pd_gamma, pd_beta]（numpy），mask 为 False 的位置为 None（比对按 SUPPRESSED 跳过）。
    """
    # ---- 1. 异常校验 ----
    mask = _check_mask(output_mask)

    # ---- 2. 前处理：numpy → torch，统一 cast 到计算精度 ----
    dy_raw = _np_to_torch(dy)
    x_raw = _np_to_torch(x)
    rstd_raw = _np_to_torch(rstd)
    mean_raw = _np_to_torch(mean)
    gamma_raw = _np_to_torch(gamma)
    compute = _compute_dtype(dy_raw, x_raw, rstd_raw, mean_raw, gamma_raw)
    dy_t = dy_raw.to(compute)
    x_t = x_raw.to(compute)
    rstd_t = rstd_raw.to(compute)
    mean_t = mean_raw.to(compute)
    gamma_t = gamma_raw.to(compute)
    # normalized_shape 即 gamma 的 shape（gamma 恒为 norm 轴 [R1..Rj]）
    normalized_shape = list(gamma_t.shape)

    # ---- 3. 计算：aten 真值 ----
    # aten 不接受 None bias，传零张量
    bias_t = torch.zeros_like(gamma_t)
    pd_x, pd_gamma, pd_beta = torch.ops.aten.native_layer_norm_backward(
        dy_t, x_t, normalized_shape, mean_t, rstd_t, gamma_t, bias_t, mask
    )

    # ---- 4. 后处理：cast 到用例声明的输出 dtype → numpy，mask=False 置 None ----
    out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
    outputs = []
    for i, result in enumerate((pd_x, pd_gamma, pd_beta)):
        if not mask[i]:
            outputs.append(None)
        elif i < len(out_dtypes):
            outputs.append(_torch_to_numpy(result, out_dtypes[i]))
        else:
            outputs.append(_torch_to_numpy(result, "float32"))
    return outputs


# ---------------------------------------------------------------------------
# aclnn golden（torch CPU 张量入/出参）
# ---------------------------------------------------------------------------
def aclnn_layer_norm_backward_golden(
    gradOut,
    input,
    normalizedShape,
    mean,
    rstd,
    weightOptional,
    biasOptional,
    outputMask,
    gradInputOut=None,
    gradWeightOut=None,
    gradBiasOut=None,
    **kwargs,
):
    """
    ACLNN golden for aclnnLayerNormBackward.
    参数名与顺序跟随 aclnn_layer_norm_backward.h 的
    aclnnLayerNormBackwardGetWorkspaceSize（不含 workspaceSize 与 executor）。

    Returns:
        [gradInputOut, gradWeightOut, gradBiasOut]（torch），mask 为 False 的位置为 None。
    """
    # ---- 1. 异常校验 ----
    mask = _check_mask(outputMask)

    # ---- 2. 前处理：cast 到计算精度并加 contiguous 适配非连续输入；weight 为空按算子语义视为全 1 ----
    raw = [gradOut, input, mean, rstd] + (
        [weightOptional] if weightOptional is not None else []
    )
    compute = _compute_dtype(*raw)
    grad_out_t = gradOut.to(compute).contiguous()
    input_t = input.to(compute).contiguous()
    mean_t = mean.to(compute).contiguous()
    rstd_t = rstd.to(compute).contiguous()
    if weightOptional is not None:
        weight_t = weightOptional.to(compute).contiguous()
    else:
        weight_t = torch.ones([int(d) for d in normalizedShape], dtype=compute)

    # ---- 3. 计算：aten 真值 ----
    # aten 不接受 None bias，传零张量
    bias_t = torch.zeros_like(weight_t)
    grad_input, grad_weight, grad_bias = torch.ops.aten.native_layer_norm_backward(
        grad_out_t,
        input_t,
        [int(d) for d in normalizedShape],
        mean_t,
        rstd_t,
        weight_t,
        bias_t,
        mask,
    )

    # ---- 4. 后处理：cast 到用例预分配的输出 dtype（CSV tensor_dtypes 决定）----
    # mask=False 时 aten 返回 None，直接透传（框架比对按 SUPPRESSED 跳过）
    outputs = []
    for result, out in zip(
        (grad_input, grad_weight, grad_bias), (gradInputOut, gradWeightOut, gradBiasOut)
    ):
        if result is None:
            outputs.append(None)
        elif out is not None:
            outputs.append(result.to(out.dtype))
        else:
            outputs.append(result)
    return outputs


# ---------------------------------------------------------------------------
# e2e golden（torch 张量入/出参）
# ---------------------------------------------------------------------------
def aten_native_layer_norm_backward_golden(
    grad_out,
    input,
    normalized_shape,
    mean,
    rstd,
    weight,
    bias,
    output_mask=None,
    **kwargs,
):
    """
    E2E golden for torch.ops.aten.native_layer_norm_backward.
    参数名与顺序跟随 aten schema。输出 dtype 遵循 torch 语义：
    grad_input/grad_bias 同 grad_out，grad_weight 同 weight。

    Returns:
        [grad_input, grad_weight, grad_bias]（torch），mask 为 False 的位置为 None。
    """
    # ---- 1. 异常校验 ----
    mask = _check_mask(output_mask)

    # ---- 2. 前处理：统一 CPU（NPU 路径传入设备张量），cast 到计算精度 ----
    grad_out_c = grad_out.detach().cpu()
    input_c = input.detach().cpu()
    mean_c = mean.detach().cpu()
    rstd_c = rstd.detach().cpu()
    weight_c = weight.detach().cpu() if weight is not None else None
    raw = [grad_out_c, input_c, mean_c, rstd_c] + (
        [weight_c] if weight_c is not None else []
    )
    compute = _compute_dtype(*raw)
    grad_out_t = grad_out_c.to(compute)
    input_t = input_c.to(compute)
    mean_t = mean_c.to(compute)
    rstd_t = rstd_c.to(compute)
    if weight_c is not None:
        weight_t = weight_c.to(compute)
        weight_dtype = weight.dtype
    else:
        weight_t = torch.ones([int(d) for d in normalized_shape], dtype=compute)
        weight_dtype = grad_out.dtype

    # ---- 3. 计算：aten 真值 ----
    # aten 不接受 None bias，传零张量
    bias_t = torch.zeros_like(weight_t)
    grad_input, grad_weight, grad_bias = torch.ops.aten.native_layer_norm_backward(
        grad_out_t,
        input_t,
        [int(d) for d in normalized_shape],
        mean_t,
        rstd_t,
        weight_t,
        bias_t,
        mask,
    )

    # ---- 4. 后处理：cast 回 torch 语义 dtype，mask=False 置 None ----
    target_dtypes = [grad_out.dtype, weight_dtype, grad_out.dtype]
    outputs = []
    for result, dtype in zip((grad_input, grad_weight, grad_bias), target_dtypes):
        outputs.append(result.to(dtype) if result is not None else None)
    return outputs
