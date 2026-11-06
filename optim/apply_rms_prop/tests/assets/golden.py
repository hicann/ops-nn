#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch

# 双 key 注册：CSV 用例的 op_name 为 PascalCase 'ApplyRMSProp'，而安装包接口名与
# GE 图 OpType 会被投影为小写 'apply_rms_prop'（a5-input-prepare 的 api_name 投影、
# GEIR smoke 的 OpType 归一化），两种形式都必须能命中 Golden 查找。
__spec__ = {
    "ApplyRMSProp": "ApplyRMSPropKernelSpec",
    "apply_rms_prop": "ApplyRMSPropKernelSpec",
}
__golden__ = {
    "kernel": {
        "ApplyRMSProp": "apply_rms_prop_golden",
        "apply_rms_prop": "apply_rms_prop_golden",
    }
}


_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _dtype_name(value):
    # P2 fix（检视意见）：golden worker 可能直接喂入 torch tensor（bfloat16
    # tensor 无法经 np.asarray 转换，numpy 不支持 BFloat16），须先按 torch
    # 形态取 dtype，numpy 路径仅作回退。
    if isinstance(value, torch.Tensor):
        return str(value.dtype).replace("torch.", "")
    dtype = np.asarray(value).dtype
    return getattr(dtype, "name", str(dtype))


def _numpy_dtype(dtype_name):
    if dtype_name == "bfloat16":
        import ml_dtypes

        return ml_dtypes.bfloat16
    return np.dtype(dtype_name)


def _to_torch(value):
    # third_party server 端喂入的是 torch tensor（bf16 tensor 无法经 np.asarray
    # 转换：numpy 不支持 BFloat16），直接透传并按需升 fp32。
    if isinstance(value, torch.Tensor):
        return value.float() if value.dtype == torch.bfloat16 else value
    array = np.asarray(value)
    if _dtype_name(array) == "bfloat16":
        return torch.from_numpy(np.asarray(array, dtype=np.float32))
    # GOLDEN.03 fix: np.ascontiguousarray 会把 0 维数组升成 (1,)，导致 rank-0 用例的
    # golden 输出形状 (1,) 与算子声明的 () 不一致；0 维数组本就连续，直接转换。
    if array.ndim == 0:
        return torch.from_numpy(array)
    return torch.from_numpy(np.ascontiguousarray(array))


def _numel(value):
    if isinstance(value, torch.Tensor):
        return value.numel()
    return np.asarray(value).size


def apply_rms_prop_golden(
    var, ms, mom, lr, rho, momentum, epsilon, grad, use_locking=False
):
    del use_locking
    if _numel(var) == 0:
        # 空输入分支同样按 dtype 转换：torch.bfloat16 无法直接 np.asarray
        # （TypeError: Got unsupported ScalarType BFloat16），经 uint16 位视图
        # 无损转 ml_dtypes.bfloat16 空数组，与 _to_torch 的非空路径同口径
        if isinstance(var, torch.Tensor) and var.dtype == torch.bfloat16:
            import ml_dtypes

            return [var.view(torch.uint16).numpy().view(ml_dtypes.bfloat16).copy()]
        return [np.asarray(var).copy()]

    target_dtype_name = _dtype_name(var)
    var_tensor, ms_tensor, mom_tensor, grad_tensor = map(
        _to_torch, (var, ms, mom, grad)
    )
    scalar_tensors = tuple(
        _to_torch(value).reshape(-1)[0] for value in (lr, rho, momentum, epsilon)
    )
    # 算子公开接口仅声明 1 个输出 var（ms/mom 为 Kernel 内原地写回，GE IR 不可观测），
    # 参考返回值与输出声明保持 1:1（GOLDEN.08）；主参考用 float64 计算避免 fp32
    # 中间量在极端梯度下溢出（GOLDEN-02）。
    var_out = _torch_rms_prop(
        var_tensor,
        ms_tensor,
        mom_tensor,
        *scalar_tensors,
        grad_tensor,
        compute_dtype=torch.float64,
    )
    return [
        np.asarray(
            var_out.detach().cpu().numpy(), dtype=_numpy_dtype(target_dtype_name)
        )
    ]


def _torch_rms_prop(
    var, ms, mom, lr, rho, momentum, epsilon, grad, compute_dtype=torch.float32
):
    # GOLDEN-02 fix: compute_dtype=float64 时用于 golden 主参考——fp32 中间量在
    # bf16/fp32 极端梯度（如 grad≈bf16 上限 9.97e37）下自身溢出，float64 无此问题；
    # compute_dtype=float32 保持给 third_party 竞品仿真（与被测实现同等精度域）。
    target_dtype = var.dtype
    v = var.to(compute_dtype)
    m = ms.to(compute_dtype)
    mo = mom.to(compute_dtype)
    gr = grad.to(compute_dtype)

    lr_value = lr.reshape(-1)[0].to(compute_dtype)
    rho_value = rho.reshape(-1)[0].to(compute_dtype)
    momentum_value = momentum.reshape(-1)[0].to(compute_dtype)
    epsilon_value = epsilon.reshape(-1)[0].to(compute_dtype)
    # 与 kernel（FR-03 fix 后）语义对齐：仅对非正值（<=0 / NaN）兜底为 FLT_MIN，
    # 保留 (0, FLT_MIN] 内的合法次正规正值。
    epsilon_floor = torch.tensor(
        torch.finfo(torch.float32).tiny, dtype=compute_dtype, device=var.device
    )
    epsilon_value = torch.where(epsilon_value > 0, epsilon_value, epsilon_floor)

    # OP-01 fix: 按 TF 左结合序 ((1-rho)*grad)*grad 先缩放再平方，避免先平方在
    # 极端梯度下溢出而真实结果可表示。
    # OP-02 fix: 先除后乘 (grad/sqrt)*lr，与 kernel 修复后的求值序一致，
    # 避免 lr*grad 先溢出为 Inf 而真实商仍可表示。
    ms_out = rho_value * m + ((1.0 - rho_value) * gr) * gr
    mo_out = momentum_value * mo + (gr / torch.sqrt(ms_out + epsilon_value)) * lr_value
    var_out = v - mo_out
    return var_out.to(target_dtype)


class _TorchRMSPropCompose:
    """cross_check 的 torch fp32 竞品（与被测实现同等精度域）。

    输出与被测算子输出同 dtype（fp16/bf16 为 RNE 舍入后的竞品结果）；
    计算序与 kernel P1-1 修复后的安全序一致。
    """

    def __init__(self, use_locking=False, **kwargs):
        del use_locking, kwargs

    def __call__(self, var, ms, mom, lr, rho, momentum, epsilon, grad, **kwargs):
        del kwargs
        # 输出 dtype 以原始输入 dtype 为准（third_party server 端喂入 torch tensor，
        # bf16 经 _to_torch 升 fp32 计算后须舍回原始 dtype）。
        target_dtype = (
            var.dtype
            if isinstance(var, torch.Tensor)
            else torch.from_numpy(np.asarray(var)).dtype
        )
        v, m, mo, gr = map(_to_torch, (var, ms, mom, grad))
        scalars = tuple(
            _to_torch(x).reshape(-1)[0] for x in (lr, rho, momentum, epsilon)
        )
        var_out = _torch_rms_prop(v, m, mo, *scalars, gr)
        return var_out.to(target_dtype)


class ApplyRMSPropKernelSpec:
    @staticmethod
    def golden(
        var,
        ms,
        mom,
        lr,
        rho,
        momentum,
        epsilon,
        grad,
        use_locking=False,
        **kwargs,
    ):
        del use_locking, kwargs
        return apply_rms_prop_golden(var, ms, mom, lr, rho, momentum, epsilon, grad)

    third_party = {"torch": _TorchRMSPropCompose}  # noqa: RUF012
    tolerance = _TOLERANCE


# No aclnn or torch e2e TestSpec is registered because this operator only exposes
# the kernel and GE IR paths.
