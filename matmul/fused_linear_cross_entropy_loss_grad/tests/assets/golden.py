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

__golden__ = {
    "kernel": {
        "fused_linear_cross_entropy_loss_grad": "fused_linear_cross_entropy_loss_grad_golden"
    },
    "aclnn": {
        "aclnnFusedLinearCrossEntropyLossGrad": "aclnn_fused_linear_cross_entropy_loss_grad_golden"
    },
    "e2e": {
        "flceg_torch.fused_linear_cross_entropy_loss_grad": "e2e_fused_linear_cross_entropy_loss_grad_golden"
    },
}


def _unpack_target_mask_vec(target_mask, BT):
    tm_t = torch.from_numpy(np.ascontiguousarray(target_mask))
    return _unpack_target_mask_vec_torch(tm_t, BT)


def fused_linear_cross_entropy_loss_grad_golden(
    grad,
    input,
    weight,
    target_mask,
    masked_target,
    logits_max=None,
    sum_exp_logits=None,
    softmax=None,
    **kwargs,
):
    """Kernel golden (torch BLAS). Params follow def.cpp inputs (without outputs)."""
    BT = input.shape[0]
    V = weight.shape[0]

    input_t = torch.from_numpy(input.astype(np.float32))
    weight_t = torch.from_numpy(weight.astype(np.float32))
    grad_t = (
        torch.from_numpy(grad.astype(np.float32))
        if grad.ndim >= 1
        else torch.ones(BT, dtype=torch.float32)
    )

    if softmax is not None and softmax.ndim >= 2:
        grad_logits = torch.from_numpy(softmax.astype(np.float32)).clone()
    else:
        logits = torch.matmul(input_t, weight_t.t())
        if logits_max is not None and logits_max.ndim >= 1:
            lm = torch.from_numpy(logits_max.astype(np.float32)).unsqueeze(1)
        else:
            lm = logits.max(dim=1, keepdim=True)[0]
        if sum_exp_logits is not None and sum_exp_logits.ndim >= 1:
            se = torch.from_numpy(sum_exp_logits.astype(np.float32)).unsqueeze(1)
        else:
            se = torch.exp(logits - lm).sum(dim=1, keepdim=True)
        grad_logits = torch.exp(logits - lm) / se

    tm = _unpack_target_mask_vec(target_mask, BT)
    mt = torch.from_numpy(masked_target.astype(np.int64)).clamp(0, V - 1)
    update = 1.0 - tm
    grad_logits[torch.arange(BT), mt] -= update
    grad_logits = grad_logits * grad_t.unsqueeze(1)

    # 对齐 kernel CastVecFull 语义: gl 先量化到输入 dtype 再参与 matmul。
    # 否则全 f32 golden 与 NPU(量化后半精度 matmul)的固有量化噪声
    # (~|值|*2^-11) 在 fp16 紧容差下擦线误报(与 third_party 实现一致)。
    grad_logits = grad_logits.to(input_t.dtype).float()

    input_grad = torch.matmul(grad_logits, weight_t)
    weight_grad = torch.matmul(grad_logits.t(), input_t)

    out_dtype = input.dtype
    return [input_grad.numpy().astype(out_dtype), weight_grad.numpy().astype(out_dtype)]


def aclnn_fused_linear_cross_entropy_loss_grad_golden(
    grad,
    input,
    weight,
    targetMask,
    maskedTarget,
    labelSmoothing=0.0,
    logitsMaxOptional=None,
    sumExpLogitsOptional=None,
    softmaxOptional=None,
    inputGradOut=None,
    weightGradOut=None,
    **kwargs,
):
    """ACLNN golden (torch). Params follow aclnnGetWorkspaceSize signature."""
    import torch

    BT = input.shape[0]
    V = weight.shape[0]

    input_f32 = input.float()
    weight_f32 = weight.float()
    grad_f32 = (
        grad.float()
        if hasattr(grad, "dim") and grad.dim() >= 1
        else torch.ones(BT, dtype=torch.float32)
    )

    if (
        softmaxOptional is not None
        and hasattr(softmaxOptional, "dim")
        and softmaxOptional.dim() >= 2
    ):
        grad_logits = softmaxOptional.float().clone()
    else:
        logits = torch.matmul(input_f32, weight_f32.t())
        lm = (
            logitsMaxOptional.float().unsqueeze(1)
            if logitsMaxOptional is not None
            and hasattr(logitsMaxOptional, "dim")
            and logitsMaxOptional.dim() >= 1
            else logits.max(dim=1, keepdim=True)[0]
        )
        se = (
            sumExpLogitsOptional.float().unsqueeze(1)
            if sumExpLogitsOptional is not None
            and hasattr(sumExpLogitsOptional, "dim")
            and sumExpLogitsOptional.dim() >= 1
            else torch.exp(logits - lm).sum(dim=1, keepdim=True)
        )
        grad_logits = torch.exp(logits - lm) / se

    tm = _unpack_target_mask_vec_torch(targetMask, BT)
    mt = maskedTarget.long().clamp(0, V - 1)
    update = 1.0 - tm
    grad_logits[torch.arange(BT), mt] -= update
    grad_logits = grad_logits * grad_f32.unsqueeze(1)

    # 对齐 kernel CastVecFull 语义: gl 先量化到输入 dtype 再参与 matmul
    # (与 third_party/_fused_grad_torch 的 `.to(input.dtype)` 一致)
    grad_logits = grad_logits.to(input.dtype).float()

    input_grad = torch.matmul(grad_logits, weight_f32)
    weight_grad = torch.matmul(grad_logits.t(), input_f32)

    out_dtype = input.dtype
    return [input_grad.to(out_dtype), weight_grad.to(out_dtype)]


def e2e_fused_linear_cross_entropy_loss_grad_golden(
    grad,
    input,
    weight,
    target_mask,
    masked_target,
    label_smoothing=0.0,
    logits_max=None,
    sum_exp_logits=None,
    softmax=None,
    **kwargs,
):
    """e2e golden (torch CPU tensors, 与 flceg_torch API 同签名)。
    语义: softmax(可选) -> update -> mul(grad) -> cast(输入dtype) -> matmul。
    """
    BT = input.shape[0]
    V = weight.shape[0]

    if softmax is not None and getattr(softmax, "dim", lambda: 0)() >= 2:
        grad_logits = softmax.float().clone()
    else:
        logits = torch.matmul(input.float(), weight.float().t())
        lm = (
            logits_max.float().unsqueeze(1)
            if logits_max is not None
            else logits.max(dim=1, keepdim=True)[0]
        )
        se = (
            sum_exp_logits.float().unsqueeze(1)
            if sum_exp_logits is not None
            else torch.exp(logits - lm).sum(dim=1, keepdim=True)
        )
        grad_logits = torch.exp(logits - lm) / se

    tm = _unpack_target_mask_vec_torch(target_mask, BT)
    mt = masked_target.long().clamp(0, V - 1)
    update = 1.0 - tm
    grad_logits[torch.arange(BT), mt] -= update
    grad_v = grad.float() if grad.dim() >= 1 else torch.ones(BT)
    grad_logits = grad_logits * grad_v.unsqueeze(1)

    # 对齐 kernel CastVecFull: gl 先量化到输入 dtype 再参与 matmul
    grad_logits = grad_logits.to(input.dtype).float()

    input_grad = torch.matmul(grad_logits, weight.float())
    weight_grad = torch.matmul(grad_logits.t(), input.float())
    return [input_grad.to(input.dtype), weight_grad.to(input.dtype)]


def _unpack_target_mask_vec_torch(targetMask, BT):
    import torch

    if targetMask.dtype == torch.bool:
        return targetMask[:BT].float()
    elif targetMask.dtype == torch.uint8:
        # GPU 安全位解包: 不能用 .item() 逐字节循环(远端 GPU 三方执行时
        # .item() 隐式拉回 CPU, 与 GPU 张量混用报跨设备错误)
        dev = targetMask.device
        bt = int(targetMask.shape[0])
        if bt > 0:
            bytesT = targetMask.flatten().to(torch.int64)
            bits = (
                torch.arange(8, device=dev, dtype=torch.int64).view(1, 8).expand(bt, 8)
            )
            unpacked = (
                ((bytesT.view(bt, 1) >> bits) & 1).to(torch.float32).flatten()[:BT]
            )
            return unpacked.contiguous()
        return torch.zeros(BT, dtype=torch.float32, device=dev)
    return targetMask[:BT].float()


class FusedLinearCrossEntropyLossGradTestSpec:
    """TestSpec for cross_check + xpu-perf"""

    def __call__(self, *args, **kwargs):
        # aclnn 模式 golden 生成路由到本类时，代理到 aclnn golden
        return aclnn_fused_linear_cross_entropy_loss_grad_golden(*args, **kwargs)

    def golden(*args, **kwargs):
        # kernel模式传5-8个位置参数(snake_case约定)，aclnn模式传11个(camelCase约定含输出)
        if len(args) >= 9 or "labelSmoothing" in kwargs or "targetMask" in kwargs:
            return aclnn_fused_linear_cross_entropy_loss_grad_golden(*args, **kwargs)
        return fused_linear_cross_entropy_loss_grad_golden(*args, **kwargs)

    def _fused_grad_torch(
        grad,
        input,
        weight,
        target_mask,
        masked_target,
        logits_max=None,
        sum_expLogits=None,
        softmax=None,
        label_smoothing=0.0,
        **kwargs,
    ):
        import torch

        BT = input.shape[0]
        V = weight.shape[0]

        if softmax is not None and softmax.dim() >= 2:
            grad_logits = softmax.clone()
        else:
            logits = torch.matmul(input.float(), weight.float().t())
            lm = (
                logits_max.float().unsqueeze(1)
                if logits_max is not None and logits_max.dim() >= 1
                else logits.max(dim=1, keepdim=True)[0]
            )
            se = (
                sum_expLogits.float().unsqueeze(1)
                if sum_expLogits is not None and sum_expLogits.dim() >= 1
                else torch.exp(logits - lm).sum(dim=1, keepdim=True)
            )
            grad_logits = torch.exp(logits - lm) / se

        tm = _unpack_target_mask_vec_torch(target_mask, BT)
        mt = masked_target.long().clamp(0, V - 1)
        update = 1.0 - tm
        grad_logits[torch.arange(BT, device=input.device), mt] -= update
        if grad.dim() >= 1:
            grad_logits = grad_logits * grad.unsqueeze(1)

        # Cast to input dtype to match NPU kernel's CastVecFull (float32 → bf16)
        grad_logits = grad_logits.to(input.dtype)

        input_grad = torch.matmul(grad_logits, weight)
        weight_grad = torch.matmul(grad_logits.t(), input)

        out_dtype = input.dtype
        return [input_grad.to(out_dtype), weight_grad.to(out_dtype)]

    third_party = {"torch": _fused_grad_torch}

    tolerance = {
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


__spec__ = {
    "fused_linear_cross_entropy_loss_grad": "FusedLinearCrossEntropyLossGradTestSpec",
    "aclnnFusedLinearCrossEntropyLossGrad": "FusedLinearCrossEntropyLossGradTestSpec",
}
