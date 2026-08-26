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

import numpy as np

__golden__ = {"aclnn": {"aclnnCrossEntropyLoss": "aclnn_cross_entropy_loss_golden"}}

_REDUCTION_MAP = {"none": 0, "mean": 1, "sum": 2}


def _to_torch_f32(tensor):
    import torch

    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().to(torch.float32)
    arr = np.asarray(tensor)
    if arr.dtype not in (
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.int16,
        np.int8,
        np.uint8,
    ):
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr).to(torch.float32)


def _compute_loss_and_logprob(
    logits_t, targets_t, weight_t, reduction, ignore_index, label_smoothing
):
    import torch
    import torch.nn.functional as F

    N, C = logits_t.shape[0], logits_t.shape[1]

    if N == 0 or C == 0:
        if reduction == "none":
            loss = torch.zeros((N,), dtype=torch.float32)
        elif reduction == "mean":
            loss = torch.full((1,), float("nan"), dtype=torch.float32)
        else:
            loss = torch.zeros((1,), dtype=torch.float32)
        log_prob = F.log_softmax(logits_t, dim=-1)
        return loss, log_prob

    reduction_int = _REDUCTION_MAP.get(reduction, 1)

    loss = torch.ops.aten.cross_entropy_loss(
        logits_t,
        targets_t,
        weight_t,
        reduction=reduction_int,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    log_prob = F.log_softmax(logits_t, dim=-1)

    if reduction != "none" and loss.dim() == 0:
        loss = loss.unsqueeze(0)

    return loss, log_prob


def aclnn_cross_entropy_loss_golden(
    input,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
    lse_square_scale_for_zloss,
    return_zloss,
    loss_out,
    log_prob_out,
    zloss_out,
    lse_for_zloss_out,
    **kwargs,
):
    """
    Aclnn golden for aclnnCrossEntropyLoss.
    All the parameters (name & order) follow \
        function `aclnnCrossEntropyLossGetWorkspaceSize` in @aclnn_cross_entropy_loss.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        (loss, log_prob, zloss, lse_for_zloss) as torch.Tensor or numpy.ndarray.
    """
    import torch

    del loss_out, log_prob_out, zloss_out, lse_for_zloss_out

    if isinstance(reduction, bytes):
        reduction = reduction.decode()

    is_torch_tensor = isinstance(input, torch.Tensor)
    out_dtype = input.dtype

    logits_t = _to_torch_f32(input)
    if weight is not None:
        weight_t = _to_torch_f32(weight)
    else:
        weight_t = None
    if isinstance(target, torch.Tensor):
        target_t = target.to(torch.int64)
    else:
        target_t = torch.from_numpy(np.asarray(target)).to(torch.int64)

    loss, log_prob = _compute_loss_and_logprob(
        logits_t, target_t, weight_t, reduction, ignore_index, label_smoothing
    )

    if is_torch_tensor:
        return (
            loss.to(out_dtype),
            log_prob.to(out_dtype),
            None,
            None,
        )
    return (
        loss.numpy().astype(out_dtype, copy=False),
        log_prob.numpy().astype(out_dtype, copy=False),
        None,
        None,
    )
