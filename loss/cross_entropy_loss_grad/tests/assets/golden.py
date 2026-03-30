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

__golden__ = {"kernel": {"cross_entropy_loss_grad": "cross_entropy_loss_grad_golden"}}


def cross_entropy_loss_grad_golden(grad_loss, log_prob, target, weight=None, grad_zloss=None, lse_for_zloss=None, *,
                                    reduction="mean", ignore_index=-100, label_smoothing=0.0, **kwargs):
    '''
    Golden function for cross_entropy_loss_grad.
    All the parameters (names and order) follow @cross_entropy_loss_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch

    input_dtype = grad_loss.dtype
    if input_dtype.name == "bfloat16" or input_dtype.name == "float16":
        grad_loss = grad_loss.astype("float32")
        log_prob = log_prob.astype("float32")
        if weight is not None:
            weight = weight.astype("float32")
    target = target.astype("int64")
    shape = log_prob.shape
    N = shape[0]
    C = shape[1]

    grad_loss = torch.from_numpy(grad_loss)
    log_prob = torch.from_numpy(log_prob)
    targets = torch.from_numpy(target)
    weight_input = None
    if weight is not None:
        weight_input = torch.from_numpy(weight)
    else:
        weight_input = torch.ones(C, dtype=torch.float32).cpu()
    weight_yn = torch.gather(weight_input, 0, targets).cpu()
    if ignore_index >= 0:
        ignore_mask = (targets - ignore_index).bool().float().cpu()
    else:
        ignore_mask = torch.ones((N,), dtype=torch.float32).cpu()
    if reduction == "mean":
        mean_out_grad = grad_loss * (1 - label_smoothing) # []
        weight_after_mask = weight_yn * ignore_mask # [N]
        weight_after_mask_sum = torch.sum(weight_after_mask, -1, keepdim=False)
        smooth_loss_grad = grad_loss / weight_after_mask_sum * (label_smoothing / C)
        loss_out_grad = mean_out_grad.unsqueeze(-1) / weight_after_mask_sum
    elif reduction == "sum":
        sum_out_grad = grad_loss * (1 - label_smoothing) # []
        smooth_loss_grad = grad_loss.unsqueeze(-1) * (label_smoothing / C)
        loss_out_grad = sum_out_grad.unsqueeze(-1)  # [N]
    else:
        none_out_grad = grad_loss * (1 - label_smoothing)
        smooth_loss_grad = grad_loss * (label_smoothing / C)
        loss_out_grad = none_out_grad
    loss_out_grad = loss_out_grad * ignore_mask
    nll_loss_grad = loss_out_grad * weight_yn
    nll_loss_grad = nll_loss_grad.squeeze(0)
    log_softmax_probs_grad_loss_out_sub_part = torch.exp(log_prob) * nll_loss_grad.unsqueeze(-1)
    predictions_grad_loss_out = torch.zeros((N, C)).float()
    if nll_loss_grad.dim() == 0:
        nll_loss_grad = nll_loss_grad.unsqueeze(0)
    for i in range(N):
        predictions_grad_loss_out[i][targets[i]] = nll_loss_grad[i]
    predictions_grad_loss_out = log_softmax_probs_grad_loss_out_sub_part - predictions_grad_loss_out

    if label_smoothing == 0:
        out = predictions_grad_loss_out.numpy()
    else:
        smooth_loss_grad = torch.mul(smooth_loss_grad, ignore_mask) # [N]
        log_softmax_probs_grad_smooth_loss = torch.mul(smooth_loss_grad.unsqueeze(-1), weight_input.unsqueeze(0))
        predictions_grad_smooth_loss = torch.exp(log_prob) * torch.sum(log_softmax_probs_grad_smooth_loss, -1, keepdim=True) - log_softmax_probs_grad_smooth_loss
        predictions_grad = predictions_grad_loss_out + predictions_grad_smooth_loss
        out = predictions_grad.numpy()
    return out.astype(input_dtype, copy=False)
