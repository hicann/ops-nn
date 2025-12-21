#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import torch
import os

def get_cpu_data(N, C, grad_loss, log_prob, target, weight, reduction, flag, ignore_index=-100, label_smoothing=0.0):
    ori_type = log_prob.dtype
    grad_loss = torch.from_numpy(grad_loss).to(torch.float32)
    log_prob = torch.from_numpy(log_prob).to(torch.float32)
    targets = torch.from_numpy(target)
    weight_input = None
    if flag == 1:
        weight_input = torch.from_numpy(weight)
    else:
        weight_input = torch.ones(C, dtype=torch.float32)
    weight_yn = torch.gather(weight_input, 0, targets)
    if ignore_index >= 0:
        ignore_mask = (targets - ignore_index).bool().float()
    else:
        ignore_mask = torch.ones((N,), dtype=torch.float32)
    if reduction == "mean":
        mean_out_grad = grad_loss * (1 - label_smoothing)
        weight_after_mask = weight_yn * ignore_mask
        weight_after_mask_sum = torch.sum(weight_after_mask, -1, keepdim=False)
        smooth_loss_grad = grad_loss / weight_after_mask_sum * (label_smoothing / C)
        loss_out_grad = mean_out_grad.unsqueeze(-1) / weight_after_mask_sum
    elif reduction == "sum":
        sum_out_grad = grad_loss * (1-label_smoothing)
        smooth_loss_grad = grad_loss.unsqueeze(-1) * (label_smoothing / C)
        loss_out_grad = sum_out_grad.unsqueeze(-1)
    else:
        none_out_grad = grad_loss * (1-label_smoothing)
        smooth_loss_grad = grad_loss* (label_smoothing / C)
        loss_out_grad = none_out_grad
    loss_out_grad = loss_out_grad * ignore_mask
    nll_loss_grad = loss_out_grad * weight_yn
    log_softmax_probs_grad_loss_out_sub_part = torch.exp(log_prob) * nll_loss_grad.unsqueeze(-1)
    predictions_grad_loss_out = torch.zeros((N, C)).float()
    for i in range(N):
        predictions_grad_loss_out[i][targets[i]] = nll_loss_grad[i]
    predictions_grad_loss_out = log_softmax_probs_grad_loss_out_sub_part - predictions_grad_loss_out
    if label_smoothing == 0:
        out = predictions_grad_loss_out
    else:
        smooth_loss_grad = torch.mul(smooth_loss_grad, ignore_mask)
        log_softmax_probs_grad_smooth_loss = torch.mul(smooth_loss_grad.unsqueeze(-1), weight_input.unsqueeze(0))
        predictions_grad_smooth_loss = torch.exp(log_prob) * torch.sum(log_softmax_probs_grad_smooth_loss, -1, keepdim=True) - log_softmax_probs_grad_smooth_loss
        predictions_grad = predictions_grad_loss_out + predictions_grad_smooth_loss
        out = predictions_grad
    return out.detach().numpy().astype(ori_type)

def gen_data_and_golden(batch_size, num_classes, dtype, reduction, flag, ignore, smooth): # flag=True传weight, False不传weight
    grad_loss = np.array(np.random.randn()).astype(dtype)
    log_prob = np.random.uniform(-10, 0, [batch_size, num_classes]).astype(dtype)
    target = np.random.uniform(0, num_classes, [batch_size,]).astype("int64")
    weight = np.random.uniform(0.1, 1, [int(num_classes),]).astype("float32")
    x_grad = get_cpu_data(batch_size, num_classes, grad_loss, log_prob, target, weight, reduction, int(flag), int(ignore), float(smooth))

    grad_loss.tofile("./grad_loss.bin")
    log_prob.tofile("./log_prob.bin")
    target.tofile("./target.bin")
    if flag:
        weight.tofile("./weight.bin")

    x_grad.tofile("./output_x_grad.bin")


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    # gen_data_and_golden(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    gen_data_and_golden(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
