# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden plugin for FusedCrossEntropyLossWithMaxSum（torch 竞品算子拼接实现）。

    loss[b]      = log(sum_exp_logits[b]) - predicted_logits[b]
    softmax[b,v] = exp(vocab_parallel_logits[b,v] - logits_max[b]) / sum_exp_logits[b]

kernel 内全部按 fp32 计算（fp16/bf16 的 vocab 先升 fp32），softmax 采用乘倒数
（与 kernel 的 1/sum_exp 语义一致）；vocab_parallel_logits 缺省时仅输出 loss。
"""

import numpy as np
import torch

__golden__ = {
    "kernel": {
        "fused_cross_entropy_loss_with_max_sum": "fused_cross_entropy_loss_with_max_sum_golden"
    },
    "aclnn": {
        "aclnnFusedCrossEntropyLossWithMaxSum": "aclnn_fused_cross_entropy_loss_with_max_sum_golden"
    },
}


def _to_torch_f32(tensor):
    """输入归一为 torch float32（接受 numpy / torch tensor，bf16 先升 fp32）。

    ttk 会把 bf16 tensor 转成 ml_dtypes.bfloat16 的 numpy，torch.from_numpy 不支持，
    需先经 numpy astype(np.float32) 升精度（exact 转换）。
    """
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


def fused_cross_entropy_loss_with_max_sum_golden(
    logits_max,
    sum_exp_logits,
    predicted_logits,
    input,
    weight,
    vocab_parallel_logits,
    label_smoothing=0.0,
    **kwargs,
):
    """Golden for fused_cross_entropy_loss_with_max_sum. Parameters follow *_def.cpp (no outputs)."""
    del input, weight, label_smoothing, kwargs
    logits_max_t = _to_torch_f32(logits_max)
    sum_exp_t = _to_torch_f32(sum_exp_logits)
    predicted_t = _to_torch_f32(predicted_logits)

    loss = torch.log(sum_exp_t) - predicted_t
    if vocab_parallel_logits is None:
        # 省显存路径：softmax_logits输出缺省，kernel不写出，返回占位与初始化的ones一致
        return [loss.numpy().astype(np.float32), np.ones(1, dtype=np.float32)]

    vocab_t = _to_torch_f32(vocab_parallel_logits)
    inv_sum = (1.0 / sum_exp_t).reshape(-1, 1)
    softmax = torch.exp(vocab_t - logits_max_t.reshape(-1, 1)) * inv_sum
    return [loss.numpy().astype(np.float32), softmax.numpy().astype(np.float32)]


def aclnn_fused_cross_entropy_loss_with_max_sum_golden(
    logits_max,
    sum_exp_logits,
    predicted_logits,
    label_smoothing,
    input,
    weight,
    vocab_parallel_logits,
    loss_out,
    softmax_out,
    **kwargs,
):
    """Golden for aclnnFusedCrossEntropyLossWithMaxSum. Parameters follow aclnn signature (attrs + outputs included)."""
    del input, weight, label_smoothing, loss_out, softmax_out, kwargs
    logits_max_t = _to_torch_f32(logits_max)
    sum_exp_t = _to_torch_f32(sum_exp_logits)
    predicted_t = _to_torch_f32(predicted_logits)

    loss = (torch.log(sum_exp_t) - predicted_t).numpy().astype(np.float32)
    if vocab_parallel_logits is None:
        # 省显存路径：softmax_logits输出缺省，仅返回loss
        return [loss]

    vocab_t = _to_torch_f32(vocab_parallel_logits)
    inv_sum = (1.0 / sum_exp_t).reshape(-1, 1)
    softmax = (
        (torch.exp(vocab_t - logits_max_t.reshape(-1, 1)) * inv_sum)
        .numpy()
        .astype(np.float32)
    )
    return [loss, softmax]
