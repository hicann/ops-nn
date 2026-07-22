# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden plugin for FusedCrossEntropyLossWithMaxSum.

    loss[b]      = log(sum_exp_logits[b]) - predicted_logits[b]
    softmax[b,v] = exp(vocab_parallel_logits[b,v] - logits_max[b]) / sum_exp_logits[b]

kernel 内全部按 fp32 计算（fp16/bf16 的 vocab 先升 fp32），softmax 采用乘倒数
（与 kernel 的 1/sum_exp 语义一致）；vocab_parallel_logits 缺省时仅输出 loss。
"""

import numpy as np

from ttk.utilities.dtypes import (
    torch_to_numpy_tensor,
)

__golden__ = {
    "kernel": {
        "fused_cross_entropy_loss_with_max_sum": "fused_cross_entropy_loss_with_max_sum_golden"
    },
    "aclnn": {
        "aclnnFusedCrossEntropyLossWithMaxSum": "aclnn_fused_cross_entropy_loss_with_max_sum_golden"
    },
}


def _to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        return torch_to_numpy_tensor(tensor.detach().cpu())
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


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
    logits_max_np = _to_numpy(logits_max).astype(np.float32)
    sum_exp_np = _to_numpy(sum_exp_logits).astype(np.float32)
    predicted_np = _to_numpy(predicted_logits).astype(np.float32)

    loss = np.log(sum_exp_np) - predicted_np
    if vocab_parallel_logits is None:
        # 省显存路径：softmax_logits输出缺省，kernel不写出，返回占位与初始化的ones一致
        return [loss.astype(np.float32), np.ones(1, dtype=np.float32)]

    vocab_np = _to_numpy(vocab_parallel_logits).astype(np.float32)
    inv_sum = (1.0 / sum_exp_np).reshape(-1, 1)
    softmax = np.exp(vocab_np - logits_max_np.reshape(-1, 1)) * inv_sum
    return [loss.astype(np.float32), softmax.astype(np.float32)]


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
    logits_max_np = _to_numpy(logits_max).astype(np.float32)
    sum_exp_np = _to_numpy(sum_exp_logits).astype(np.float32)
    predicted_np = _to_numpy(predicted_logits).astype(np.float32)

    loss = (np.log(sum_exp_np) - predicted_np).astype(np.float32)
    if vocab_parallel_logits is None:
        # 省显存路径：softmax_logits输出缺省，仅返回loss
        return [loss]

    vocab_np = _to_numpy(vocab_parallel_logits).astype(np.float32)
    inv_sum = (1.0 / sum_exp_np).reshape(-1, 1)
    softmax = (np.exp(vocab_np - logits_max_np.reshape(-1, 1)) * inv_sum).astype(
        np.float32
    )
    return [loss, softmax]
