#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK input generator for CrossEntropySumExpAndIndexLogit.

TTK's default random generator creates the three inputs independently.  That is
not valid: ``global_logits_max`` must be >= max_j(logits[i,j]) per row to
prevent exp overflow in the golden and kernel.  This generator corrects
``global_logits_max`` per row.
"""

import numpy as np

__input__ = {
    "kernel": {
        "cross_entropy_sum_exp_and_index_logit": "cross_entropy_sum_exp_and_index_logit_inputs",
    },
    "aclnn": {
        "aclnnCrossEntropySumExpAndIndexLogit": "aclnn_cross_entropy_sum_exp_and_index_logit_inputs",
    },
    "e2e": {
        "torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit": "e2e_cross_entropy_sum_exp_and_index_logit_inputs",
    },
}


def _fix_global_logits_max_numpy(logits, gmax):
    """numpy in-place: gmax[i] = max(gmax[i], max_j(logits[i,j]))"""
    per_row_max = logits.max(axis=-1, keepdims=False)
    np.maximum(gmax, per_row_max, out=gmax)
    return gmax


def _fix_global_logits_max_torch(logits, gmax):
    """torch in-place: gmax[i] = max(gmax[i], max_j(logits[i,j]))"""
    import torch

    local_max = torch.max(logits, dim=-1, keepdim=False).values
    torch.max(gmax, local_max, out=gmax)
    return gmax


def cross_entropy_sum_exp_and_index_logit_inputs(
    vocab_parallel_logits,
    target,
    global_logits_max,
    *,
    vocab_start_index=0,
    vocab_end_index=0,
    **kwargs,
):
    """Kernel 模式 input 插件：确保 global_logits_max >= max_j(logits[i,j]) 每行。

    参数名与顺序对齐 cross_entropy_sum_exp_and_index_logit_def.cpp 输入（不含输出）。
    """
    _fix_global_logits_max_numpy(vocab_parallel_logits, global_logits_max)
    return (vocab_parallel_logits, target, global_logits_max)


def aclnn_cross_entropy_sum_exp_and_index_logit_inputs(
    vocabParallelLogits,
    target,
    globalLogitsMax,
    vocabStartIndex,
    vocabEndIndex,
    predictedLogitsOut,
    sumExpLogitsOut,
    expLogitsOut,
    targetOffsetOut,
    targetMaskOut,
    **kwargs,
):
    """ACLNN 模式 input 插件。

    参数顺序与 aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize 一致
    （不含 workspaceSize/executor）。
    注意：aclnn 模式的 _call_custom_input 不消费返回值，必须 in-place 修改传入的
    torch.Tensor 才能生效。
    """
    _fix_global_logits_max_torch(vocabParallelLogits, globalLogitsMax)
    return (
        vocabParallelLogits,
        target,
        globalLogitsMax,
        vocabStartIndex,
        vocabEndIndex,
        predictedLogitsOut,
        sumExpLogitsOut,
        expLogitsOut,
        targetOffsetOut,
        targetMaskOut,
    )


def e2e_cross_entropy_sum_exp_and_index_logit_inputs(
    vocab_parallel_logits,
    target,
    global_logits_max,
    vocab_start_index,
    vocab_end_index,
    **kwargs,
):
    """E2E 模式 input 插件。

    参数顺序与 torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit 一致。
    注意：e2e 模式的 input 插件不消费返回值，必须 in-place 修改传入的
    torch.Tensor 才能生效。
    """
    _fix_global_logits_max_torch(vocab_parallel_logits, global_logits_max)
    return (
        vocab_parallel_logits,
        target,
        global_logits_max,
        vocab_start_index,
        vocab_end_index,
    )
