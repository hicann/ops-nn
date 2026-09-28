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
"""
TTK kernel golden for CTCLossV2Grad (arch35 / Ascend950).

真值来源：竞品接口 ``torch.ops.aten._ctc_loss_backward``（与前向 golden
``loss/ctc_loss_v2`` 使用 ``torch.ops.aten._ctc_loss`` 一致，非 numpy 纯公式）。

参数名/顺序 = ctc_loss_v2_grad_def.cpp 的输入（不含输出）：
    inputs : grad_out, log_probs, targets, input_lengths, target_lengths,
             neg_log_likelihood, log_alpha
    output : grad（单输出，shape = log_probs = (T, N, C)）
    attrs  : blank(OPTIONAL int=0), reduction(OPTIONAL str="mean"),
             zero_infinity(OPTIONAL bool=False)

语义核对（对照 op_kernel/arch35/ctc_loss_v2_grad.h）：
  - grad[t,b,c] = (exp(lp) - exp(res + nll - lp)) * grad_out[b]，即 grad_out 为逐 batch
    上游梯度；这正是 aten._ctc_loss_backward 的定义。
  - reduction 属性在 kernel 中【不参与计算】（tiling 仅读取 blank/zero_infinity），
    grad_out 已在框架侧按 reduction 完成缩放；故本 golden 接收 reduction 但不使用它。
  - zero_infinity=True 时，对 neg_log_likelihood 为 ±inf 的 batch 输出 0 梯度
    （由 aten 内核保证，与 NPU kernel 一致）。

dtype：fp16/bf16 在 CPU aten 上不支持，统一抬 fp32 计算后舍回原 dtype
（与 NPU 内核 fp32 累加一致，不抬 fp64）。整型 targets 统一转 int64 供 aten 使用。
"""

import numpy as np

__golden__ = {"kernel": {"ctc_loss_v2_grad": "ctc_loss_v2_grad_golden"}}


def _to_float32(arr):
    """低精度浮点（fp16/bf16，含 ml_dtypes.bfloat16）统一抬到 float32。"""
    a = np.asarray(arr)
    if a.dtype == np.float32 or a.dtype == np.float64:
        return a.astype(np.float32, copy=False)
    # float16 / bfloat16(ml_dtypes) 等：astype 到 float32
    return a.astype(np.float32)


def _int_list(arr):
    return [int(x) for x in np.asarray(arr).reshape(-1).tolist()]


def ctc_loss_v2_grad_golden(
    grad_out,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    *,
    blank=0,
    reduction="mean",
    zero_infinity=False,
    **kwargs,
):
    """CTCLossV2Grad 反向真值。返回 [grad]（单输出，dtype 与 log_probs 一致）。

    Args:
        grad_out: (N,) 逐 batch 上游梯度。
        log_probs: (T, N, C) 对数概率。
        targets: (N, S) 或 (sum(target_lengths),) 目标序列。
        input_lengths: (N,) 每个 batch 的有效输入长度。
        target_lengths: (N,) 每个 batch 的有效目标长度。
        neg_log_likelihood: (N,) 前向损失。
        log_alpha: (N, T, 2*max(target_lengths)+1) 前向 alpha。
        blank/reduction/zero_infinity: 算子属性（reduction 不参与计算）。
        **kwargs: TTK 注入的元信息（{input,output}_{dtypes,ori_shapes,...}, testcase_name 等）。

    Returns:
        list[numpy.ndarray]: [grad]，shape=(T, N, C)，dtype 同 log_probs。
    """
    import torch

    out_dtype = np.asarray(log_probs).dtype

    grad_out_f = torch.from_numpy(_to_float32(grad_out))
    log_probs_f = torch.from_numpy(_to_float32(log_probs))
    neg_log_likelihood_f = torch.from_numpy(_to_float32(neg_log_likelihood))
    log_alpha_f = torch.from_numpy(_to_float32(log_alpha))
    targets_i = torch.from_numpy(np.asarray(targets).astype(np.int64, copy=False))

    input_lengths_l = _int_list(input_lengths)
    target_lengths_l = _int_list(target_lengths)

    grad = torch.ops.aten._ctc_loss_backward(
        grad_out_f,
        log_probs_f,
        targets_i,
        input_lengths_l,
        target_lengths_l,
        neg_log_likelihood_f,
        log_alpha_f,
        int(blank),
        bool(zero_infinity),
    )

    return [grad.numpy().astype(out_dtype, copy=False)]
