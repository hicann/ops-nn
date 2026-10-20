# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden plugin for FusedCrossEntropyLossWithMaxSum（torch 小算子拼接，L0）。

loss[b]      = log(sum_exp_logits[b]) - predicted_logits[b]
softmax[b,v] = exp(vocab_parallel_logits[b,v] - logits_max[b]) / sum_exp_logits[b]
"""

import functools

import numpy as np

__spec__ = {
    "fused_cross_entropy_loss_with_max_sum": "FusedCrossEntropyLossWithMaxSumKernelSpec",
    "FusedCrossEntropyLossWithMaxSum": "FusedCrossEntropyLossWithMaxSumKernelSpec",
    "aclnnFusedCrossEntropyLossWithMaxSum": "FusedCrossEntropyLossWithMaxSumAclnnSpec",
    "torch_npu.npu_fused_cross_entropy_loss_with_max_sum": "FusedCrossEntropyLossWithMaxSumE2eSpec",
}

__golden__ = {
    "kernel": {
        "fused_cross_entropy_loss_with_max_sum": "fused_cross_entropy_loss_with_max_sum_golden"
    },
    "aclnn": {
        "aclnnFusedCrossEntropyLossWithMaxSum": "aclnn_fused_cross_entropy_loss_with_max_sum_golden"
    },
}

_L0 = {
    "float32": {"standard": "cross_check", "level": "L0"},
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
}
_KERNEL_TOLERANCE = dict(_L0)
_ACLNN_TOLERANCE = dict(_L0)


def _output_dtype(kwargs, index, default="float32"):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return str(dtype)


def _target_dtype(kwargs, *tensors):
    """Promote 模式下保持 fp64，否则 fp32（kernel 读 golden_mode；aclnn 按输入 dtype 检测）。"""
    import torch

    if str(kwargs.get("golden_mode", "")).lower() == "promote":
        return torch.float64
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _to_torch(tensor, dtype=None):
    """numpy / torch tensor 归一为 torch 指定精度（bf16 numpy 先 astype fp32）。"""
    import torch

    if dtype is None:
        dtype = torch.float32
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().to(dtype)
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
    return torch.from_numpy(arr).to(dtype)


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
    """Golden for fused_cross_entropy_loss_with_max_sum. Parameters follow *_def.cpp."""
    import torch

    dtype = _target_dtype(kwargs)
    logits_max_t = _to_torch(logits_max, dtype)
    sum_exp_t = _to_torch(sum_exp_logits, dtype)
    predicted_t = _to_torch(predicted_logits, dtype)

    loss = torch.log(sum_exp_t) - predicted_t
    if vocab_parallel_logits is None:
        return [loss.to(dtype).numpy(), np.ones(1, dtype=np.float32)]

    vocab_t = _to_torch(vocab_parallel_logits, dtype)
    inv_sum = (1.0 / sum_exp_t).reshape(-1, 1)
    softmax = torch.exp(vocab_t - logits_max_t.reshape(-1, 1)) * inv_sum
    return [loss.to(dtype).numpy(), softmax.to(dtype).numpy()]


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
    """Golden for aclnnFusedCrossEntropyLossWithMaxSum. Parameters follow aclnn signature."""
    import torch

    dtype = _target_dtype(kwargs)
    logits_max_t = _to_torch(logits_max, dtype)
    sum_exp_t = _to_torch(sum_exp_logits, dtype)
    predicted_t = _to_torch(predicted_logits, dtype)

    loss = (torch.log(sum_exp_t) - predicted_t).to(dtype).numpy()
    if vocab_parallel_logits is None:
        return [loss]

    vocab_t = _to_torch(vocab_parallel_logits, dtype)
    inv_sum = (1.0 / sum_exp_t).reshape(-1, 1)
    softmax = (
        (torch.exp(vocab_t - logits_max_t.reshape(-1, 1)) * inv_sum).to(dtype).numpy()
    )
    return [loss, softmax]


def _compute(
    logits_max,
    sum_exp_logits,
    predicted_logits,
    vocab_parallel_logits=None,
    dtype=None,
):
    import torch

    if dtype is None:
        dtype = torch.float32
    logits_max = logits_max.to(dtype)
    sum_exp_logits = sum_exp_logits.to(dtype)
    predicted_logits = predicted_logits.to(dtype)
    loss = torch.log(sum_exp_logits) - predicted_logits
    if vocab_parallel_logits is None:
        return [loss]

    vocab_parallel_logits = vocab_parallel_logits.to(dtype)
    inv_sum = (1.0 / sum_exp_logits).reshape(-1, 1)
    softmax = torch.exp(vocab_parallel_logits - logits_max.reshape(-1, 1)) * inv_sum
    return [loss, softmax]


def _make_torch_bridge():
    """third_party 桥接：__call__ 内零精度升降（vocab fp16/bf16→fp32 经闭包缓存）。"""
    bridge_cache = {}

    def _bridge(t):
        import torch

        if t is None or t.dtype == torch.float32:
            return t
        key = (t.data_ptr(), str(t.dtype), tuple(t.shape))
        hit = bridge_cache.get(key)
        if hit is None or hit[0] is not t:
            if len(bridge_cache) >= 8:
                bridge_cache.clear()
            hit = (t, t.to(torch.float32))
            bridge_cache[key] = hit
        return hit[1]

    def _fce_call(
        logits_max,
        sum_exp_logits,
        predicted_logits,
        vocab_parallel_logits=None,
        **kwargs,
    ):
        import torch

        outs = _compute(
            logits_max, sum_exp_logits, predicted_logits, _bridge(vocab_parallel_logits)
        )
        if vocab_parallel_logits is None and len(outs) == 1:
            outs = outs + [torch.ones(1, dtype=outs[0].dtype)]
        return outs

    return functools.partial(_fce_call)


def _make_torch_bridge_aclnn():
    """third_party 桥接（aclnn 参数名）。"""
    bridge_cache = {}

    def _bridge(t):
        import torch

        if t is None or t.dtype == torch.float32:
            return t
        key = (t.data_ptr(), str(t.dtype), tuple(t.shape))
        hit = bridge_cache.get(key)
        if hit is None or hit[0] is not t:
            if len(bridge_cache) >= 8:
                bridge_cache.clear()
            hit = (t, t.to(torch.float32))
            bridge_cache[key] = hit
        return hit[1]

    def _fce_aclnn_call(
        logitsMax,
        sumExpLogits,
        predictedLogits,
        vocabParallelLogitsOptional=None,
        **kwargs,
    ):
        return _compute(
            logitsMax,
            sumExpLogits,
            predictedLogits,
            _bridge(vocabParallelLogitsOptional),
        )

    return functools.partial(_fce_aclnn_call)


class FusedCrossEntropyLossWithMaxSumKernelSpec:
    """kernel spec. The golden entry receives numpy arrays."""

    def golden(
        logits_max,
        sum_exp_logits,
        predicted_logits,
        input=None,
        weight=None,
        vocab_parallel_logits=None,
        label_smoothing=0.0,
        **kwargs,
    ):
        outs = fused_cross_entropy_loss_with_max_sum_golden(
            logits_max,
            sum_exp_logits,
            predicted_logits,
            input,
            weight,
            vocab_parallel_logits,
            label_smoothing=label_smoothing,
        )
        return [
            outs[i].astype(_output_dtype(kwargs, i), copy=False)
            for i in range(len(outs))
        ]

    third_party = {"torch": _make_torch_bridge()}
    tolerance = _KERNEL_TOLERANCE


class FusedCrossEntropyLossWithMaxSumAclnnSpec:
    """aclnn spec. The golden entry receives torch tensors."""

    def golden(*inputs, **kwargs):
        logits_max = kwargs.get("logitsMax", inputs[0] if len(inputs) > 0 else None)
        sum_exp_logits = kwargs.get(
            "sumExpLogits", inputs[1] if len(inputs) > 1 else None
        )
        predicted_logits = kwargs.get(
            "predictedLogits", inputs[2] if len(inputs) > 2 else None
        )
        vocab_parallel_logits = kwargs.get("vocabParallelLogitsOptional")
        if vocab_parallel_logits is None:
            if len(inputs) >= 9:
                vocab_parallel_logits = inputs[6]
            elif len(inputs) >= 6:
                vocab_parallel_logits = inputs[5]
        return _compute(
            logits_max,
            sum_exp_logits,
            predicted_logits,
            vocab_parallel_logits,
            dtype=_target_dtype(kwargs, logits_max),
        )

    third_party = {"torch": _make_torch_bridge_aclnn()}
    tolerance = _ACLNN_TOLERANCE


class FusedCrossEntropyLossWithMaxSumE2eSpec:
    """e2e spec（torch_npu.npu_fused_cross_entropy_loss_with_max_sum）."""

    def golden(
        logits_max,
        sum_exp_logits,
        predicted_logits,
        label_smoothing=0.0,
        input=None,
        weight=None,
        vocab_parallel_logits=None,
        **kwargs,
    ):
        return _compute(
            logits_max,
            sum_exp_logits,
            predicted_logits,
            vocab_parallel_logits,
            dtype=_target_dtype(kwargs, logits_max),
        )

    third_party = {"torch": _make_torch_bridge()}
    tolerance = _ACLNN_TOLERANCE
