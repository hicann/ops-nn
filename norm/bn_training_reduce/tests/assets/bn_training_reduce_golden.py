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
"""BNTrainingReduce 独立 Golden：Kernel、GEIR、ACLNN 与 PyTorch Eager 通路。"""

from typing import Sequence

import numpy as np
import torch

__spec__ = {
    # Kernel and GEIR resolve the exact op_name and share the same Spec.
    # Keep both the snake_case name and the canonical GE OpType as aliases.
    "bn_training_reduce": "BNTrainingReduceKernelSpec",
    "BNTrainingReduce": "BNTrainingReduceKernelSpec",
    "aclnnBatchNormReduce": "BNTrainingReduceAclnnSpec",
    "torch_npu.batch_norm_reduce": "BNTrainingReduceE2ESpec",
    "torch.ops.npu.batch_norm_reduce": "BNTrainingReduceE2ESpec",
    "torch.ops.npu.batch_norm_reduce.default": "BNTrainingReduceE2ESpec",
    "npu.batch_norm_reduce": "BNTrainingReduceE2ESpec",
}

_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}

_SUPPORTED_FORMATS = {"NCHW", "NHWC", "NCDHW"}
_NORMAL_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_PROMOTED_INPUT_DTYPES = {torch.float32, torch.float64}


def _as_torch(x) -> torch.Tensor:
    """把 Kernel/GEIR 的 NumPy 输入桥接为 Torch；ACLNN 输入原样复用。"""
    if isinstance(x, torch.Tensor):
        return x
    array = np.asarray(x)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _compute_torch(x, data_format: str, golden_mode=None) -> list:
    """唯一数学实现：使用 Torch 小算子计算每通道 sum 与 square_sum。"""
    tensor = _as_torch(x)
    fmt = str(data_format).upper()
    if fmt == "NHWC":
        c_axis = max(tensor.ndim - 1, 0)
    elif fmt in ("NCHW", "NCDHW"):
        c_axis = 1 if tensor.ndim > 1 else 0
    else:
        # 非法 format/rank 由 GEIR/ACLNN 通路负责拒绝；Golden 只生成可比较参考值，
        # 避免在异常值门禁中抢先截断真实的 CANN 拦截结果。
        c_axis = 1 if tensor.ndim > 1 else 0

    # 普通路径严格模拟算子 FP32 计算/输出；Promote 输入已由 TTK 提升，用作高精度真值。
    # Kernel/GEIR 会透传 golden_mode，ACLNN 当前只提升 Tensor dtype、不透传该元信息，
    # 因此 ACLNN 的 FP32 -> FP64 Promote 需要通过实际 dtype 识别。
    if golden_mode == "Promote":
        if tensor.dtype not in _PROMOTED_INPUT_DTYPES:
            raise TypeError(f"Promote 输入应为 float32/float64，实际为 {tensor.dtype}")
        compute_tensor = tensor
    else:
        if tensor.dtype in _NORMAL_INPUT_DTYPES:
            compute_tensor = tensor.float()
        elif golden_mode is None and tensor.dtype == torch.float64:
            compute_tensor = tensor
        else:
            # 不支持 dtype 的判定属于被测通路；Golden 仍升到 FP32 生成参考值。
            compute_tensor = tensor.float()

    axes: Sequence[int] = tuple(i for i in range(tensor.ndim) if i != c_axis)
    sum_t = torch.sum(compute_tensor, dim=tuple(axes), keepdim=False)
    square_sum_t = torch.sum(
        torch.mul(compute_tensor, compute_tensor), dim=tuple(axes), keepdim=False
    )
    return [sum_t, square_sum_t]


def _resolve_format(kwargs) -> str:
    """兼容 Kernel/GEIR 的 input_formats 与 ACLNN 的 tensor_formats。"""
    first_format = None
    for key in ("input_formats", "input_ori_formats", "tensor_formats", "format"):
        formats = kwargs.get(key)
        if formats is None:
            continue
        if isinstance(formats, (list, tuple)):
            if not formats:
                continue
            candidate = formats[0]
        else:
            candidate = formats
        if candidate is None:
            continue
        candidate = str(candidate)
        if first_format is None:
            first_format = candidate
        if candidate.upper() in _SUPPORTED_FORMATS:
            return candidate

    return first_format or "NCHW"


def bn_training_reduce_golden(x, **kwargs):
    """Kernel/GEIR Golden：边界为 NumPy，计算内核统一使用 Torch。"""
    results = _compute_torch(x, _resolve_format(kwargs), kwargs.get("golden_mode"))
    return [result.detach().cpu().numpy() for result in results]


def aclnn_batch_norm_reduce_golden(x, sum=None, squareSum=None, **kwargs):
    """ACLNN Golden：按 GetWorkspaceSize 签名接收并返回 Torch Tensor。"""
    # 正向 ACLNN 只接受 NCHW 4维；负向用例的格式/rank 拒绝由被测接口验证。
    results = _compute_torch(x, "NCHW", kwargs.get("golden_mode"))
    outputs = (sum, squareSum)
    normalized = []
    for result, output in zip(results, outputs):
        if output is None:
            normalized.append(result)
            continue
        output.copy_(
            result.reshape(output.shape).to(device=output.device, dtype=output.dtype)
        )
        normalized.append(output)
    return normalized


def batch_norm_reduce_e2e_golden(input, eps=1.0e-5, **kwargs):
    """E2E Golden：eps 不参与 BNTrainingReduce 的 sum/square_sum 公式。"""
    del eps
    return _compute_torch(input, "NCHW", kwargs.get("golden_mode"))


class BNTrainingReduceCastGraph(torch.nn.Module):
    """显式 Cast 到 BNTrainingReduce 接口 dtype 后调用真实 torch_npu API。"""

    def forward(self, input, eps):
        casted = input.to(torch.float32)
        return torch.ops.npu.batch_norm_reduce(casted, eps)


class _TorchBNTrainingReduce:
    def __call__(self, x=None, **kwargs):
        if x is None:
            x = kwargs.get("input")
        if x is None:
            raise ValueError("必选输入 x/input 不能为空")
        return _compute_torch(x, _resolve_format(kwargs), kwargs.get("golden_mode"))


class BNTrainingReduceKernelSpec:
    golden = staticmethod(bn_training_reduce_golden)
    third_party = {"torch": _TorchBNTrainingReduce}
    tolerance = _L0


class BNTrainingReduceAclnnSpec:
    golden = staticmethod(aclnn_batch_norm_reduce_golden)
    third_party = {"torch": _TorchBNTrainingReduce}
    tolerance = _L0


class BNTrainingReduceE2ESpec:
    golden = staticmethod(batch_norm_reduce_e2e_golden)
    third_party = {"torch": _TorchBNTrainingReduce}
    torch_graph = BNTrainingReduceCastGraph
    tolerance = _L0
