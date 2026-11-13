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

import sys
from typing import Sequence

import numpy as np

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


def _as_torch(x):
    """兼容 NumPy 输入，已有 Torch Tensor 保留其设备。"""
    import torch

    if isinstance(x, torch.Tensor):
        return x
    array = np.asarray(x)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _reduction_axes(rank: int, data_format: str) -> Sequence[int]:
    """保留通道轴，归约其他轴；非法格式/rank 仍由被测通路校验。"""
    fmt = str(data_format).upper()
    if fmt == "NHWC":
        c_axis = max(rank - 1, 0)
    elif fmt in ("NCHW", "NCDHW"):
        c_axis = 1 if rank > 1 else 0
    else:
        # 非法 format/rank 由 GEIR/ACLNN 通路负责拒绝；Golden 只生成可比较参考值，
        # 避免在异常值门禁中抢先截断真实的 CANN 拦截结果。
        c_axis = 1 if rank > 1 else 0
    return tuple(i for i in range(rank) if i != c_axis)


def _compute_torch(x, data_format: str) -> list:
    """CPU 使用 FP64 平方和归约，GPU 使用 FP32，两路结果均舍入到 FP32。"""
    import torch

    tensor = _as_torch(x)
    compute_dtype = torch.float64 if tensor.device.type == "cpu" else torch.float32
    compute_tensor = tensor.to(dtype=compute_dtype)
    axes = _reduction_axes(tensor.ndim, data_format)
    sum_t = torch.sum(compute_tensor, dim=tuple(axes), keepdim=False)
    square_sum_t = torch.sum(
        torch.mul(compute_tensor, compute_tensor), dim=tuple(axes), keepdim=False
    )
    return [sum_t.to(torch.float32), square_sum_t.to(torch.float32)]


def _cpu_array_fp64(x):
    """保持实际输入值，BF16 先精确扩展到 FP32，再统一转换为 FP64。"""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.numpy()
    array = np.asarray(x)
    if array.dtype.name == "bfloat16":
        array = array.astype(np.float32)
    return array.astype(np.float64, copy=False)


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
    """Kernel/GEIR：TensorFlow CPU FP64 计算，返回 FP32 NumPy 结果。"""
    import tensorflow as tf

    with tf.device("/CPU:0"):
        tensor = tf.convert_to_tensor(_cpu_array_fp64(x), dtype=tf.float64)
        axes = _reduction_axes(tensor.shape.rank, _resolve_format(kwargs))
        results = [
            tf.reduce_sum(tensor, axis=axes),
            tf.reduce_sum(tf.multiply(tensor, tensor), axis=axes),
        ]
    if any(
        value.dtype != tf.float64 or "CPU" not in value.device.upper()
        for value in results
    ):
        raise RuntimeError("TensorFlow CPU Golden dtype/device mismatch")
    with tf.device("/CPU:0"):
        rounded = [tf.cast(value, tf.float32) for value in results]
    if any(
        value.dtype != tf.float32 or "CPU" not in value.device.upper()
        for value in rounded
    ):
        raise RuntimeError("TensorFlow CPU Golden output cast mismatch")
    return [value.numpy() for value in rounded]


def aclnn_batch_norm_reduce_golden(x, sum=None, squareSum=None, **kwargs):
    """ACLNN Golden：按 GetWorkspaceSize 签名接收并返回 Torch Tensor。"""
    # 正向 ACLNN 只接受 NCHW 4维；负向用例的格式/rank 拒绝由被测接口验证。
    results = _compute_torch(x, "NCHW")
    outputs = (sum, squareSum)
    normalized = []
    for result, output in zip(results, outputs):
        if output is None:
            normalized.append(result)
            continue
        # 若调用方提供 FP64 buffer，先舍入到 FP32，再回填以保留输出布局和对象身份。
        output.copy_(
            result.reshape(output.shape).to(device=output.device, dtype=output.dtype)
        )
        normalized.append(output)
    return normalized


def batch_norm_reduce_e2e_golden(input, eps=1.0e-5, **kwargs):
    """E2E Golden：eps 不参与 BNTrainingReduce 的 sum/square_sum 公式。"""
    del eps
    return _compute_torch(input, "NCHW")


class _TorchGraph:
    """仅 E2E 访问 torch_graph 时加载 Torch，并返回真正的 nn.Module 子类。"""

    def __init__(self):
        self._module_cls = None

    def __get__(self, instance, owner=None):
        if self._module_cls is None:
            import torch

            class BNTrainingReduceCastGraph(torch.nn.Module):
                """显式 Cast 到接口 dtype 后调用真实 torch_npu API。"""

                def forward(self, input, eps):
                    casted = input.to(torch.float32)
                    return torch.ops.npu.batch_norm_reduce(casted, eps)

            self._module_cls = BNTrainingReduceCastGraph
        return self._module_cls


class _TorchBNTrainingReduce:
    def __call__(self, x=None, **kwargs):
        if x is None:
            x = kwargs.get("input")
        if x is None:
            raise ValueError("必选输入 x/input 不能为空")
        return _compute_torch(x, _resolve_format(kwargs))


class _TensorFlowBNTrainingReduce:
    """由 TTK 注入 GPU 设备，使用 TensorFlow FP32 平方和归约。"""

    def __init__(self, device):
        self.device = str(device)

    def __call__(self, x=None, **kwargs):
        import tensorflow as tf

        if x is None:
            x = kwargs.get("input")
        if x is None:
            raise ValueError("必选输入 x/input 不能为空")
        if "GPU" not in self.device.upper():
            raise RuntimeError(
                "TensorFlow GPU reference requires a GPU device: " + self.device
            )
        with tf.device(self.device):
            tensor = tf.identity(tf.cast(tf.convert_to_tensor(x), tf.float32))
            axes = _reduction_axes(tensor.shape.rank, _resolve_format(kwargs))
            results = [
                tf.reduce_sum(tensor, axis=axes),
                tf.reduce_sum(tf.multiply(tensor, tensor), axis=axes),
            ]
        if any(
            value.dtype != tf.float32 or "GPU" not in value.device.upper()
            for value in results
        ):
            raise RuntimeError(
                "TensorFlow GPU reference fell back to CPU or changed dtype"
            )
        return results


class BNTrainingReduceKernelSpec:
    golden = staticmethod(bn_training_reduce_golden)
    third_party = {"tf": _TensorFlowBNTrainingReduce}
    tolerance = _L0


class BNTrainingReduceAclnnSpec:
    golden = staticmethod(aclnn_batch_norm_reduce_golden)
    third_party = {"torch": _TorchBNTrainingReduce}
    tolerance = _L0


class BNTrainingReduceE2ESpec:
    golden = staticmethod(batch_norm_reduce_e2e_golden)
    third_party = {"torch": _TorchBNTrainingReduce}
    torch_graph = _TorchGraph()
    tolerance = _L0
