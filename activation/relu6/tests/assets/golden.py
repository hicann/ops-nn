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

__golden__ = {
    "kernel": {"relu6": "relu6_golden"},
    "e2e": {"torch.nn.functional.relu6": "torch_relu6_e2e_golden"},
}

_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L0"},
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "int32": {"standard": "binary_equal"},
}


def relu6_golden(x, **kwargs):
    """
    Kernel golden for relu6.
    All the parameters (names and order) follow @relu6_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: full_soc_version, short_soc_version, testcase_name,
        {input,output}_{dtypes,ori_shapes,formats,ori_formats}.

    Semantics: y = min(max(x, 0), 6) = clamp(x, 0, 6) (element-wise, same shape).
    FP16/BF16 are promoted to FP32 before the computation and cast back; INT32
    is computed at its own precision.
    """
    import torch

    ori_dtype = x.dtype
    ori_dtype_str = str(ori_dtype).lower()
    is_bf16 = "bfloat16" in ori_dtype_str
    is_fp16 = ori_dtype == np.float16

    if is_bf16 or is_fp16:
        x_compute = x.astype(np.float32)
    else:
        x_compute = x

    x_torch = torch.from_numpy(np.ascontiguousarray(x_compute))
    result = torch.nn.functional.relu6(x_torch)
    return [result.numpy().astype(ori_dtype)]


class _Relu6Compose:
    """Torch reference for the kernel's element-wise clamp semantics."""

    def __call__(self, x, **kwargs):
        del kwargs
        import torch

        output_dtype = x.dtype
        compute = x
        if output_dtype in (torch.float16, torch.bfloat16):
            compute = x.to(torch.float32)
        return [torch.clamp(compute, min=0.0, max=6.0).to(output_dtype)]


class Relu6KernelSpec:
    @staticmethod
    def golden(x, **kwargs):
        return relu6_golden(x, **kwargs)

    third_party = {"torch": _Relu6Compose}
    tolerance = _TOL_KERNEL


def tensorflow_relu6_golden(features, name=None, **kwargs):
    """
    CPU golden following tf.nn.relu6(features, name=None).
    Independent numpy computation; keep the input dtype.
    """
    del name, kwargs
    arr = np.asarray(features)
    result = np.clip(arr, 0, 6)
    if result.dtype != arr.dtype:
        result = result.astype(arr.dtype)
    return [result]


class Relu6TensorFlowSpec:
    """TensorFlow E2E spec registered by the CSV ``api_name``."""

    golden = staticmethod(tensorflow_relu6_golden)
    third_party = {"tf": "tf.nn.relu6"}
    tolerance = _TOL_KERNEL


def torch_relu6_e2e_golden(x, inplace=False, **kwargs):
    """
    E2E golden for torch.nn.functional.relu6.

    Independent CPU torch computation (clamp) rather than calling the tested API.
    The E2E contract preserves the input dtype, including BF16 and INT32.
    """
    del inplace, kwargs
    import torch

    if isinstance(x, torch.Tensor):
        tensor = x.detach().cpu()
    else:
        tensor = torch.as_tensor(np.ascontiguousarray(x))
    if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        result = torch.clamp(tensor, min=0, max=6)
    else:
        result = torch.clamp(tensor, min=0.0, max=6.0)
    return [result]


class Relu6TorchE2ESpec:
    """Torch E2E spec registered by the CSV ``api_name`` torch.nn.functional.relu6."""

    golden = staticmethod(torch_relu6_e2e_golden)
    third_party = {"torch": "torch.nn.functional.relu6"}
    tolerance = _TOL_KERNEL


__spec__ = {
    "relu6": "Relu6KernelSpec",
    "tf.nn.relu6": "Relu6TensorFlowSpec",
    "torch.nn.functional.relu6": "Relu6TorchE2ESpec",
}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec) + TensorFlow E2E(tf.nn.relu6)
#         + Torch E2E(torch.nn.functional.relu6)。
# aclnn: 未交付——算子 CMakeLists 设置 ACLNNTYPE aclnn_exclude 且无 op_api/ 目录，
#        未注册 aclnnRelu6。
# Torch 图模式: 未交付——无 torch_extension，未声明 torch_graph。
# 融合 pass / ONNX: 未交付。
