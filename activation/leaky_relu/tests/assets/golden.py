#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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
    "aclnn": {
        "aclnnLeakyRelu": "aclnn_leaky_relu_golden",
        "aclnnInplaceLeakyRelu": "aclnn_inplace_leaky_relu_golden",
    },
    "kernel": {"leaky_relu": "leaky_relu_golden"},
}


def leaky_relu_golden(x, *, negative_slope=0, **kwargs):
    """
    Golden function for leaky_relu.
    All the parameters (names and order) follow @leaky_relu_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: Input tensor.
        negative_slope: Negative slope value (default: 0).
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    if "bfloat16" in x.dtype.name:
        x_torch = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    else:
        x_torch = torch.from_numpy(x)

    result = torch.ops.aten.leaky_relu(x_torch, negative_slope)
    if "bfloat16" in x.dtype.name:
        return result.view(torch.int16).numpy().view(x.dtype)
    return result.numpy()


def aclnn_inplace_leaky_relu_golden(selfRef, negativeSlope, **kwargs):
    """
    Aclnn golden for aclnnInplaceLeakyRelu.
    Parameters follow @aclnnInplaceLeakyReluGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    if hasattr(negativeSlope, "item"):
        negativeSlope = negativeSlope.item()
    return [torch.nn.functional.leaky_relu(selfRef, negative_slope=negativeSlope)]


def aclnn_leaky_relu_golden(self, negativeSlope, out=None, **kwargs):
    """
    Aclnn golden for aclnnLeakyRelu.
    Parameters follow @aclnnLeakyReluGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    if hasattr(negativeSlope, "item"):
        negativeSlope = negativeSlope.item()
    return [torch.nn.functional.leaky_relu(self, negative_slope=negativeSlope)]


# ----------------------------------------------------------------------------
# E2E 通路（纯新增，上方存量 kernel/aclnn golden 与 __golden__ 注册保持原样）:
# torch.ops.aten.leaky_relu 的 NPU 侧由 torch_npu 的 PrivateUse1 kernel 分派到
# aclnnLeakyRelu（本算子交付的 API），golden 侧直接调 ATen CPU kernel，两侧实现
# 来源独立。tolerance 不另行声明，走框架缺省（浮点 mix_tolerance / 整型逐位比对）。
# ----------------------------------------------------------------------------
__spec__ = {"torch.ops.aten.leaky_relu": "LeakyReluE2eSpec"}


def leaky_relu_e2e_golden(input, negative_slope=0.0, **kwargs):
    """
    E2E golden for torch.ops.aten.leaky_relu.
    Parameters follow the aten schema (self / negative_slope) without outputs;
    tensors are CPU torch.Tensor bound positionally via the param plan.
    Runs the ATen CPU kernel as the reference against the torch_npu dispatch
    (aten::leaky_relu -> aclnnLeakyRelu).

    Returns:
        [output]
    """
    import torch

    if hasattr(negative_slope, "item"):
        negative_slope = negative_slope.item()
    return [torch.nn.functional.leaky_relu(input, negative_slope=float(negative_slope))]


class LeakyReluE2eSpec:
    golden = staticmethod(leaky_relu_e2e_golden)
