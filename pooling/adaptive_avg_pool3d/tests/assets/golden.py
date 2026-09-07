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
    "aclnn": {
        "aclnnAdaptiveAvgPool3d": "aclnn_adaptive_avg_pool3d_golden",
    },
    "kernel": {"adaptive_avg_pool3d": "adaptive_avg_pool3d_golden"},
}


def adaptive_avg_pool3d_golden(x, output_size, data_format="NDHWC", **kwargs):
    """
    Kernel golden for adaptive_avg_pool3d.
    All the parameters follow @adaptive_avg_pool3d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn as nn

    xDtype = x.dtype
    data_format = data_format if data_format != "" else "NDHWC"

    m = nn.AdaptiveAvgPool3d(output_size)

    if xDtype in ["float16", "bfloat16"]:
        x = x.astype(np.float32)  ## torch cpu不支持fp16, bf16不支持from_numpy
    x_tensor = torch.from_numpy(x)

    # NDHWC先转NCDHW后再进行运算，运算后再转回NDHWC, 支持4/5维转换
    if data_format == "NDHWC" and len(x_tensor.shape) == 5:
        x_tensor = x_tensor.permute(0, 4, 1, 2, 3)
    elif data_format == "NDHWC" and len(x_tensor.shape) == 4:
        x_tensor = x_tensor.permute(3, 0, 1, 2)

    output = m(x_tensor)

    # 运算后，结果转置回NDHWC
    if data_format == "NDHWC" and len(x_tensor.shape) == 5:
        output = output.permute(0, 2, 3, 4, 1)
    elif data_format == "NDHWC" and len(x_tensor.shape) == 4:
        output = output.permute(1, 2, 3, 0)

    output = output.numpy().astype(xDtype)

    return output


def aclnn_adaptive_avg_pool3d_golden(self, outputSize=0, out=None, **kwargs):
    """
    Aclnn golden for aclnnAdaptiveAvgPool3d.
    Parameters follow @aclnnAdaptiveAvgPool3dGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    if hasattr(outputSize, "tolist"):
        outputSize = outputSize.tolist()
    elif isinstance(outputSize, int):
        outputSize = [outputSize] * 3
    orig_dtype = self.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        self = self.to(torch.float32)
    result = torch.nn.functional.adaptive_avg_pool3d(self, outputSize)
    return [result.to(orig_dtype)]
