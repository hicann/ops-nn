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
from copy import deepcopy

__golden__ = {
    "aclnn": {
        "aclnnMaxPool3dWithArgmax": "aclnn_max_pool3d_with_argmax_golden",
        "aclnnMaxPool2dWithIndices": "aclnn_max_pool2d_with_indices_golden",
    },
    "kernel": {"max_pool3d_with_argmax_v2": "max_pool3d_with_argmax_v2_golden"},
}


def max_pool3d_with_argmax_v2_golden(
    x,
    *,
    ksize,
    strides,
    pads,
    dilation=[1, 1, 1],
    ceil_mode=False,
    data_format="NCDHW",
    dtype=3,
    **kwargs,
):
    """
    Golden function for max_pool3d_with_argmax_v2.
    All the parameters (names and order) follow @max_pool3d_with_argmax_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (y, argmax)
    """
    import torch
    import torch.nn as nn

    input_x = deepcopy(x)
    input_x_format = kwargs["input_formats"][0]
    input_x_dtype = input_x.dtype

    out_y_format = kwargs["output_formats"][0]
    out_argmax_format = kwargs["output_formats"][1]

    attr_re_ksize = ksize
    if len(attr_re_ksize) == 3:
        attr_re_ksize = [attr_re_ksize[0], attr_re_ksize[1], attr_re_ksize[2]]

    attr_re_strides = strides
    if len(attr_re_strides) == 3:
        attr_re_strides = [attr_re_strides[0], attr_re_strides[1], attr_re_strides[2]]

    attr_re_pads = pads
    if len(attr_re_pads) == 3:
        attr_re_pads = [attr_re_pads[0], attr_re_pads[1], attr_re_pads[2]]

    attr_op_dtype = dtype
    attr_op_dilations = dilation
    if attr_op_dilations is not None and len(attr_op_dilations) == 3:
        attr_op_dilations = [
            attr_op_dilations[0],
            attr_op_dilations[1],
            attr_op_dilations[2],
        ]

    attr_op_ceil_mode = ceil_mode
    attr_op_format = data_format

    input_x_format = attr_op_format

    # NDHWC -> NCDHW (if needed)
    if input_x_format == "NDHWC":
        input_x = input_x.transpose(0, 4, 1, 2, 3)

    # Convert to float32 if input is float16/bfloat16
    if str(input_x_dtype) in ("float16", "bfloat16"):
        input_x = input_x.astype(np.float32)

    input_x = torch.from_numpy(input_x)

    # Torch MaxPool3d setup
    attr = {
        "kernel_size": attr_re_ksize,
        "stride": attr_re_strides,
        "padding": attr_re_pads,
        "return_indices": True,
    }
    if attr_op_dilations is not None:
        attr["dilation"] = attr_op_dilations
    if attr_op_ceil_mode is not None:
        attr["ceil_mode"] = attr_op_ceil_mode

    cpuMaxPool3d = nn.MaxPool3d(**attr)
    max_out, max_indices = cpuMaxPool3d(input_x)

    if str(input_x_dtype) == "bfloat16":
        out_y = max_out.numpy().astype(input_x_dtype, copy=False)
    else:
        out_y = max_out.numpy().astype(input_x_dtype)
    if out_y_format == "NDHWC":
        out_y = out_y.transpose(0, 2, 3, 4, 1)

    if attr_op_dtype == 3 or attr_op_dtype is None:
        out_argmax = max_indices.numpy().astype(np.int32)
    elif attr_op_dtype == 9:
        out_argmax = max_indices.numpy().astype(np.int64)
    else:
        out_argmax = max_indices.numpy()

    if out_argmax_format == "NDHWC":
        out_argmax = out_argmax.transpose(0, 2, 3, 4, 1)
    return out_y, out_argmax


def aclnn_max_pool3d_with_argmax_golden(
    self,
    kernelSize=0,
    stride=0,
    padding=0,
    dilation=0,
    ceilMode=0,
    out=None,
    indices=None,
    **kwargs,
):
    """
    Aclnn golden for aclnnMaxPool3dWithArgmax.
    Parameters follow @aclnnMaxPool3dWithArgmaxGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch
    import torch.nn.functional as F
    from copy import deepcopy

    input_x = deepcopy(self)
    inpu_x_dtype = input_x.dtype
    input_x = input_x.to(torch.float32)
    input_x_format = kwargs.get("tensor_formats", ["NCDHW"])[0]
    output_indices = deepcopy(indices) if indices is not None else None
    output_indices_dtype = (
        output_indices.dtype if output_indices is not None else torch.int64
    )
    if input_x_format == "NDHWC":
        input_x = input_x.permute(0, 4, 1, 2, 3)
    elif input_x_format == "NHWC":
        input_x = input_x.permute(3, 0, 1, 2)

    attr = {"return_indices": True}
    attr["kernel_size"] = kwargs.get("attributes", {}).get("kernelSize", kernelSize)
    attr["stride"] = kwargs.get("attributes", {}).get("stride", stride)
    attr["padding"] = kwargs.get("attributes", {}).get("padding", padding)
    attr["dilation"] = kwargs.get("attributes", {}).get("dilation", dilation)
    attr["ceil_mode"] = kwargs.get("attributes", {}).get("ceilMode", ceilMode)

    out_y, out_argmax = F.max_pool3d(input_x, **attr)

    out_y = out_y.to(inpu_x_dtype)
    out_argmax = out_argmax.to(output_indices_dtype)
    if input_x_format == "NDHWC":
        out_y = out_y.permute(0, 2, 3, 4, 1)
        out_argmax = out_argmax.permute(0, 2, 3, 4, 1)
    elif input_x_format == "NHWC":
        out_y = out_y.permute(1, 2, 3, 0)
        out_argmax = out_argmax.permute(1, 2, 3, 0)

    return out_y, out_argmax


def aclnn_max_pool2d_with_indices_golden(
    self,
    kernelSize=0,
    stride=0,
    padding=0,
    dilation=0,
    ceilMode=0,
    out=None,
    indices=None,
    **kwargs,
):
    """
    Aclnn golden for aclnnMaxPool2dWithIndices.
    Parameters follow @aclnnMaxPool2dWithIndicesGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch
    import torch.nn as nn
    from copy import deepcopy

    input_x = deepcopy(self)

    inpu_x_dtype = input_x.dtype

    input_x_format = kwargs.get("tensor_formats", ["NCHW"])[0]
    output_indices = deepcopy(indices) if indices is not None else None
    output_indices_dtype = (
        output_indices.dtype if output_indices is not None else torch.int64
    )
    attrs = kwargs.get("attributes", {})
    attr_re_ksize = attrs.get("kernelSize", kernelSize) if attrs else kernelSize
    if isinstance(attr_re_ksize, list) and len(attr_re_ksize) == 4:
        attr_re_ksize = [attr_re_ksize[1], attr_re_ksize[2]]
    attr_re_strides = attrs.get("stride", stride) if attrs else stride
    if isinstance(attr_re_strides, list) and len(attr_re_strides) == 4:
        attr_re_strides = [attr_re_strides[1], attr_re_strides[2]]
    attr_re_pads = attrs.get("padding", padding) if attrs else padding
    if isinstance(attr_re_pads, list) and len(attr_re_pads) == 4:
        attr_re_pads = [attr_re_pads[1], attr_re_pads[2]]

    attr_op_dilations = attrs.get("dilation", dilation) if attrs else dilation
    if isinstance(attr_op_dilations, list) and len(attr_op_dilations) == 4:
        attr_op_dilations = [attr_op_dilations[1], attr_op_dilations[2]]
    if attr_op_dilations == 0:
        attr_op_dilations = None

    attr_op_ceil_mode = attrs.get("ceilMode", ceilMode) if attrs else ceilMode
    if attr_op_ceil_mode == 0:
        attr_op_ceil_mode = False

    if input_x_format == "NHWC":
        input_x = input_x.permute(0, 3, 1, 2)

    if "float16" == str(inpu_x_dtype) or "bfloat16" == str(inpu_x_dtype):
        input_x = input_x.to(torch.float32)

    attr = {
        "kernel_size": attr_re_ksize,
        "stride": attr_re_strides,
        "padding": attr_re_pads,
    }
    if attr_op_dilations is not None:
        attr["dilation"] = attr_op_dilations
    if attr_op_ceil_mode is not None or attr_op_ceil_mode is False:
        attr["ceil_mode"] = bool(attr_op_ceil_mode) if attr_op_ceil_mode else False
    attr["return_indices"] = True
    cpuMaxPool = nn.MaxPool2d(**attr)
    max_out, max_indices = cpuMaxPool(input_x)

    if "bfloat16" == str(inpu_x_dtype):
        out_y = max_out.to(inpu_x_dtype, copy=False)
    else:
        out_y = max_out.to(inpu_x_dtype)
    if input_x_format == "NHWC":
        out_y = out_y.permute(0, 2, 3, 1)

    if str(output_indices_dtype) == "torch.int32":
        out_argmax = max_indices.to(torch.int32)
    else:
        out_argmax = max_indices.to(torch.int64)

    if input_x_format == "NHWC":
        out_argmax = out_argmax.permute(0, 2, 3, 1)

    return out_y, out_argmax
