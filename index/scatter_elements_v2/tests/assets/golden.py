import torch
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


__golden__ = {
    "aclnn": {
        "aclnnScatter": "aclnn_scatter_golden",
    },
    "kernel": {"scatter_elements_v2": "scatter_elements_v2_golden"},
}


def scatter_elements_v2_golden(
    var, indices, updates, *, axis=0, reduction="none", include_self=True, **kwargs
):
    """
    Golden function for scatter_elements_v2.
    All the parameters (names and order) follow @scatter_elements_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    x_dtype = var.dtype
    if "bfloat16" in str(x_dtype):
        var = var.astype("float32")
        updates = updates.astype("float32")
    tensor_x = torch.from_numpy(var).clone()
    tensor_index = torch.from_numpy(indices).to(torch.int64)
    tensor_src = torch.from_numpy(updates)

    if reduction == "add":
        out = torch.scatter_reduce(
            tensor_x, axis, tensor_index, tensor_src, reduce="sum"
        )
    elif reduction == "mul":
        out = torch.scatter_reduce(
            tensor_x, axis, tensor_index, tensor_src, reduce="prod"
        )
    else:
        out = torch.scatter(tensor_x, axis, tensor_index, tensor_src)
    res = out.numpy()
    if "bfloat16" in str(x_dtype):
        res = res.astype(x_dtype, copy=False)
    return res


def aclnn_scatter_golden(self, dim, index, src, reduce, out=None, **kwargs):
    """
    Aclnn golden for aclnnScatter.
    Parameters follow @aclnnScatterGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    tensor_x = self
    tensor_src = src

    if hasattr(dim, "item"):
        dim = dim.item()
    if hasattr(reduce, "item"):
        reduce = reduce.item()

    tensor_index = index.to(torch.int64)
    if reduce == 1:
        out = torch.scatter_reduce(
            tensor_x, dim, tensor_index, tensor_src, reduce="sum"
        )
    elif reduce == 2:
        out = torch.scatter_reduce(
            tensor_x, dim, tensor_index, tensor_src, reduce="prod"
        )
    else:
        out = torch.scatter(tensor_x, dim, tensor_index, tensor_src)
    return out
