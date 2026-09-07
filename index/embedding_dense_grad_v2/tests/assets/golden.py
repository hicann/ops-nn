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
        "aclnnEmbeddingDenseBackward": "aclnn_embedding_dense_backward_golden",
    },
    "kernel": {"embedding_dense_grad_v2": "embedding_dense_grad_v2_golden"},
}


def embedding_dense_grad_v2_golden(
    grad,
    sort_indices,
    pos_idx,
    *,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    **kwargs,
):
    """
    Golden function for embedding_dense_grad_v2.
    All the parameters (names and order) follow @embedding_dense_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    grad_dtype = grad.dtype

    if "float16" in grad_dtype.name:
        grad = grad.astype("float32")

    indices_shape = sort_indices.shape
    grad = torch.from_numpy(grad)
    sort_indices = torch.from_numpy(sort_indices).reshape(-1)
    pos_idx = torch.from_numpy(pos_idx).to(torch.int64).reshape(-1)

    indices = torch.zeros_like(sort_indices)
    indices.scatter_(-1, pos_idx, sort_indices)
    indices = indices.reshape(indices_shape)

    result = torch.ops.aten.embedding_dense_backward(
        grad, indices, int(num_weights), padding_idx, scale_grad_by_freq
    )

    if "float16" in grad_dtype.name:
        result = result.numpy().astype(grad_dtype, copy=False)
    else:
        result = result.numpy()
    return result


def aclnn_embedding_dense_backward_golden(
    grad, indices, numWeights=0, paddingIdx=0, scaleGradByFreq=0, out=None, **kwargs
):
    """
    Aclnn golden for aclnnEmbeddingDenseBackward.
    Parameters follow @aclnnEmbeddingDenseBackwardGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    grad_dtype = grad.dtype
    indices_dtype = indices.dtype

    if grad_dtype == torch.float16 or grad_dtype == torch.bfloat16:
        grad = grad.to(torch.float32)

    if indices_dtype != torch.int32 and indices_dtype != torch.int64:
        if indices_dtype == torch.float64:
            indices = indices.to(torch.int64)
        else:
            indices = indices.to(torch.int32)
    attrs = kwargs.get("attributes", {})
    num_weights = attrs.get("numWeights", -1)
    padding_idx = attrs.get("paddingIdx", -1)
    scale_grad_by_freq = attrs.get("scaleGradByFreq", False)
    result = torch.ops.aten.embedding_dense_backward(
        grad, indices, num_weights, padding_idx, scale_grad_by_freq
    )
    if grad_dtype == torch.float16 or grad_dtype == torch.bfloat16:
        result = result.to(grad_dtype)
    return result
