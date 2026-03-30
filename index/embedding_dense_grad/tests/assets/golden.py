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

__golden__ = {"kernel": {"embedding_dense_grad": "embedding_dense_grad_golden"}}


def embedding_dense_grad_golden(grad, indices, *, num_weights, padding_idx=-1, scale_grad_by_freq=False, **kwargs):
    '''
    Golden function for embedding_dense_grad.
    All the parameters (names and order) follow @embedding_dense_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    grad_dtype = grad.dtype

    if 'float16' in grad_dtype.name:
        grad = grad.astype('float32')
    grad = torch.from_numpy(grad)
    indices = torch.from_numpy(indices)

    result = torch.ops.aten.embedding_dense_backward(grad, indices, int(num_weights), padding_idx,
                                                     scale_grad_by_freq)
    if 'float16' in grad_dtype.name:
        result = result.numpy().astype(grad_dtype, copy=False)
    else:
        result = result.numpy()
    return result
