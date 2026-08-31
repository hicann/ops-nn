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
import torch

__golden__ = {
    "aclnn": {
        "aclnnGather": "aclnn_gather_golden",
    }
}


def aclnn_gather_golden(self, dim, index, out=None, **kwargs):
    """
    Aclnn golden for aclnnGather.
    Parameters follow @aclnnGatherGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """

    x = self
    tensor_x = x
    index = index
    dim = dim
    np_out = torch.gather(input=tensor_x, dim=dim, index=index)

    return np_out
