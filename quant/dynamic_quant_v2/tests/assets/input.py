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

__input__ = {
    "kernel": {"dynamic_quant_v2": "dynamic_quant_v2_input"},
    "aclnn": {"aclnnDynamicQuantV2": "aclnn_dynamic_quant_v2_input"},
}


def dynamic_quant_v2_input(x, smooth_scales=None, group_index=None, **kwargs):
    """
    Input function for dynamic_quant_v2.
    All the input Tensors are numpy.ndarray.
    """
    if group_index is not None:
        S = np.prod(x.shape[:-1])
        E = group_index.shape[0]
        group_index = np.random.choice(np.arange(1, S + 1), size=E, replace=False)
        group_index.sort()
        group_index[-1] = S
        group_index = group_index.astype("int32")
    return [x, smooth_scales, group_index]


def aclnn_dynamic_quant_v2_input(*args, **kwargs):
    """
    Input function for aclnnDynamicQuantV2.
    All the input Tensors are torch.Tensor.
    TTK passes all tensor args; we only need to process group_index (3rd arg).
    """
    import torch

    x = args[0] if len(args) > 0 else kwargs.get("x")
    smooth_scales = args[1] if len(args) > 1 else kwargs.get("smoothScalesOptional")
    group_index = args[2] if len(args) > 2 else kwargs.get("groupIndexOptional")

    if group_index is not None:
        S = np.prod(x.shape[:-1])
        E = group_index.shape[0]
        gi = np.random.choice(np.arange(1, S + 1), size=E, replace=False)
        gi.sort()
        gi[-1] = S
        group_index = torch.from_numpy(gi.astype("int32"))
    return [x, smooth_scales, group_index]
