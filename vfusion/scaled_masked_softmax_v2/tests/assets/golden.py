#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
ScaledMaskedSoftmaxV2 golden spec for TTK.

y = softmax(masked(x * scale), dim=-1); masked positions are filled with the
compute-dtype minimum (not -inf):
  - float16 x: scale and fill happen in the float16 domain (fill = -65504),
    softmax accumulates in float32, and the result is cast back to float16;
  - float32 / bfloat16 x: scale and fill happen in float32 (fill = -3.4e+38),
    softmax in float32; bfloat16 results are cast back to the input dtype.

aclnn pathway (`aclnnScaledMaskedSoftmax`): arguments arrive positionally in
aclnnScaledMaskedSoftmaxGetWorkspaceSize(x, mask, scale, fixedTriuMask, y, ...)
parameter order (without workspaceSize/executor) as torch.Tensor / Python
scalars. `y` is the framework-allocated output slot and its incoming value is
ignored; `**kwargs` absorbs framework-injected run metadata (testcase_name,
tensor_dtypes, short_soc_version, ...). `fixedTriuMask=True` is rejected with
a ValueError, mirroring the op host check (ACLNN_ERR_PARAM_INVALID, 161002).
"""

import torch

__spec__ = {
    "aclnnScaledMaskedSoftmax": "AclnnScaledMaskedSoftmaxTestSpec",
}


class AclnnScaledMaskedSoftmaxTestSpec:
    def golden(x, mask, scale=None, fixedTriuMask=False, y=None, **kwargs):
        if bool(fixedTriuMask):
            raise ValueError(
                "aclnnScaledMaskedSoftmax only supports fixedTriuMask=False "
                "(op rejects true with ACLNN_ERR_PARAM_INVALID); no golden reference exists"
            )
        scale = 1.0 if scale is None else float(scale)
        if x.dtype == torch.float16:
            scaled = x * scale
            fill = torch.finfo(torch.float16).min
        else:
            scaled = x.to(torch.float32) * scale
            fill = torch.finfo(torch.float32).min
        masked = scaled.masked_fill(mask, fill)
        out = torch.softmax(masked.to(torch.float32), dim=-1)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return [out]
