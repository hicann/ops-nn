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
        "aclnnBucketize": "aclnn_bucketize_golden",
    },
    "kernel": {"bucketize_v2": "bucketize_v2_golden"},
}


def bucketize_v2_golden(x, boundaries, out_int32=False, right=False, **kwargs):
    """
    Kernel golden for bucketize_v2.
    All the parameters follow @bucketize_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch

    data_t = torch.from_numpy(x)
    boundaries_t = torch.from_numpy(boundaries)
    res = torch.bucketize(data_t, boundaries_t, out_int32=out_int32, right=right)
    return res.numpy()


def aclnn_bucketize_golden(self, boundaries, outInt32=0, right=0, out=None, **kwargs):
    if hasattr(outInt32, "item"):
        outInt32 = bool(outInt32.item())
    elif isinstance(outInt32, int):
        outInt32 = bool(outInt32)
    if hasattr(right, "item"):
        right = bool(right.item())
    elif isinstance(right, int):
        right = bool(right)
    return [torch.bucketize(self, boundaries, out_int32=outInt32, right=right)]
