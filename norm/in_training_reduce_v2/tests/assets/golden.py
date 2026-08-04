#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
INTrainingReduceV2 golden reference.

语义: InstanceNorm 训练前向 reduce 阶段。对每个 (n, c) 实例，在空间轴上求和/平方和。
  sum        = Σ x        (over spatial axes dim2..end)
  square_sum = Σ x^2      (over spatial axes dim2..end)
保留 dim0=N、dim1=C，归约 dim2..末（NCHW 归 H,W；NCDHW 归 D,H,W；ND 归 dim2..末）。
fp16 输入按 kernel 行为先升 fp32 再累加；输出恒 fp32。

R3: kernel golden 使用 torch.sum / torch.square 竞品接口。
"""

import numpy as np
import torch

__golden__ = {
    "kernel": {"in_training_reduce_v2": "in_training_reduce_v2_golden"},
}


def in_training_reduce_v2_golden(x, *args, **kwargs):
    """
    Golden function for in_training_reduce_v2 (kernel level).
    All the parameters (names and order) follow @in_training_reduce_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Uses torch.sum / torch.square competitive interfaces (R3).

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of (sum, square_sum), both numpy.ndarray of dtype float32.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.ascontiguousarray(x))
    x32 = x.to(torch.float32)
    if x32.ndim > 2:
        dims = list(range(2, x32.ndim))
        s = torch.sum(x32, dim=dims, keepdim=True)
        sq = torch.sum(torch.square(x32), dim=dims, keepdim=True)
    else:
        s = x32.clone()
        sq = torch.square(x32)
    return s.to(torch.float32).contiguous().numpy(), sq.to(
        torch.float32
    ).contiguous().numpy()
