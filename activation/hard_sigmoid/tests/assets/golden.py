#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch


__golden__ = {"kernel": {"hard_sigmoid": "hard_sigmoid_golden"}}
__input__ = {"kernel": {"hard_sigmoid": "hard_sigmoid_input"}}


def hard_sigmoid_golden(x, alpha=1.0 / 6.0, beta=0.5, **kwargs):
    """Compute HardSigmoid through PyTorch operator composition."""
    del kwargs
    input_dtype = x.dtype
    if input_dtype.name == "bfloat16":
        input_tensor = torch.tensor(
            np.asarray(x, dtype=np.float32), dtype=torch.bfloat16
        )
    else:
        input_tensor = torch.from_numpy(np.asarray(x))
    alpha32 = float(np.float32(alpha))
    beta32 = float(np.float32(beta))
    result = torch.clamp(
        input_tensor.to(torch.float32) * alpha32 + beta32, min=0.0, max=1.0
    )
    if not input_tensor.is_floating_point():
        result = torch.trunc(result)
    result = result.to(input_tensor.dtype)
    return (
        result.to(torch.float32).cpu().numpy()
        if input_dtype.name == "bfloat16"
        else result.cpu().numpy()
    )


def hard_sigmoid_input(x, alpha=1.0 / 6.0, beta=0.5, **kwargs):
    """Inject clamp boundaries and special values while retaining each case's requested dtype and shape."""
    if x.size == 0:
        return [x]

    testcase_name = kwargs.get("testcase_name", "")
    if testcase_name == "hard_sigmoid_fp32_special":
        tiny = np.finfo(np.float32).tiny
        critical = np.array(
            [
                -np.inf,
                -8.0,
                -3.0001,
                -3.0,
                -2.9999,
                -1.0,
                -tiny,
                -0.0,
                0.0,
                tiny,
                1.0,
                2.9999,
                3.0,
                3.0001,
                8.0,
                np.inf,
                np.nan,
            ],
            dtype=np.float32,
        )
    else:
        alpha32 = np.float32(alpha)
        beta32 = np.float32(beta)
        if alpha32 == 0:
            critical = np.array([-8.0, -1.0, 0.0, 1.0, 8.0], dtype=np.float32)
        else:
            zero_boundary = -beta32 / alpha32
            one_boundary = (np.float32(1.0) - beta32) / alpha32
            epsilon = np.float32(1.0e-3)
            critical = np.array(
                [
                    zero_boundary - epsilon,
                    zero_boundary,
                    zero_boundary + epsilon,
                    -1.0,
                    0.0,
                    1.0,
                    one_boundary - epsilon,
                    one_boundary,
                    one_boundary + epsilon,
                ],
                dtype=np.float32,
            )

    result = np.array(x, copy=True)
    flat = result.reshape(-1)
    count = min(flat.size, critical.size)
    flat[:count] = critical[:count].astype(result.dtype, copy=False)
    return [flat.reshape(result.shape)]
