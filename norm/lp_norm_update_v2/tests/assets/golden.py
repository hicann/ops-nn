#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ==============================================================================================

"""LpNormUpdateV2 独立 Golden：Torch。"""

import math

import torch

__spec__ = {
    "lp_norm_update_v2": "LpNormUpdateV2KernelSpec",
    "LpNormUpdateV2": "LpNormUpdateV2KernelSpec",
}

_FLT_MIN = float(torch.finfo(torch.float32).tiny)
_P32_POS_INF = float(torch.tensor(2**31, dtype=torch.float32).item())
_P32_NEG_INF = float(torch.tensor(-(2**31), dtype=torch.float32).item())
_IDENTITY_P32 = {0.0, 1.0, _P32_POS_INF, _P32_NEG_INF}


def _to_p32(p):
    return float(torch.tensor(float(p), dtype=torch.float32).item())


def _is_identity_p(p):
    p32 = _to_p32(p)
    return p32 in _IDENTITY_P32 or math.isinf(p32)


def _compute(x, p=2.0, epsilon=1e-12):
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    is_torch = isinstance(x, torch.Tensor)
    tensor = x if is_torch else numpy_to_torch_tensor(x)

    output_dtype = tensor.dtype
    value = tensor.to(torch.float64)
    if tensor.numel() == 0:
        return tensor if is_torch else torch_to_numpy_tensor(tensor.detach().cpu())

    p_value = float(p)

    if _is_identity_p(p_value):
        result = value
    elif math.isclose(p_value, 2.0):
        result = torch.sqrt(torch.clamp(value, min=_FLT_MIN))
    else:
        result = torch.pow(torch.clamp(value, min=_FLT_MIN), 1.0 / p_value)

    epsilon_value = float(epsilon)
    if output_dtype == torch.float16 and 0.0 < epsilon_value <= 1e-7:
        epsilon_value = 1e-7
    epsilon_tensor = torch.tensor(
        epsilon_value, dtype=result.dtype, device=result.device
    )
    result = torch.maximum(result, epsilon_tensor).to(output_dtype)

    return result if is_torch else torch_to_numpy_tensor(result.detach().cpu())


def lp_norm_update_v2_golden(x, p=2.0, epsilon=1e-12, **kwargs):
    del kwargs
    return [_compute(x, p, epsilon)]


class _TorchLpNormUpdateV2:
    def __init__(self, x, *, p=2.0, epsilon=1e-12, **kwargs):
        del kwargs
        self.x = x
        self.p = float(p)
        self.epsilon = float(epsilon)

    def __call__(self, *args, **kwargs):
        del args, kwargs
        return [_compute(self.x, self.p, self.epsilon)]


class LpNormUpdateV2KernelSpec:
    golden = staticmethod(lp_norm_update_v2_golden)
    third_party = {"torch": _TorchLpNormUpdateV2}
