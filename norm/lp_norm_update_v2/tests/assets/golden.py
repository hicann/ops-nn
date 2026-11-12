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

"""LpNormUpdateV2 独立 Golden：raw。"""

import math

import torch

__spec__ = {
    "lp_norm_update_v2": "LpNormUpdateV2KernelSpec",
    "LpNormUpdateV2": "LpNormUpdateV2KernelSpec",
}

_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}

_INT_MAX = 2147483647.0
_INT_MAX_F32 = 2147483648.0
_INT_MIN = -2147483648.0


def _compute(x, p=2.0, epsilon=1e-12):
    import torch
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    is_torch = isinstance(x, torch.Tensor)
    tensor = x if is_torch else numpy_to_torch_tensor(x)

    output_dtype = tensor.dtype
    value = tensor.to(torch.float32)

    float_max = torch.finfo(torch.float32).max
    p_value = float(p)

    if p_value in (0.0, 1.0, _INT_MAX, _INT_MAX_F32, _INT_MIN) or math.isinf(p_value):
        result = value
    elif math.isclose(p_value, 2.0):
        result = torch.sqrt(torch.clamp(value, min=torch.finfo(torch.float32).tiny))
    else:
        inverse_p = 1.0 / p_value
        if math.isinf(inverse_p):
            inverse_p = math.copysign(float(float_max), inverse_p)
        result = torch.exp(
            torch.log(torch.clamp(value, min=torch.finfo(torch.float32).tiny))
            * inverse_p
        )

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


def _get_torch_input(args, kwargs):
    """从位置参数中取第一个 torch.Tensor, 否则按别名取, 均无则报错"""
    x = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if x is not None:
        return x
    for name in ("x", "self", "input", "input_x"):
        x = kwargs.get(name)
        if x is not None:
            return x
    raise ValueError(
        "Missing tensor input; expected positional Tensor or one of: x, self, input, input_x"
    )


class _TorchLpNormUpdateV2:
    def __init__(self, *, p=2.0, epsilon=1e-12, **kwargs):
        del kwargs
        self.p = float(p)
        self.epsilon = float(epsilon)

    def __call__(self, *args, **kwargs):
        import torch

        x = _get_torch_input(args, kwargs)
        del kwargs

        FLT_MIN = 1.1754943508e-38
        p_value = self.p

        if p_value in (0.0, 1.0, _INT_MAX, _INT_MAX_F32, _INT_MIN) or math.isinf(
            p_value
        ):
            result = x
        elif math.isclose(p_value, 2.0):
            result = torch.sqrt(torch.clamp(x, min=FLT_MIN))
        else:
            inverse_p = 1.0 / p_value
            if math.isinf(inverse_p):
                inverse_p = math.copysign(3.4028234663852886e38, inverse_p)
            result = torch.exp(
                torch.mul(
                    torch.log(torch.clamp(x, min=FLT_MIN)),
                    inverse_p,
                )
            )

        eps_value = self.epsilon
        if x.dtype == torch.float16 and 0.0 < eps_value <= 1e-7:
            eps_value = 1e-7
        eps_tensor = torch.tensor(eps_value, dtype=x.dtype, device=x.device)
        return [torch.maximum(result, eps_tensor)]


class LpNormUpdateV2KernelSpec:
    golden = staticmethod(lp_norm_update_v2_golden)
    third_party = {"torch": _TorchLpNormUpdateV2}
    tolerance = _L2
