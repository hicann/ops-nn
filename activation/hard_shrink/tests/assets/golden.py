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
"""HardShrink 独立 Golden：raw、ACLNN、Torch 单算子/图模式。"""

import torch

__spec__ = {
    "HardShrink": "HardShrinkKernelSpec",
    "hard_shrink": "HardShrinkKernelSpec",
    "hardshrink": "HardShrinkKernelSpec",
    "aclnnHardshrink": "HardShrinkAclnnSpec",
    "torch.nn.functional.hardshrink": "HardShrinkTorchSpec",
}

_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}


def _normalize_lambd(lambd):
    """按算子声明的 FLOAT 属性精度解释 lambd，再交给参考实现按输入 dtype 比较。

    算子属性 lambd 是 float32；TTK 在 mix_tolerance/cross_check 下会把输入张量升精度，
    若不先把 lambd 舍入到 float32，边界 |x|==lambd 会与算子产生相反结论。
    """
    import numpy as np

    return float(np.float32(float(lambd)))


def _compute(x, lambd=0.5):
    import torch
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    is_torch = isinstance(x, torch.Tensor)
    tensor = x if is_torch else numpy_to_torch_tensor(x)
    result = torch.nn.functional.hardshrink(tensor, lambd=_normalize_lambd(lambd))
    return result if is_torch else torch_to_numpy_tensor(result.detach().cpu())


def hard_shrink_golden(self, lambd=0.5, **kwargs):
    del kwargs
    return [_compute(self, lambd)]


def aclnn_hard_shrink_golden(self, lambd=0.5, out=None, **kwargs):
    del kwargs
    result = _compute(self, lambd)
    if out is not None:
        result = result.reshape(out.shape).to(out.dtype)
    return [result]


def torch_hard_shrink_golden(input, lambd=0.5, **kwargs):
    del kwargs
    return [_compute(input, lambd)]


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


class _TorchHardShrink:
    def __init__(self, lambd=0.5):
        self.lambd = _normalize_lambd(lambd)

    def __call__(instance, *args, **kwargs):
        import torch

        x = _get_torch_input(args, kwargs)
        return [torch.nn.functional.hardshrink(x, lambd=instance.lambd)]


class _HardShrinkGraph(torch.nn.Module):
    def __init__(self, lambd=0.5, **kwargs):
        super().__init__()
        del kwargs
        self.lambd = _normalize_lambd(lambd)

    def forward(self, input, **kwargs):
        del kwargs
        return torch.nn.functional.hardshrink(input, lambd=self.lambd)


class HardShrinkKernelSpec:
    golden = staticmethod(hard_shrink_golden)
    third_party = {"torch": _TorchHardShrink}
    tolerance = _L2


class HardShrinkAclnnSpec:
    golden = staticmethod(aclnn_hard_shrink_golden)
    third_party = {"torch": _TorchHardShrink}
    tolerance = _L2


class HardShrinkTorchSpec:
    golden = staticmethod(torch_hard_shrink_golden)
    third_party = {"torch": "torch.nn.functional.hardshrink"}
    torch_graph = _HardShrinkGraph
    tolerance = _L2
