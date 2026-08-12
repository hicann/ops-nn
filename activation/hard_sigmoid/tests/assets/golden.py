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


__spec__ = {
    "hard_sigmoid": "HardSigmoidKernelSpec",
    "aclnnHardsigmoid": "HardSigmoidAclnnSpec",
    "aclnnInplaceHardsigmoid": "HardSigmoidAclnnInplaceSpec",
}
__input__ = {"kernel": {"hard_sigmoid": "hard_sigmoid_input"}}

_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}
_TOL_LOCAL = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
    "int32": {"standard": "binary_equal"},
}


def _to_torch_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    input_dtype = np.asarray(x).dtype
    if input_dtype.name == "bfloat16":
        return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.bfloat16)
    return torch.from_numpy(np.asarray(x))


def _prepare_hard_sigmoid_input(x, alpha, beta):
    input_tensor = _to_torch_tensor(x)
    if (
        input_tensor.dtype in (torch.float16, torch.bfloat16)
        or not input_tensor.is_floating_point()
    ):
        compute_tensor = input_tensor.to(torch.float32)
        alpha_value = float(np.float32(alpha))
        beta_value = float(np.float32(beta))
    else:
        compute_tensor = input_tensor
        alpha_value = float(alpha)
        beta_value = float(beta)
    return input_tensor, compute_tensor, alpha_value, beta_value


def _hard_sigmoid_golden_compute(x, alpha=1.0 / 6.0, beta=0.5):
    input_tensor, compute_tensor, alpha_value, beta_value = _prepare_hard_sigmoid_input(
        x, alpha, beta
    )
    linear = compute_tensor * alpha_value + beta_value
    result = torch.where(
        linear <= 0.0,
        torch.zeros_like(linear),
        torch.where(linear >= 1.0, torch.ones_like(linear), linear),
    )
    if not input_tensor.is_floating_point():
        result = torch.trunc(result)
    return result.to(input_tensor.dtype)


def _hard_sigmoid_third_party_compute(x, alpha=1.0 / 6.0, beta=0.5):
    input_tensor, compute_tensor, alpha_value, beta_value = _prepare_hard_sigmoid_input(
        x, alpha, beta
    )
    result = torch.clamp(compute_tensor * alpha_value + beta_value, min=0.0, max=1.0)
    if not input_tensor.is_floating_point():
        result = torch.trunc(result)
    return result.to(input_tensor.dtype)


def _torch_to_numpy(tensor, target_dtype):
    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float32).cpu().numpy()
    return tensor.cpu().numpy().astype(target_dtype, copy=False)


class _HardSigmoidCompose:
    def __init__(self, alpha=1.0 / 6.0, beta=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_x, **kwargs):
        return [_hard_sigmoid_third_party_compute(input_x, self.alpha, self.beta)]


def _hard_sigmoid_aclnn_third_party(
    self, alpha=1.0 / 6.0, beta=0.5, out=None, **kwargs
):
    return [_hard_sigmoid_third_party_compute(self, alpha, beta)]


class HardSigmoidKernelSpec:
    @staticmethod
    def golden(input_x, alpha=1.0 / 6.0, beta=0.5, **kwargs):
        result = _hard_sigmoid_golden_compute(input_x, alpha, beta)
        return [_torch_to_numpy(result, np.asarray(input_x).dtype)]

    third_party = {"torch": _HardSigmoidCompose}
    tolerance = _TOL_KERNEL


class HardSigmoidAclnnSpec:
    @staticmethod
    def golden(self, out=None, **kwargs):
        return [_hard_sigmoid_golden_compute(self)]

    third_party = {"torch": _hard_sigmoid_aclnn_third_party}
    tolerance = _TOL_LOCAL


class HardSigmoidAclnnInplaceSpec:
    @staticmethod
    def golden(self, **kwargs):
        return [_hard_sigmoid_golden_compute(self)]

    third_party = {"torch": _hard_sigmoid_aclnn_third_party}
    tolerance = _TOL_LOCAL


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


# Not registered in __spec__: no e2e, TensorFlow, ONNX, or fusion path is delivered.
