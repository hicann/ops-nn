# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import ClassVar

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _BF16
except ImportError:
    _BF16 = None


__spec__ = {
    "hard_sigmoid": "HardSigmoidKernelSpec",
    "aclnnHardsigmoid": "HardSigmoidAclnnSpec",
    "aclnnInplaceHardsigmoid": "HardSigmoidAclnnInplaceSpec",
    "torch.nn.functional.hardsigmoid": "HardSigmoidTorchE2ESpec",
}
__input__ = {"kernel": {"hard_sigmoid": "hard_sigmoid_input"}}
__golden__ = {
    "kernel": {"hard_sigmoid": "hard_sigmoid_golden"},
    "aclnn": {
        "aclnnHardsigmoid": "aclnn_hardsigmoid_golden",
        "aclnnInplaceHardsigmoid": "aclnn_inplace_hardsigmoid_golden",
    },
    "e2e": {
        "torch.nn.functional.hardsigmoid": "hard_sigmoid_torch_e2e_golden",
    },
}

_TOL_FLOAT = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}
_TOL_FLOAT_INT32 = {
    **_TOL_FLOAT,
    "int32": {"standard": "binary_equal"},
}

_ONNX_DEFAULT_ALPHA = 0.2
_ONNX_DEFAULT_BETA = 0.5


def _to_torch_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    array = np.ascontiguousarray(x)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


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
        if _BF16 is None:
            raise RuntimeError("ml_dtypes is required for bfloat16 golden output")
        return tensor.contiguous().view(torch.uint16).cpu().numpy().view(_BF16)
    return tensor.cpu().numpy().astype(target_dtype, copy=False)


def hard_sigmoid_golden(input_x, alpha=1.0 / 6.0, beta=0.5, **kwargs):
    result = _hard_sigmoid_golden_compute(input_x, alpha, beta)
    if isinstance(input_x, torch.Tensor):
        target_dtype = (
            np.dtype(np.float32)
            if input_x.dtype == torch.float32
            else np.dtype(np.float16)
            if input_x.dtype == torch.float16
            else np.dtype(np.int32)
            if input_x.dtype == torch.int32
            else None
        )
    else:
        target_dtype = np.asarray(input_x).dtype
    return _torch_to_numpy(result, target_dtype)


def hard_sigmoid_onnx_golden(
    input_x, alpha=_ONNX_DEFAULT_ALPHA, beta=_ONNX_DEFAULT_BETA, **kwargs
):
    """ONNX HardSigmoid semantics for parser-driven Kernel/GEIR cases.

    TTK does not expose an ONNX route.  Keep this callable unregistered and
    exercise it through explicit alpha/beta attributes on Kernel/GEIR cases.
    """
    return hard_sigmoid_golden(input_x, alpha, beta, **kwargs)


def aclnn_hardsigmoid_golden(self, out=None, **kwargs):
    return [_hard_sigmoid_golden_compute(self)]


def aclnn_inplace_hardsigmoid_golden(self, **kwargs):
    return [_hard_sigmoid_golden_compute(self)]


def _promote_e2e_input(x):
    """Mirror TTK Promote for custom E2E golden callables.

    Current TTK promotes raw inputs only on its built-in CPU API path. Custom
    E2E golden functions receive testcase.tensors at the original dtype, so
    they must perform the one-level promotion themselves for cross_check.
    """
    tensor = _to_torch_tensor(x).detach().cpu()
    if tensor.dtype == torch.float32:
        return tensor.to(torch.float64)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


def hard_sigmoid_torch_e2e_golden(input, inplace=False, **kwargs):
    del inplace
    source = _to_torch_tensor(input).detach().cpu()
    promoted = _promote_e2e_input(source)
    result = _hard_sigmoid_golden_compute(promoted)
    # Promotion is only for numerical stability; the E2E contract preserves
    # the input dtype for every supported type, including BF16 and INT32.
    return [result.to(source.dtype)]


class _HardSigmoidCompose:
    def __init__(self, alpha=1.0 / 6.0, beta=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_x, **kwargs):
        return [_hard_sigmoid_third_party_compute(input_x, self.alpha, self.beta)]


class _HardSigmoidTfCompose:
    """Independent TensorFlow CPU composition for kernel cross-checks."""

    def __init__(self, alpha=1.0 / 6.0, beta=0.5, **kwargs):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_x, **kwargs):
        import tensorflow as tf

        x = tf.convert_to_tensor(input_x)
        compute_dtype = (
            tf.float32
            if x.dtype in (tf.float16, tf.bfloat16) or not x.dtype.is_floating
            else x.dtype
        )
        value = tf.cast(x, compute_dtype) * tf.cast(self.alpha, compute_dtype)
        value = value + tf.cast(self.beta, compute_dtype)
        value = tf.clip_by_value(
            value, tf.cast(0.0, compute_dtype), tf.cast(1.0, compute_dtype)
        )
        if not x.dtype.is_floating:
            # tf.cast from floating point to integer is defined as truncation
            # toward zero, matching the Torch reference after clipping.
            value = tf.cast(value, x.dtype)
        return [tf.cast(value, x.dtype)]


class _HardSigmoidAclnnThirdParty:
    def __call__(self, *args, **kwargs):
        del kwargs
        if len(args) != 1:
            raise TypeError("HardSigmoid ACLNN third-party requires one tensor input")
        return [_hard_sigmoid_third_party_compute(args[0])]


class HardSigmoidKernelSpec:
    @staticmethod
    def golden(input_x, alpha=1.0 / 6.0, beta=0.5, **kwargs):
        return [hard_sigmoid_golden(input_x, alpha, beta)]

    third_party: ClassVar[dict] = {
        "torch": _HardSigmoidCompose,
        "tf": _HardSigmoidTfCompose,
    }
    tolerance: ClassVar[dict] = _TOL_FLOAT_INT32


class HardSigmoidAclnnSpec:
    @staticmethod
    def golden(self, out=None, **kwargs):  # noqa: PLW0211
        return aclnn_hardsigmoid_golden(self, out)

    third_party: ClassVar[dict] = {"torch": _HardSigmoidAclnnThirdParty}
    tolerance: ClassVar[dict] = _TOL_FLOAT_INT32


class HardSigmoidAclnnInplaceSpec:
    @staticmethod
    def golden(self, **kwargs):  # noqa: PLW0211
        return aclnn_inplace_hardsigmoid_golden(self)

    third_party: ClassVar[dict] = {"torch": _HardSigmoidAclnnThirdParty}
    tolerance: ClassVar[dict] = _TOL_FLOAT_INT32


class HardSigmoidTorchE2ESpec:
    @staticmethod
    def golden(input, inplace=False, **kwargs):
        return hard_sigmoid_torch_e2e_golden(input, inplace, **kwargs)

    third_party: ClassVar[dict] = {"torch": "torch.nn.functional.hardsigmoid"}
    tolerance: ClassVar[dict] = _TOL_FLOAT


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


# The out-of-place torch API is exercised in both eager and torchair graph
# modes by TTK.
