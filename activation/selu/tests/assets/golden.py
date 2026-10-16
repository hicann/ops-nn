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
"""TestSpec definitions for Selu raw op, ACLNN, Torch, and TensorFlow paths.

Selu: y = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
      alpha = 1.6732632423543772
      scale = 1.0507009873554805

Registered names:
  - "selu" / "Selu"            <- CSV op_name (Kernel / GEIR path, numpy input;
                                  "selu" = legacy lowercase dialect,
                                  "Selu" = ascendc-st-design generated dialect)
  - "aclnnSelu"                <- CSV api_name (ACLNN path, torch input)
  - "tf.nn.selu" / "tf.raw_ops.Selu" <- CSV api_name (TensorFlow path; both
                                  keys point to SeluTensorFlowSpec, per OA)

Not registered:
  - torch.selu / torch.nn.functional.selu — the torch path is compound-
    decomposed to aclnnElu (aclnn built-in) and never reaches this
    operator (T7.1 conclusion); torch stays declarative via third_party.

Integer dtype note: torch.selu does not support int8/int32 (NotImplementedError).
  INT32/INT8 golden is computed in float32 then truncated back to the integer
  dtype, matching spec dtype_policy (Cast-FP32 + CAST_TRUNC / Int8-HalfPath).
  Overflow follows the device cast semantics: int8 wraps (torch .to(int8)
  truncation, bit-exact vs numpy astype(int8) over the full int8 domain) and
  int32 saturates (device Cast saturates on overflow; clamped in float64
  because float32 cannot represent INT32_MAX).

Import discipline: torch is imported lazily inside function bodies only —
TTK loads this file inside the aclnn process where aclrt is already
initialized, and a module-level competitor-framework import risks SIGSEGV.
"""

__spec__ = {
    "selu": "SeluKernelTestSpec",
    "Selu": "SeluKernelTestSpec",
    "aclnnSelu": "AclnnSeluTestSpec",
    "tf.nn.selu": "SeluTensorFlowSpec",
    "tf.raw_ops.Selu": "SeluTensorFlowSpec",
    # torch.selu / torch.nn.functional.selu 不注册: torch 路经复合分解走 aclnnElu(aclnn built-in),不经本算子(T7.1)
}

ALPHA = 1.6732632423543772
SCALE = 1.0507009873554805

# Tolerance per the LT baseline: floats use the cross_check standard (third-
# party comparison; LT level L2, OA repo uses L1) with no in-file numeric
# two-way tolerances; int32/int8 binary_equal. Without a competitor runtime,
# TTK is invoked with the CLI-level `--compare mixed` flag so the two-way
# comparison uses TTK's built-in default tolerances; that flag belongs to
# the execution layer and is intentionally not encoded here.
_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L2"},
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "int32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}
_TOLERANCE_FLOAT = {k: _TOLERANCE[k] for k in ("float32", "float16", "bfloat16")}


def _selu_numpy(x):
    """SELU (elementwise) computed with torch on numpy inputs, returns an
    array of the same dtype as ``x``.

    Integer inputs follow spec dtype_policy: compute in float32 then
    truncate back to the integer dtype (torch tensor cast semantics).
    bfloat16 arrives as an ml_dtypes numpy array (TTK kernel/GEIR input
    convention); torch has no numpy interop for that dtype, so it
    round-trips through float32 (exact widening, ml_dtypes RNE cast back).
    """
    import torch

    if x.dtype.name == "bfloat16":
        xf = torch.as_tensor(x.astype("float32"))
        yt = SCALE * (
            torch.clamp(xf, min=0.0) + torch.clamp(ALPHA * torch.expm1(xf), max=0.0)
        )
        return yt.numpy().astype(x.dtype)
    xt = torch.as_tensor(x)
    xf = xt.to(torch.float32)
    yt = SCALE * (
        torch.clamp(xf, min=0.0) + torch.clamp(ALPHA * torch.expm1(xf), max=0.0)
    )
    if xt.dtype is torch.int32:
        yt = _saturate_to_int32_range(yt)
    return yt.to(xt.dtype).numpy()


def _saturate_to_int32_range(y):
    """int32 overflow conversion guard (device Cast saturates on overflow).

    torch ``.to(int32)`` yields INT_MIN on out-of-range floats (x86 cvttss2si
    "integer indefinite"), while the device saturates — T6.3 int32 overflow
    cases (data_range 2147483647) are bit-exact vs saturation. Clamp runs in
    float64: float32 cannot represent INT32_MAX (rounds up to 2147483648.0),
    which would push the bound past the valid range.
    """
    import torch

    info = torch.iinfo(torch.int32)
    return y.to(torch.float64).clamp(float(info.min), float(info.max))


def _selu_torch(x):
    """torch SELU computation; floats use torch.selu, ints use fp32 + trunc."""
    import torch

    if x.dtype in (torch.int8, torch.int32):
        y = SCALE * (
            torch.clamp(x.to(torch.float32), min=0.0)
            + torch.clamp(ALPHA * torch.expm1(x.to(torch.float32)), max=0.0)
        )
        if x.dtype is torch.int32:
            y = _saturate_to_int32_range(y)
        return y.to(x.dtype)
    return torch.selu(x)


class SeluKernelTestSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name`` "selu"/"Selu"
    (numpy input)."""

    def golden(x, **kwargs):
        return [_selu_numpy(x)]

    third_party = {"torch": "torch.selu"}
    tolerance = _TOLERANCE


class AclnnSeluTestSpec:
    """ACLNN spec registered by CSV ``api_name`` (torch input).

    Attention: aclnnSeluGetWorkspaceSize first tensor arg is named ``self``;
    class-form golden renames it to ``selfT`` (positional fallback per
    ttk-how-write-plugin aclnn rules).
    """

    def golden(selfT, out=None, **kwargs):
        result = _selu_torch(selfT)
        if out is not None:
            out.copy_(result)
            return [out]
        return [result]

    third_party = {"torch": "torch.selu"}
    tolerance = _TOLERANCE


class TorchSeluTestSpec:
    """Torch Single/Graph spec (not registered in ``__spec__``; kept for spec
    cross-validation — E2E dispatches F.selu to aclnnElu, not this operator)."""

    def golden(input, inplace=False, **kwargs):
        return [_selu_torch(input)]

    third_party = {"torch": "torch.selu"}
    tolerance = _TOLERANCE_FLOAT


class SeluTensorFlowSpec:
    """TF raw-op spec registered by CSV ``api_name`` "tf.raw_ops.Selu".

    golden keeps the torch primary implementation (independent reference);
    the TF counterpart is referenced declaratively via third_party.
    """

    def golden(features, **kwargs):
        return [_selu_torch(features)]

    third_party = {"tf": "tf.raw_ops.Selu"}
    tolerance = _TOLERANCE
