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

"""SGD multi-path TestSpec golden (kernel + GEIR).

The CPU true-value path is a composition of Torch tensor operators.  Its
arithmetic is kept identical to the legacy ``sgd_golden`` entry:

    grad     = gradient + parameters * weight_decay      (when weight_decay != 0)
    accum_t  = accum * momentum + grad                   (unconditional)
    accum_t -= grad * (1 - stat) * dampening             (when dampening != 0)
    p_out    = p - (grad*lr + accum_t*m*lr)               (Nesterov)
               p - accum_t*lr                            (otherwise)

``momentum == 0`` only masks the accum/stat writeback; it never skips the
``accum * momentum`` calculation.  Consequently ``0 * inf`` still propagates
NaN into parameters according to IEEE 754, while accum/stat retain their input
bit patterns (including NaN payloads and negative zero).

The third-party implementation deliberately differs only in the Nesterov
parenthesization: arch35 ``sgd_dag.h`` computes
``p - (grad + accum_t * momentum) * learning_rate`` to satisfy its TensorScalar
slot constraint.  Cross-check must reproduce that actual device operation
order, then cast all three writeback values to the NPU input dtype.

Promote contract: half/bfloat16 inputs may be lifted to float32 for CPU support;
float32 and float64 inputs are never narrowed.  TTK promotes both inputs and
``output_dtypes`` before invoking this Spec when ``cross_check`` is active.
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _NP_BFLOAT16
except ImportError:  # pragma: no cover - TTK normally provides ml_dtypes.
    _NP_BFLOAT16 = None


__spec__ = {"sgd": "SgdKernelSpec"}

# Compatibility entry for existing kernel CSVs and callers.  Both entries call
# the same _compute function; this mapping must remain until all old loaders have
# migrated to TestSpec.
__golden__ = {"kernel": {"sgd": "sgd_golden"}}


_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    """Read a TTK attribute and normalize CSV string values."""
    value = kwargs.get(name)
    if value is None and isinstance(kwargs.get("attributes"), dict):
        value = kwargs["attributes"].get(name)
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if isinstance(default, bool):
            if text in ("true", "yes", "1"):
                return True
            if text in ("false", "no", "0"):
                return False
        try:
            return type(default)(value)
        except (TypeError, ValueError):
            return default
    return value


def _numpy_to_torch(array):
    """Convert a contiguous NumPy/ML-dtypes array without changing its value.

    ``torch.from_numpy`` does not accept ML-dtypes bfloat16.  Viewing its bits as
    uint16 and then as ``torch.bfloat16`` is lossless and also supports the raw
    uint16 representation used by a few legacy tests.
    """
    value = np.ascontiguousarray(np.asarray(array))
    if value.dtype == np.uint16 or value.dtype.name == "bfloat16":
        bits = value.view(np.uint16)
        return torch.from_numpy(bits).view(torch.bfloat16)
    return torch.from_numpy(value)


def _f32_floor(tensor):
    """Lift only fp16/bf16; preserve fp32/fp64 supplied by Promote."""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


def _scalar_value(tensor):
    """Read an already-quantized scalar without rebuilding it from a literal."""
    return float(_f32_floor(tensor).reshape(-1)[0].item())


def _compute(
    parameters,
    gradient,
    learning_rate,
    accum,
    momentum,
    stat,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    **kwargs,
):
    """Torch competitor composition shared by TestSpec and legacy entry.

    Inputs and outputs are Torch tensors.  This is the high-precision CPU
    true-value path, so it keeps the legacy expanded Nesterov formula; the
    independent ``_SgdCompose`` below follows the grouped arch35 DAG order.
    """
    p = _f32_floor(parameters)
    g = _f32_floor(gradient)
    a = _f32_floor(accum)
    s = _f32_floor(stat)
    learning_rate_value = _scalar_value(learning_rate)
    momentum_value = _scalar_value(momentum)

    attrs = {
        "dampening": dampening,
        "weight_decay": weight_decay,
        "nesterov": nesterov,
        **kwargs,
    }
    dampening_value = float(_attr(attrs, "dampening", 0.0))
    weight_decay_value = float(_attr(attrs, "weight_decay", 0.0))
    use_nesterov = bool(_attr(attrs, "nesterov", False))

    # Step 1: the zero branch is genuinely absent; writing ``p * 0`` would
    # incorrectly turn an infinite parameter into NaN.
    grad = g + p * weight_decay_value if weight_decay_value != 0.0 else g

    # Step 2 is unconditional.  In particular, momentum==0 does not permit
    # eliding this multiply because 0*inf must remain NaN.
    accum_t = a * momentum_value + grad

    # Step 3: likewise, dampening==0 removes the whole sub-expression.
    if dampening_value != 0.0:
        accum_t = accum_t - grad * ((1.0 - s) * dampening_value)

    # Preserve the already-validated legacy true-value ordering.  This is
    # mathematically equivalent to the grouped DAG form, but intentionally
    # independent from it for cross-checking.
    if use_nesterov:
        p_new = p - (
            grad * learning_rate_value + accum_t * momentum_value * learning_rate_value
        )
    else:
        p_new = p - accum_t * learning_rate_value

    if momentum_value != 0.0:  # IEEE !=: -0.0 is zero; NaN is non-zero.
        return [p_new, accum_t, torch.zeros_like(s)]

    # No numeric reconstruction on the masked writeback path.  Keeping the raw
    # tensors is what preserves NaN payloads, infinities and the sign of zero.
    return [p_new, accum.clone(), stat.clone()]


def _normalize_dtype_name(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, (tuple, list)):
        dtype = dtype[0] if dtype else None
    if dtype is None:
        return None
    name = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "fp64": "float64",
        "double": "float64",
    }
    return aliases.get(name, name)


def _output_dtype_names(kwargs):
    dtypes = kwargs.get("output_dtypes") or ()
    names = [_normalize_dtype_name(dtype) for dtype in dtypes]
    if len(names) == 1:
        # SGD exposes one graph output but the golden also returns the two
        # in-place writeback slots; all six inputs are constrained to one dtype.
        names *= 3
    return names


def _torch_to_numpy(tensor, target_dtype=None, reference=None):
    """Convert one result to the requested TTK/NPU dtype."""
    result = tensor.detach().cpu().contiguous()
    target_name = _normalize_dtype_name(target_dtype)
    ref = np.asarray(reference) if reference is not None else None

    if target_name is None and ref is not None:
        target_name = ref.dtype.name

    if target_name == "bfloat16":
        bits = result.to(torch.bfloat16).view(torch.uint16).numpy().copy()
        if _NP_BFLOAT16 is None:
            # Loader/import remains usable without ml_dtypes; TTK installations
            # that execute bf16 cases provide it and therefore take the branch
            # above with the semantic bfloat16 dtype.
            return result.to(torch.float32).numpy()
        return bits.view(_NP_BFLOAT16)

    # Legacy raw-bf16 input representation.  A masked writeback tensor is still
    # bfloat16 and must preserve its bits; computed float tensors retain the old
    # numeric astype(uint16) behaviour.
    if target_name == "uint16" and result.dtype == torch.bfloat16:
        return result.view(torch.uint16).numpy().copy()

    if result.dtype == torch.bfloat16:
        result = result.to(torch.float32)
    array = result.numpy()
    return array.astype(np.dtype(target_name), copy=False) if target_name else array


def _kernel_golden(
    parameters,
    gradient,
    learning_rate,
    accum,
    momentum,
    stat,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    **kwargs,
):
    """Kernel/GEIR container shell: NumPy in, NumPy out."""
    inputs = [
        _numpy_to_torch(value)
        for value in (parameters, gradient, learning_rate, accum, momentum, stat)
    ]
    outputs = _compute(
        *inputs,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        **kwargs,
    )
    references = (parameters, accum, stat)
    output_dtypes = _output_dtype_names(kwargs)
    return [
        _torch_to_numpy(
            output,
            output_dtypes[index] if index < len(output_dtypes) else None,
            references[index],
        )
        for index, output in enumerate(outputs)
    ]


def sgd_golden(
    parameters,
    gradient,
    learning_rate,
    accum,
    momentum,
    stat,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    **kwargs,
):
    """Compatibility ``__golden__`` entry; delegates to the shared core."""
    return _kernel_golden(
        parameters,
        gradient,
        learning_rate,
        accum,
        momentum,
        stat,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        **kwargs,
    )


class _SgdCompose:
    """GPU competitor matching the arch35 DAG operation order exactly."""

    def __init__(
        self,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        **kwargs,
    ):
        attrs = {
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            **kwargs,
        }
        # Host tiling stores both float attributes as float32 before selecting
        # the zero/non-zero DAG branch, so quantize attributes the same way.
        self.dampening = float(
            torch.tensor(
                float(_attr(attrs, "dampening", 0.0)), dtype=torch.float32
            ).item()
        )
        self.weight_decay = float(
            torch.tensor(
                float(_attr(attrs, "weight_decay", 0.0)), dtype=torch.float32
            ).item()
        )
        self.nesterov = bool(_attr(attrs, "nesterov", False))

    def __call__(
        self,
        parameters,
        gradient,
        learning_rate,
        accum,
        momentum,
        stat,
        **kwargs,
    ):
        del kwargs
        npu_dtype = parameters.dtype

        # arch35 casts every input to float32 before the elementwise DAG.
        p = parameters.to(torch.float32)
        g = gradient.to(torch.float32)
        a = accum.to(torch.float32)
        s = stat.to(torch.float32)
        learning_rate_value = learning_rate.to(torch.float32).reshape(-1)[0]
        momentum_value = momentum.to(torch.float32).reshape(-1)[0]

        # Step 1: OpParamMulWd -> OpGradWithWd.
        if self.weight_decay != 0.0:
            parameter_decay = torch.mul(p, self.weight_decay)
            grad = torch.add(g, parameter_decay)
        else:
            grad = g

        # Step 2: OpAccMulMom -> OpAccumTBase.
        accum_momentum = torch.mul(a, momentum_value)
        accum_t = torch.add(accum_momentum, grad)

        # Step 3: OpStatNeg -> OpStatAct -> OpStatActMulD -> OpDampTerm.
        if self.dampening != 0.0:
            stat_negative = torch.mul(s, -1.0)
            stat_active = torch.add(stat_negative, 1.0)
            stat_active_damped = torch.mul(stat_active, self.dampening)
            dampening_term = torch.mul(grad, stat_active_damped)
            accum_t = torch.sub(accum_t, dampening_term)

        # Step 4: reproduce the real grouped Nesterov DAG, not the expanded
        # legacy true-value expression.
        if self.nesterov:
            accum_t_momentum = torch.mul(accum_t, momentum_value)
            nesterov_sum = torch.add(grad, accum_t_momentum)
            update = torch.mul(nesterov_sum, learning_rate_value)
        else:
            update = torch.mul(accum_t, learning_rate_value)
        parameters_out = torch.sub(p, update).to(npu_dtype)

        if bool(torch.ne(momentum_value, 0.0).item()):
            accum_out = accum_t.to(npu_dtype)
            stat_out = torch.zeros_like(s).to(npu_dtype)
        else:
            # Match the no-writeback branch, including payload/sign bits.
            accum_out = accum.clone()
            stat_out = stat.clone()

        return [parameters_out, accum_out, stat_out]


class SgdKernelSpec:
    """Kernel and GEIR share the snake_case ``sgd`` registration."""

    def golden(
        parameters,
        gradient,
        learning_rate,
        accum,
        momentum,
        stat,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        **kwargs,
    ):
        return _kernel_golden(
            parameters,
            gradient,
            learning_rate,
            accum,
            momentum,
            stat,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs,
        )

    third_party = {"torch": _SgdCompose}
    tolerance = _TOL


# 【不存在】ACLNN 通路：optim/sgd/CMakeLists.txt 使用 ACLNNTYPE
# aclnn_exclude，且算子目录没有 op_api/aclnn_sgd 实现；因此 __spec__ 不注册 aclnnSgd。
# 【不存在】e2e 通路：需求明确不提供本算子的 torch_npu 绑定，也没有可调用的
# aclnnSgd 底层接口；因此不保留 TorchSpec 空壳。
