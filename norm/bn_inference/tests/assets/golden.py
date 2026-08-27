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

"""BNInference kernel/GEIR golden in the TestSpec multi-path format.

Pathway support copied from the approved requirement baseline:

| pathway | support | evidence |
|---------|---------|----------|
| kernel  | yes | ``op_kernel/`` contains the arch35 implementation |
| GEIR    | yes | ``op_graph/`` contains the guarded ``REG_OP(BNInference)`` |
| ACLNN   | no  | no ``op_host/op_api`` implementation or ACLNN symbol |
| e2e     | no  | the installed torch_npu binary has no ``aclnnBNInference`` symbol |

The CPU golden follows the two observable canndev ``BnHost`` foldings::

    no scale/offset:
      factor = momentum.flat[0] == 0 ? 0 : 1 / momentum.flat[0]
      alpha = -factor * mean
      beta = 1 / sqrt(factor * variance + epsilon)
      y = (x + alpha) * beta

    scale and offset present:
      s = sqrt(variance + epsilon)
      inv_s = 1 / s
      beta = scale * inv_s
      alpha = (offset / scale) * s - mean
      y = (x + alpha) * beta

For nonzero ``mode``, the scale-only and offset-only combinations are an
Ascend950-compatible extension and retain the standard inference BatchNorm
expression with the missing affine parameter replaced by one or zero.  With
``mode == 0``, ``mean`` and ``variance`` carry pre-folded alpha and beta and
the golden evaluates ``(x + alpha) * beta`` followed by optional scale and
offset.  Offset without scale is rejected in this mode.  This is an intentional
Ascend950 extension: canndev ignores ``mode`` and therefore still executes the
full BN path for ``mode == 0``.  TTK-promoted inputs retain their promoted
precision.  The independent competitor uses native PyTorch tensor primitives
for the ordered pointwise paths and native inference BatchNorm for the
nonzero-mode one-sided paths.  ``use_global_stats`` remains an ABI-compatible
attribute and does not affect the result.
"""

import ast

import numpy as np
import torch


# Kernel and GEIR use the same snake-case registration key and Spec class.
__spec__ = {"bn_inference": "BNInferenceKernelSpec"}

# Compatibility entry for the repository's legacy kernel golden loader.  It
# delegates to the same computation core as the TestSpec entry below.
__golden__ = {"kernel": {"bn_inference": "bn_inference_golden"}}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_TTK_SPEC_CONTEXT_KEY = "__ttk_spec_context__"


def _numpy_dtype(dtype):
    """Resolve NumPy dtypes, including the optional bfloat16 extension."""
    name = _normalize_dtype_name(dtype)
    if name == "bfloat16":
        try:
            from ml_dtypes import bfloat16
        except ImportError as exc:
            raise RuntimeError(
                "BNInference bfloat16 golden requires the ml-dtypes package"
            ) from exc
        return bfloat16
    return np.dtype(name)


def _normalize_dtype_name(dtype):
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0] if dtype else None
    if dtype is None:
        return None
    name = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    return {
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "bf16": "bfloat16",
    }.get(name, name)


def _as_torch_tensor(value):
    """Convert a NumPy/torch input without losing a BF16 payload."""
    if isinstance(value, torch.Tensor):
        return value
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _to_numpy(tensor):
    """Convert a torch output to contiguous NumPy, including BF16."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(_numpy_dtype("bfloat16"))
    return np.ascontiguousarray(tensor.numpy())


def _sequence_value(value):
    """Parse the CSV string representation used by older TTK versions."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped.upper().startswith("FORMAT_"):
        return value
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value


def _attribute_value(kwargs, name, default):
    """Resolve a direct attribute and the nested CSV/TestSpec representation."""
    value = kwargs.get(name, default)
    nested_attributes = _sequence_value(kwargs.get("attributes"))
    if isinstance(nested_attributes, dict):
        value = nested_attributes.get(name, value)
    return value


def _format_name(values):
    """Return the first normalized format name from one metadata value."""
    values = _sequence_value(values)
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        if not values:
            return None
        value = values[0]
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
    else:
        value = values
    if value is not None:
        return str(value).upper().replace("FORMAT_", "")
    return None


def _first_format(kwargs, key):
    """Read direct metadata first, then the TestSpec-only reserved context."""
    direct_format = _format_name(kwargs.get(key))
    if direct_format is not None:
        return direct_format

    spec_context = _sequence_value(kwargs.get(_TTK_SPEC_CONTEXT_KEY))
    if isinstance(spec_context, dict):
        return _format_name(spec_context.get(key))
    return None


def _channel_axis(x, channel_count, kwargs):
    """Resolve the public storage layout and its logical origin format."""
    if x.ndim not in (4, 5):
        raise ValueError(f"x rank must be 4 or 5, got {x.ndim}")

    storage_format = _first_format(kwargs, "input_formats")
    origin_format = _first_format(kwargs, "input_ori_formats")
    if storage_format is not None:
        if storage_format == "ND":
            logical_format = origin_format or "ND"
            if logical_format == "ND":
                # Match the legacy canndev BNInferenceD fallback for a plain ND descriptor.
                return 1
            if logical_format in ("NCHW", "NHWC") and x.ndim != 4:
                raise ValueError(
                    f"unsupported x storage/origin/rank: ND/{logical_format}/{x.ndim}"
                )
            if logical_format in ("NCDHW", "NDHWC") and x.ndim != 5:
                raise ValueError(
                    f"unsupported x storage/origin/rank: ND/{logical_format}/{x.ndim}"
                )
            if logical_format not in ("NCHW", "NHWC", "NCDHW", "NDHWC"):
                raise ValueError(f"unsupported ND origin format: {logical_format}")
            return -1 if logical_format in ("NHWC", "NDHWC") else 1
        if storage_format in ("NCHW", "NHWC") and x.ndim == 4:
            return -1 if storage_format == "NHWC" else 1
        if storage_format in ("NCDHW", "NDHWC") and x.ndim == 5:
            return -1 if storage_format == "NDHWC" else 1
        if storage_format not in ("NCHW", "NHWC", "NCDHW", "NDHWC"):
            raise ValueError(
                f"unsupported x format/rank pair: {storage_format}/{x.ndim}"
            )
        raise ValueError(f"unsupported x format/rank pair: {storage_format}/{x.ndim}")

    if origin_format is not None:
        if origin_format == "ND":
            return 1
        if origin_format in ("NCHW", "NHWC") and x.ndim == 4:
            return -1 if origin_format == "NHWC" else 1
        if origin_format in ("NCDHW", "NDHWC") and x.ndim == 5:
            return -1 if origin_format == "NDHWC" else 1
        raise ValueError(f"unsupported x origin/rank pair: {origin_format}/{x.ndim}")

    # Older/direct callers may omit format metadata.  A unique shape/C match is
    # then the only sound fallback.  Never guess an ambiguous nonempty layout:
    # doing so would create a false cross-check result for shapes whose axis 1
    # and last axis both equal C.
    candidates = []
    if x.shape[1] == channel_count:
        candidates.append(1)
    if x.shape[-1] == channel_count:
        candidates.append(-1)
    if not candidates:
        raise ValueError("mean length does not match any public channel axis")
    if len(candidates) != 1:
        raise ValueError(
            "ambiguous channel axis without input format metadata; use a "
            "cross-check case whose axis 1 and last axis are not both C"
        )
    return candidates[0]


def _validate_empty_input(x, channel_count, scale, offset, kwargs):
    """Validate an empty input without guessing between equivalent layouts."""
    if x.ndim not in (4, 5):
        raise ValueError(f"x rank must be 4 or 5, got {x.ndim}")
    for name, value in (("scale", scale), ("offset", offset)):
        if value is not None and value.numel() != channel_count:
            raise ValueError(
                f"{name} must contain C={channel_count} elements, got {value.numel()}"
            )

    # The output of an empty tensor is layout-independent.  When format
    # metadata is available, still validate the declared channel dimension;
    # otherwise require at least one public channel axis to match C, but do not
    # reject the case merely because both empty axes match.
    has_format = (
        _first_format(kwargs, "input_formats") is not None
        or _first_format(kwargs, "input_ori_formats") is not None
    )
    if has_format:
        channel_axis = _channel_axis(x, channel_count, kwargs)
        if x.shape[channel_axis] != channel_count:
            raise ValueError("mean length does not match the public channel axis")
    elif x.shape[1] != channel_count and x.shape[-1] != channel_count:
        raise ValueError("mean length does not match any public channel axis")


def _parameter_compute(
    value, channel_count, rank, channel_axis, name, reference, compute_dtype
):
    if value is None:
        return None
    tensor = _as_torch_tensor(value)
    if tensor.numel() != channel_count:
        raise ValueError(
            f"{name} must contain C={channel_count} elements, got {tensor.numel()}"
        )
    shape = [1] * rank
    shape[channel_axis] = channel_count
    return torch.reshape(tensor.to(device=reference.device, dtype=compute_dtype), shape)


def _golden_compute_dtype(*tensors):
    """Keep TTK Promote inputs; only raise CPU FP16/BF16 to FP32."""
    return (
        torch.float64
        if any(
            tensor is not None and tensor.dtype == torch.float64 for tensor in tensors
        )
        else torch.float32
    )


def _momentum_factor(momentum, reference, compute_dtype):
    """Read only momentum[0], matching canndev BnHost's public 4-input path."""
    momentum_tensor = _as_torch_tensor(momentum)
    if momentum_tensor.numel() == 0:
        raise ValueError("momentum must contain at least one element")
    first = momentum_tensor.reshape(-1)[0].to(
        device=reference.device, dtype=compute_dtype
    )
    zero = torch.zeros_like(first)
    one = torch.ones_like(first)
    return torch.where(torch.eq(first, zero), zero, torch.div(one, first))


def _compute(
    x,
    mean,
    variance,
    momentum,
    scale=None,
    offset=None,
    epsilon=1e-5,
    mode=1,
    **kwargs,
):
    """Single high-precision Torch golden core for both mode families."""
    x_tensor = _as_torch_tensor(x)
    mean_tensor = _as_torch_tensor(mean)
    variance_tensor = _as_torch_tensor(variance)
    momentum_tensor = _as_torch_tensor(momentum)
    scale_tensor = None if scale is None else _as_torch_tensor(scale)
    offset_tensor = None if offset is None else _as_torch_tensor(offset)
    if mean_tensor.numel() != variance_tensor.numel():
        raise ValueError("mean and variance must have the same C elements")

    channel_count = mean_tensor.numel()
    optional_mask = int(scale_tensor is not None) | (
        int(offset_tensor is not None) << 1
    )
    pre_folded = int(mode) == 0
    if pre_folded and optional_mask == 2:
        raise ValueError("mode=0 requires scale when offset is present")
    if x_tensor.numel() == 0:
        _validate_empty_input(
            x_tensor, channel_count, scale_tensor, offset_tensor, kwargs
        )
        return [x_tensor.clone()]
    channel_axis = _channel_axis(x_tensor, channel_count, kwargs)
    compute_inputs = [
        x_tensor,
        mean_tensor,
        variance_tensor,
        scale_tensor,
        offset_tensor,
    ]
    if not pre_folded and optional_mask == 0:
        compute_inputs.append(momentum_tensor)
    compute_dtype = _golden_compute_dtype(*compute_inputs)
    x_compute = x_tensor.to(dtype=compute_dtype)
    mean_compute = _parameter_compute(
        mean_tensor,
        channel_count,
        x_tensor.ndim,
        channel_axis,
        "mean",
        x_compute,
        compute_dtype,
    )
    variance_compute = _parameter_compute(
        variance_tensor,
        channel_count,
        x_tensor.ndim,
        channel_axis,
        "variance",
        x_compute,
        compute_dtype,
    )
    gamma_compute = _parameter_compute(
        scale_tensor,
        channel_count,
        x_tensor.ndim,
        channel_axis,
        "scale",
        x_compute,
        compute_dtype,
    )
    beta_compute = _parameter_compute(
        offset_tensor,
        channel_count,
        x_tensor.ndim,
        channel_axis,
        "offset",
        x_compute,
        compute_dtype,
    )
    if pre_folded:
        base = torch.mul(torch.add(x_compute, mean_compute), variance_compute)
        if gamma_compute is not None:
            base = torch.mul(base, gamma_compute)
            if beta_compute is not None:
                base = torch.add(base, beta_compute)
        return [base]

    # The attribute itself is Float, so preserve its FP32 value even when TTK
    # promotes tensor inputs to FP64 for the cross-check reference.
    epsilon_compute = variance_compute.new_tensor(np.float32(epsilon).item())
    if optional_mask == 0:
        factor = _momentum_factor(momentum_tensor, x_compute, compute_dtype)
        alpha = torch.neg(torch.mul(factor, mean_compute))
        scaled_variance = torch.mul(factor, variance_compute)
        sqrt_radicand = torch.sqrt(torch.add(scaled_variance, epsilon_compute))
        folded_beta = torch.div(torch.ones_like(sqrt_radicand), sqrt_radicand)
        return [torch.mul(torch.add(x_compute, alpha), folded_beta)]

    sqrt_radicand = torch.sqrt(torch.add(variance_compute, epsilon_compute))
    if optional_mask == 3:
        inverse_sqrt = torch.div(torch.ones_like(sqrt_radicand), sqrt_radicand)
        folded_beta = torch.mul(gamma_compute, inverse_sqrt)
        offset_over_scale = torch.div(beta_compute, gamma_compute)
        alpha = torch.sub(torch.mul(offset_over_scale, sqrt_radicand), mean_compute)
        return [torch.mul(torch.add(x_compute, alpha), folded_beta)]

    # The two one-sided optional-input combinations are a 950 extension.  Keep
    # the standard BatchNorm expression and inject the missing affine value.
    if gamma_compute is None:
        gamma_compute = torch.ones_like(mean_compute)
    if beta_compute is None:
        beta_compute = torch.zeros_like(mean_compute)
    rstd = torch.div(torch.ones_like(sqrt_radicand), sqrt_radicand)
    centered = torch.sub(x_compute, mean_compute)
    normalized = torch.mul(centered, rstd)
    scaled = torch.mul(normalized, gamma_compute)
    affine = torch.add(scaled, beta_compute)
    return [affine]


def _output_dtype_names(kwargs):
    return [
        _normalize_dtype_name(value) for value in (kwargs.get("output_dtypes") or ())
    ]


def _kernel_golden(
    x,
    mean,
    variance,
    momentum,
    scale=None,
    offset=None,
    epsilon=1e-5,
    use_global_stats=True,
    mode=1,
    **kwargs,
):
    """Kernel/GEIR adapter: NumPy inputs and a one-element NumPy list out."""
    epsilon_value = float(_attribute_value(kwargs, "epsilon", epsilon))
    mode_value = int(_attribute_value(kwargs, "mode", mode))
    del use_global_stats
    outputs = _compute(
        x,
        mean,
        variance,
        momentum,
        scale,
        offset,
        epsilon=epsilon_value,
        mode=mode_value,
        **kwargs,
    )
    dtype_names = _output_dtype_names(kwargs)
    is_promote = str(kwargs.get("golden_mode", "")).lower() == "promote"
    result = []
    for index, output in enumerate(outputs):
        array = _to_numpy(output)
        if (
            not is_promote
            and index < len(dtype_names)
            and dtype_names[index] is not None
        ):
            array = array.astype(_numpy_dtype(dtype_names[index]), copy=False)
        result.append(np.ascontiguousarray(array))
    return result


class _BNInferenceNative:
    """Independent competitor composed from native PyTorch primitives."""

    def __init__(
        self,
        epsilon=1e-5,
        use_global_stats=True,
        mode=1,
        **kwargs,
    ):
        epsilon_value = _attribute_value(kwargs, "epsilon", epsilon)
        self.epsilon = float(
            torch.tensor(float(epsilon_value), dtype=torch.float32).item()
        )
        self.mode = int(_attribute_value(kwargs, "mode", mode))
        self._layout_kwargs = {
            key: kwargs[key]
            for key in (
                "input_formats",
                "input_ori_formats",
                _TTK_SPEC_CONTEXT_KEY,
            )
            if key in kwargs
        }
        del use_global_stats

    def __call__(
        self,
        x,
        mean,
        variance,
        momentum,
        scale=None,
        offset=None,
        **kwargs,
    ):
        layout_kwargs = dict(self._layout_kwargs)
        layout_kwargs.update(kwargs)
        channel_count = mean.numel()
        if variance.numel() != channel_count:
            raise ValueError("mean and variance must have the same C elements")
        optional_mask = int(scale is not None) | (int(offset is not None) << 1)
        if self.mode == 0 and optional_mask == 2:
            raise ValueError("mode=0 requires scale when offset is present")
        if x.numel() == 0:
            _validate_empty_input(x, channel_count, scale, offset, layout_kwargs)
            return [x.clone()]
        # PyTorch BatchNorm consumes a channel-first logical view.  movedim is
        # a view, so a channel-last tensor retains its channels-last storage
        # and can still use the framework's native optimized implementation.
        channel_axis = _channel_axis(x, channel_count, layout_kwargs)
        x_channel_first = x.movedim(channel_axis, 1) if channel_axis == -1 else x
        if x_channel_first.shape[1] != channel_count:
            raise ValueError("mean length does not match the public channel axis")

        parameters = (mean, variance, scale, offset)
        for name, value in zip(("mean", "variance", "scale", "offset"), parameters):
            if value is not None and value.numel() != channel_count:
                raise ValueError(
                    f"{name} must contain C={channel_count} elements, "
                    f"got {value.numel()}"
                )

        def channel_parameter(value):
            shape = [1] * x_channel_first.ndim
            shape[1] = channel_count
            return value.reshape(shape)

        if self.mode == 0:
            pre_folded_inputs = [x, mean, variance, scale, offset]
            compute_dtype = (
                torch.float64
                if any(
                    value is not None and value.dtype == torch.float64
                    for value in pre_folded_inputs
                )
                else torch.float32
            )
            alpha = channel_parameter(
                mean.to(device=x.device, dtype=compute_dtype).reshape(channel_count)
            )
            beta = channel_parameter(
                variance.to(device=x.device, dtype=compute_dtype).reshape(channel_count)
            )
            add_output = torch.add(x_channel_first.to(dtype=compute_dtype), alpha).to(
                dtype=x.dtype
            )
            output_cf = torch.mul(add_output.to(dtype=compute_dtype), beta)
            if scale is not None:
                # The BNInferenceD graph exposes the base Mul in x.dtype before
                # its optional affine stage; this is a tensor boundary, not an
                # NPU implementation detail.
                output_cf = output_cf.to(dtype=x.dtype).to(dtype=compute_dtype)
                native_scale = channel_parameter(
                    scale.to(device=x.device, dtype=compute_dtype).reshape(
                        channel_count
                    )
                )
                output_cf = torch.mul(output_cf, native_scale)
                if offset is not None:
                    native_offset = channel_parameter(
                        offset.to(device=x.device, dtype=compute_dtype).reshape(
                            channel_count
                        )
                    )
                    output_cf = torch.add(output_cf, native_offset)
            output_cf = output_cf.to(dtype=x.dtype)
            output = output_cf.movedim(1, -1) if channel_axis == -1 else output_cf
            return [output]

        if optional_mask in (0, 3):
            if optional_mask == 0 and momentum.numel() == 0:
                raise ValueError("momentum must contain at least one element")

            # BnHost materializes alpha/beta tensors whose dtypes follow
            # mean/variance, and BNInferenceD materializes Add in x's dtype
            # before Mul.  Those casts are graph-visible tensor semantics, not
            # an emulation of an NPU reduction tree or another device detail.
            # Use native Torch pointwise primitives while preserving exactly
            # those typed boundaries.  Actual registered inputs are at most
            # FP32; the FP64 branch keeps direct/promoted callers lossless.
            folded_inputs = [x, mean, variance, scale, offset]
            if optional_mask == 0:
                folded_inputs.append(momentum)
            fold_dtype = (
                torch.float64
                if any(
                    value is not None and value.dtype == torch.float64
                    for value in folded_inputs
                )
                else torch.float32
            )

            mean_fold = mean.to(device=x.device, dtype=fold_dtype).reshape(
                channel_count
            )
            variance_fold = variance.to(device=x.device, dtype=fold_dtype).reshape(
                channel_count
            )
            epsilon_tensor = variance_fold.new_tensor(np.float32(self.epsilon).item())

            if optional_mask == 0:
                momentum_first = momentum.reshape(-1)[0].to(
                    device=x.device, dtype=fold_dtype
                )
                zero = torch.zeros_like(momentum_first)
                factor = torch.where(
                    torch.eq(momentum_first, zero),
                    zero,
                    torch.div(torch.ones_like(momentum_first), momentum_first),
                )
                # The legacy FP16 BnHost path stores factor in FP16 before it
                # produces the hidden coefficients.  BF16 has no corresponding
                # legacy host branch and remains the Ascend950 support superset.
                if momentum.dtype == torch.float16:
                    factor = factor.to(dtype=torch.float16).to(dtype=fold_dtype)
                folded_alpha = torch.neg(torch.mul(factor, mean_fold))
                scaled_variance = torch.mul(factor, variance_fold)
                folded_beta = torch.div(
                    torch.ones_like(scaled_variance),
                    torch.sqrt(torch.add(scaled_variance, epsilon_tensor)),
                )
            else:
                scale_fold = scale.to(device=x.device, dtype=fold_dtype).reshape(
                    channel_count
                )
                offset_fold = offset.to(device=x.device, dtype=fold_dtype).reshape(
                    channel_count
                )
                sqrt_radicand = torch.sqrt(torch.add(variance_fold, epsilon_tensor))
                inverse_sqrt = torch.div(torch.ones_like(sqrt_radicand), sqrt_radicand)
                folded_beta = torch.mul(scale_fold, inverse_sqrt)
                folded_alpha = torch.sub(
                    torch.mul(torch.div(offset_fold, scale_fold), sqrt_radicand),
                    mean_fold,
                )

            # Hidden BnHost outputs follow the statistics dtypes.  Convert them
            # back to the FP32/FP64 arithmetic type only after the semantic
            # tensor round-trip, matching the public two-op chain.
            folded_alpha = folded_alpha.to(dtype=mean.dtype).to(dtype=fold_dtype)
            folded_beta = folded_beta.to(dtype=variance.dtype).to(dtype=fold_dtype)

            add_output = torch.add(
                x_channel_first.to(dtype=fold_dtype),
                channel_parameter(folded_alpha),
            ).to(dtype=x.dtype)
            output_cf = torch.mul(
                add_output.to(dtype=fold_dtype),
                channel_parameter(folded_beta),
            ).to(dtype=x.dtype)
            output = output_cf.movedim(1, -1) if channel_axis == -1 else output_cf
            return [output]

        # Native CUDA BatchNorm supports parameter tensors matching a
        # FP16/BF16 input or FP32 parameters.  Mixed legacy ABI rows are lifted
        # losslessly to FP32; promoted FP32/FP64 inputs are never cast down.
        if x.dtype in (torch.float16, torch.bfloat16):
            parameter_dtypes = {
                value.dtype for value in parameters if value is not None
            }
            native_parameter_dtype = (
                x.dtype if parameter_dtypes == {x.dtype} else torch.float32
            )
        else:
            native_parameter_dtype = x.dtype

        def native_parameter(value):
            if value is None:
                return None
            return value.to(device=x.device, dtype=native_parameter_dtype).reshape(
                channel_count
            )

        native_mean = native_parameter(mean)
        native_variance = native_parameter(variance)
        native_scale = native_parameter(scale)
        native_offset = native_parameter(offset)

        # Scale-only and offset-only are the standard BatchNorm extensions and
        # therefore use the competitor's optimized native BatchNorm API.
        output_cf = torch.nn.functional.batch_norm(
            x_channel_first,
            native_mean,
            native_variance,
            native_scale,
            native_offset,
            training=False,
            momentum=0.0,
            eps=self.epsilon,
        ).to(dtype=x.dtype)
        output = output_cf.movedim(1, -1) if channel_axis == -1 else output_cf
        return [output]


_MOMENTUM_CONTROL_TAGS = {
    "momctl0": 0.0,
    "momctl05": 0.5,
    "momctl1": 1.0,
    "momctl2": 2.0,
}


def _customize_inputs(
    x,
    mean,
    variance,
    momentum,
    scale=None,
    offset=None,
    testcase_name=None,
    **kwargs,
):
    """Make momentum[0] deterministic for tagged mask-0 regression cases."""
    mode = int(_attribute_value(kwargs, "mode", 1))
    name = testcase_name or ""
    name_tokens = set(name.split("_"))
    controlled_value = next(
        (value for tag, value in _MOMENTUM_CONTROL_TAGS.items() if tag in name_tokens),
        None,
    )
    if controlled_value is not None and mode != 0 and scale is None and offset is None:
        flat_momentum = momentum.reshape(-1)
        if flat_momentum.size:
            flat_momentum[0] = controlled_value
            if flat_momentum.size > 1:
                tail_value = 0.5 if controlled_value == 2.0 else 2.0
                flat_momentum[1:] = tail_value
    return x, mean, variance, momentum, scale, offset


class BNInferenceKernelSpec:
    """Shared TestSpec for the BNInference kernel and GEIR pathways."""

    golden = _kernel_golden
    third_party = {"torch": _BNInferenceNative}
    customize_inputs = _customize_inputs
    tolerance = _TOL


def bn_inference_golden(
    x,
    mean,
    variance,
    momentum,
    scale=None,
    offset=None,
    epsilon=1e-5,
    use_global_stats=True,
    mode=1,
    **kwargs,
):
    """Compatibility entry whose parameter order follows ``REG_OP`` exactly."""
    legacy_kwargs = dict(kwargs)
    if not legacy_kwargs.get("output_dtypes"):
        legacy_kwargs["output_dtypes"] = [
            _normalize_dtype_name(_as_torch_tensor(x).dtype)
        ]
    return tuple(
        _kernel_golden(
            x,
            mean,
            variance,
            momentum,
            scale,
            offset,
            epsilon=epsilon,
            use_global_stats=use_global_stats,
            mode=mode,
            **legacy_kwargs,
        )
    )


# 【不适用】ACLNN通路：2026-08-23 对项目私有 CANN 9.2.0 的 libopapi.so
# 执行 strings，aclnnBNInference 命中 0；对照 aclnnBatchNorm 系列有命中。
# 本算子也没有 op_host/op_api 实现或 aclnn 接口文档。
# 【不适用】e2e通路：2026-08-23 对 torch_npu 2.7.1.post5 的
# libtorch_npu.so 执行 strings，aclnnBNInference 命中 0；对照
# aclnnBatchNorm/aclnnBatchNormBackward 均有命中。该二进制时间为
# 2026-07-30，晚于 BNInference 公共 REG_OP 的 2020-10-16 入库时间。
