# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Self-contained BatchNormGradV3 TestSpecs (950/960 semantics).

Registrations follow the multi-path pattern used by UndGenQkvRmsNormRopeCache:
Kernel/GEIR, ACLNN, and Torch/E2E have separate adapters and a shared reference.
All three expose a device-preserving ``third_party['torch']`` native ATen reference.
No TensorFlow/fast-ACLNN mapping is claimed.

CPU truth calls aten.native_batch_norm_backward in FP64, using the supplied saved
mean/inverse standard deviation. It never recomputes or overwrites these inputs.
Empty tensors are handled before native ATen (some CPU versions crash on N=0).
Input hooks leave optional None values, invalid shapes and broadcast views intact.
Only writable ACLNN output buffers are filled with the disabled-output sentinel.

Kernel inference computes dx only: its two unspecified auxiliary outputs, and
masked Kernel outputs, return None. TTK reports those slots as SUPPRESSED; they
are not numerical comparisons. ACLNN inference computes all requested outputs.
Torch/E2E targets the 950/960 torch_npu binding, including ACLNN's 0/1
default-fill policy. Disabled Torch outputs remain None. Raw CPU ATen does not
accept every optional combination supported by that binding.

Kernel Enable honors output_dtypes; Promote retains FP64. ACLNN and Torch truth
outputs retain FP64 (ACLNN does not consistently forward golden_mode), avoiding
loss of promoted precision or accidental half rounding of FP32 parameter grads.
Comparisons must use the DUT's declared/actual output dtype. Third-party outputs
use normal interface dtypes and stay on their input CPU/CUDA device. NPU native
backward is deliberately not used as an oracle for itself.
"""

import numpy as np
import torch

__spec__ = {
    "batch_norm_grad_v3": "BatchNormGradV3TestSpec",
    "BatchNormGradV3": "BatchNormGradV3TestSpec",
    "aclnnBatchNormBackward": "AclnnBatchNormBackwardTestSpec",
    "torch.ops.aten.native_batch_norm_backward": "TorchBatchNormBackwardTestSpec",
    "torch.ops.aten.native_batch_norm_backward.default": "TorchBatchNormBackwardTestSpec",
}

_SENTINEL = 13
_UNSET = object()


def _torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).removeprefix("torch.")
    if name == "float":
        name = "float32"
    result = getattr(torch, name, None)
    if isinstance(result, torch.dtype):
        return result
    return getattr(torch, np.dtype(dtype).name)


def _as_tensor(value, device="cpu"):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(device) if device is not None else value.detach()
    array = np.asarray(value)
    if array.dtype.name == "bfloat16":
        tensor = torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    else:
        # Read-only/negative-stride arrays are valid reference inputs; never write through them.
        tensor = torch.from_numpy(np.array(array, copy=True, order="C"))
    return tensor.to(device) if device is not None else tensor


def _as_numpy(value):
    if value is None:
        return None
    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        import ml_dtypes

        return value.float().numpy().astype(ml_dtypes.bfloat16)
    return value.numpy()


def _mask(value, default=True):
    if value is None:
        if not default:
            raise ValueError("outputMask must be a non-null list of length 3")
        return (True, True, True)
    result = tuple(bool(x) for x in value)
    if len(result) != 3:
        raise ValueError("output mask must have length 3")
    return result


def _format(kwargs, route):
    for key in (
        ("input_formats", "input_ori_formats")
        if route == "kernel"
        else ("tensor_formats", "input_formats", "input_ori_formats")
    ):
        values = kwargs.get(key)
        if values:
            return str(values[0]).removeprefix("FORMAT_")
    return str(kwargs.get("data_format", "NCHW"))


def _attr(kwargs, name, default):
    return kwargs.get(name, (kwargs.get("attributes") or {}).get(name, default))


def _vector(value, channels, like, name, default=None):
    if value is None:
        if default is None:
            return None
        return torch.full((channels,), default, dtype=like.dtype, device=like.device)
    value = _as_tensor(value, like.device)
    if value.ndim != 1 or value.numel() != channels:
        raise ValueError(
            f"{name} must have shape ({channels},), got {tuple(value.shape)}"
        )
    return value.to(like.dtype)


def _native_reference(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_invstd,
    training,
    eps,
    output_mask,
    data_format="NCHW",
    *,
    device="cpu",
    compute_dtype=torch.float64,
):
    """Return high-precision values; adapters own output dtype/mask/buffer conventions."""
    x, dy = _as_tensor(input, device), _as_tensor(grad_out, device)
    if x is None or dy is None:
        raise ValueError("input and grad_out are required")
    if x.ndim < 2 or tuple(x.shape) != tuple(dy.shape):
        raise ValueError("input/grad_out must have equal shapes and rank >= 2")
    if not x.dtype.is_floating_point or not dy.dtype.is_floating_point:
        raise ValueError("batch norm reference requires floating input tensors")
    if x.device.type not in ("cpu", "cuda"):
        raise NotImplementedError(
            "Use CPU truth or a CUDA third_party; NPU backward would call the DUT itself"
        )
    mask = _mask(output_mask)
    channel_last = data_format in ("NHWC", "NDHWC")
    if channel_last:
        x, dy = x.movedim(-1, 1), dy.movedim(-1, 1)
    channels = x.shape[1]
    # FP64 supplied by Promote must never be narrowed in the third-party normalization either.
    values = (x, dy, weight, running_mean, running_var, save_mean, save_invstd)
    if any(
        v is not None and _as_tensor(v, None).dtype == torch.float64 for v in values
    ):
        compute_dtype = torch.float64
    x, dy = x.to(compute_dtype).contiguous(), dy.to(compute_dtype).contiguous()
    if x.numel() == 0:
        result = (torch.empty_like(x), x.new_zeros(channels), x.new_zeros(channels))
    else:
        gamma = _vector(weight, channels, x, "weight", 1)
        if training:
            mean = _vector(save_mean, channels, x, "save_mean", 0)
            invstd = _vector(save_invstd, channels, x, "save_invstd", 1)
            rm, rv = None, None
        else:
            rm = _vector(running_mean, channels, x, "running_mean", 0)
            rv = _vector(running_var, channels, x, "running_var", 1)
            # CUDA's non-fused backward requires defined saved-stat tensors even
            # in inference. Empty tensors make ATen use running statistics.
            mean, invstd = (
                (x.new_empty(0), x.new_empty(0))
                if x.device.type == "cuda"
                else (None, None)
            )
        result = torch.ops.aten.native_batch_norm_backward(
            dy, x, gamma, rm, rv, mean, invstd, bool(training), float(eps), list(mask)
        )
    result = tuple(value if enabled else None for value, enabled in zip(result, mask))
    if channel_last and result[0] is not None:
        result = (result[0].movedim(1, -1), result[1], result[2])
    return result


def _cast_outputs(values, dtypes, *, numpy=False):
    result = tuple(
        None if v is None else v.to(_torch_dtype(dt)) for v, dt in zip(values, dtypes)
    )
    return tuple(_as_numpy(v) for v in result) if numpy else result


def _clone(value, device="cpu"):
    value = _as_tensor(value, device)
    return None if value is None else value.clone()


def _fill_output(value):
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        # Scalar fill is valid even for expanded/overlapping output views.
        with torch.no_grad():
            value.fill_(_SENTINEL)
    elif isinstance(value, np.ndarray) and value.flags.writeable:
        value.fill(_SENTINEL)


def _aclnn_reference(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_invstd,
    training,
    eps,
    output_mask,
    grad_input,
    grad_weight,
    grad_bias,
    *,
    device="cpu",
    compute_dtype=torch.float64,
    data_format="NCHW",
):
    initial = (grad_input, grad_weight, grad_bias)
    x, dy = _as_tensor(input, device), _as_tensor(grad_out, device)
    if x is None or dy is None:
        raise ValueError("input and gradOut must not be null")
    # Match the public API's empty early return, including disabled-output preservation.
    if x.numel() == 0 or dy.numel() == 0:
        result = [_clone(v, device) for v in initial]
        if output_mask is not None:
            bits = tuple(bool(x) for x in output_mask)
            if bits and bits[0] and result[0] is not None and result[0].numel() == 0:
                result[0] = result[0].to(compute_dtype)
            for i in (1, 2):
                if i < len(bits) and bits[i] and result[i] is not None:
                    result[i] = torch.zeros_like(result[i], dtype=compute_dtype)
        return tuple(result)
    mask = _mask(output_mask, default=False)
    values = _native_reference(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        training,
        eps,
        mask,
        data_format=data_format,
        device=device,
        compute_dtype=compute_dtype,
    )
    return tuple(
        value if enabled else _clone(initial[i], device)
        for i, (value, enabled) in enumerate(zip(values, mask))
    )


def golden_batch_norm_grad_v3(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    training=True,
    eps=1e-5,
    data_format="NCHW",
    *,
    save_mean=_UNSET,
    save_invstd=_UNSET,
):
    """Compatibility helper. Omitted saved stats are derived, explicitly supplied stats are consumed.

    Registered TestSpecs always supply saved-stat arguments, including explicit None.
    This backward-compatible helper is the only opt-in forward-stat derivation path.
    """
    if training and (save_mean is _UNSET or save_invstd is _UNSET):
        x = _as_tensor(input).double()
        if data_format in ("NHWC", "NDHWC"):
            x = x.movedim(-1, 1)
        axes = (0,) + tuple(range(2, x.ndim))
        if x.numel() == 0:
            mean, invstd = x.new_zeros(x.shape[1]), x.new_ones(x.shape[1])
        else:
            variance, mean = torch.var_mean(x, dim=axes, correction=0)
            invstd = torch.rsqrt(variance + eps)
        if save_mean is _UNSET:
            save_mean = mean
        if save_invstd is _UNSET:
            save_invstd = invstd
    return _native_reference(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        None if save_mean is _UNSET else save_mean,
        None if save_invstd is _UNSET else save_invstd,
        training,
        eps,
        (True, True, True),
        data_format,
    )


def _check_truth_device(kwargs):
    if torch.device(kwargs.get("device", "cpu")).type != "cpu":
        raise NotImplementedError(
            "CPU truth must run on CPU; use third_party for the independent CUDA reference"
        )


def _kernel_values(
    dy,
    x,
    weight,
    running_mean,
    running_var,
    save_mean,
    save_rstd,
    kwargs,
    *,
    device="cpu",
    compute_dtype=torch.float64,
):
    required = (dy, x, weight, running_mean, running_var, save_mean, save_rstd)
    if any(value is None for value in required):
        raise ValueError("Kernel/GEIR BatchNormGradV3 requires all seven input tensors")
    training = bool(_attr(kwargs, "is_training", True))
    mask = _mask(_attr(kwargs, "output_mask", None))
    if not training:
        mask = (mask[0], False, False)
    return _native_reference(
        dy,
        x,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_rstd,
        training,
        float(_attr(kwargs, "epsilon", 1e-5)),
        mask,
        _format(kwargs, "kernel"),
        device=device,
        compute_dtype=compute_dtype,
    )


class BatchNormGradV3KernelXpuRef:
    def __call__(
        self, dy, x, weight, running_mean, running_var, save_mean, save_rstd, **kwargs
    ):
        values = _kernel_values(
            dy,
            x,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_rstd,
            kwargs,
            device=None,
            compute_dtype=torch.float32,
        )
        dtypes = kwargs.get("output_dtypes") or (dy.dtype, weight.dtype, weight.dtype)
        return _cast_outputs(values, dtypes)


class AclnnBatchNormBackwardXpuRef:
    def __call__(
        self,
        gradOut,
        input,
        weight,
        runningMean,
        runningVar,
        saveMean,
        saveInvstd,
        training,
        eps,
        outputMask,
        gradInput=None,
        gradWeight=None,
        gradBias=None,
        **kwargs,
    ):
        # XPU transport may omit pure output buffers. Rebuild this Spec's sentinel fixture,
        # so empty-input zeros and disabled-output preservation retain their ACLNN meaning.
        x = _as_tensor(input, None)
        wd = weight.dtype if weight is not None else torch.float32
        channels = (
            x.shape[-1] if _format(kwargs, "aclnn") in ("NHWC", "NDHWC") else x.shape[1]
        )
        shapes = (x.shape, (channels,), (channels,))
        buffers = [gradInput, gradWeight, gradBias]
        fallback = (gradOut.dtype, wd, wd)
        for i, buffer in enumerate(buffers):
            if buffer is None:
                buffers[i] = torch.full(
                    shapes[i], _SENTINEL, dtype=fallback[i], device=x.device
                )
        gradInput, gradWeight, gradBias = buffers
        values = _aclnn_reference(
            gradOut,
            input,
            weight,
            runningMean,
            runningVar,
            saveMean,
            saveInvstd,
            training,
            eps,
            outputMask,
            gradInput,
            gradWeight,
            gradBias,
            data_format=_format(kwargs, "aclnn"),
            device=None,
            compute_dtype=torch.float32,
        )
        wd = weight.dtype if weight is not None else torch.float32
        buffers = (gradInput, gradWeight, gradBias)
        fallback = (gradOut.dtype, wd, wd)
        declared = (kwargs.get("tensor_dtypes") or ())[7:10]
        dtypes = tuple(
            declared[i]
            if len(declared) == 3 and declared[i] is not None
            else (buf.dtype if buf is not None else fallback[i])
            for i, buf in enumerate(buffers)
        )
        return _cast_outputs(values, dtypes)


class TorchBatchNormBackwardXpuRef:
    def __call__(
        self,
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
        **kwargs,
    ):
        values = _native_reference(
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            train,
            eps,
            output_mask,
            device=None,
            compute_dtype=torch.float32,
        )
        param_dtype = (
            torch.float32
            if input.dtype in (torch.float16, torch.bfloat16)
            else input.dtype
        )
        return _cast_outputs(values, (input.dtype, param_dtype, param_dtype))


class BatchNormGradV3TestSpec:
    """Kernel and GEIR share their actual seven-input operator signature."""

    third_party = {"torch": BatchNormGradV3KernelXpuRef}

    def customize_inputs(
        dy, x, weight, running_mean, running_var, save_mean, save_rstd, **kwargs
    ):
        # Keep malformed/optional inputs and supplied statistics intact for API validation and the oracle.
        return (dy, x, weight, running_mean, running_var, save_mean, save_rstd)

    def golden(
        dy, x, weight, running_mean, running_var, save_mean, save_rstd, **kwargs
    ):
        _check_truth_device(kwargs)
        values = _kernel_values(
            dy, x, weight, running_mean, running_var, save_mean, save_rstd, kwargs
        )
        if str(kwargs.get("golden_mode", "Enable")).lower() == "promote":
            return tuple(_as_numpy(v) for v in values)
        dtypes = kwargs.get("output_dtypes") or (dy.dtype, weight.dtype, weight.dtype)
        return _cast_outputs(values, dtypes, numpy=True)


class AclnnBatchNormBackwardTestSpec:
    third_party = {"torch": AclnnBatchNormBackwardXpuRef}

    def customize_inputs(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        training,
        eps,
        output_mask,
        grad_input,
        grad_weight,
        grad_bias,
        **kwargs,
    ):
        for output in (grad_input, grad_weight, grad_bias):
            _fill_output(output)

    def golden(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        training,
        eps,
        output_mask,
        grad_input,
        grad_weight,
        grad_bias,
        **kwargs,
    ):
        _check_truth_device(kwargs)
        values = _aclnn_reference(
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            training,
            eps,
            output_mask,
            grad_input,
            grad_weight,
            grad_bias,
            data_format=_format(kwargs, "aclnn"),
        )
        return (
            tuple(_as_numpy(v) for v in values)
            if kwargs.get("use_numpy", False)
            else values
        )


class TorchBatchNormBackwardTestSpec:
    third_party = {"torch": TorchBatchNormBackwardXpuRef}

    def customize_inputs(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
        **kwargs,
    ):
        return None

    def golden(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
        **kwargs,
    ):
        _check_truth_device(kwargs)
        values = _native_reference(
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            train,
            eps,
            output_mask,
        )
        return (
            tuple(_as_numpy(v) for v in values)
            if kwargs.get("use_numpy", False)
            else values
        )
