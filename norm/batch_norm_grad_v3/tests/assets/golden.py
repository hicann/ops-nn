# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU PyTorch BatchNorm forward/backward reference for TTK ACLNN cases.

Run with --plugin <operator>/tests/assets/golden.py --compare close.
Training saved statistics come from a matching PyTorch BatchNorm forward.
Disabled outputs must retain the sentinel value 13 supplied by the input hook.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _as_cpu_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    array = np.asarray(value)
    # torch.from_numpy does not accept ml_dtypes.bfloat16; fp32 is lossless here.
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        # Resolve Torch names before NumPy, which needs ml_dtypes for bfloat16.
        torch_dtype = getattr(torch, dtype, None)
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
    return getattr(torch, np.dtype(dtype).name)


def _as_numpy(value):
    if value.dtype == torch.bfloat16:
        import ml_dtypes

        return value.float().numpy().astype(ml_dtypes.bfloat16)
    return value.numpy()


def _channel_first(value, data_format):
    value = _as_cpu_tensor(value).to(torch.float64)
    return (
        value.movedim(-1, 1).contiguous() if data_format in ("NHWC", "NDHWC") else value
    )


@torch.enable_grad()
def golden_batch_norm_grad_v3(
    grad_out,
    input,
    weight,
    running_mean,
    running_var,
    training=True,
    eps=1e-5,
    data_format="NCHW",
):
    """Use the PyTorch BatchNorm forward graph to obtain dx/dgamma/dbeta."""
    x = _channel_first(input, data_format).requires_grad_(True)
    dy = _channel_first(grad_out, data_format)
    gamma = _as_cpu_tensor(weight).to(torch.float64).requires_grad_(True)
    beta = torch.zeros_like(gamma, requires_grad=True)
    mean = None if training else _as_cpu_tensor(running_mean).to(torch.float64)
    var = None if training else _as_cpu_tensor(running_var).to(torch.float64)
    y = F.batch_norm(x, mean, var, gamma, beta, training=training, eps=eps)
    dx, dgamma, dbeta = torch.autograd.grad(y, (x, gamma, beta), grad_outputs=dy)
    if data_format in ("NHWC", "NDHWC"):
        dx = dx.movedim(1, -1)
    return dx, dgamma, dbeta


class AclnnBatchNormBackwardTestSpec:
    """ACLNN arguments follow the header; tensor output indexes are (7, 8, 9)."""

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
        if training:
            data_format = kwargs.get("tensor_formats", ["NCHW"])[0]
            x = _channel_first(input, data_format)
            # Native forward exposes the same saved statistics used by F.batch_norm.
            _, mean, invstd = torch.native_batch_norm(
                x, None, None, None, None, True, 0.0, eps
            )
            for dst, value in ((save_mean, mean), (save_invstd, invstd)):
                if isinstance(dst, torch.Tensor):
                    dst.copy_(value.to(dst.dtype))
                else:
                    dst[...] = value.numpy().astype(dst.dtype)
        for output in (grad_input, grad_weight, grad_bias):
            if isinstance(output, torch.Tensor):
                output.fill_(13)
            else:
                output[...] = 13

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
        initial_outputs = (grad_input, grad_weight, grad_bias)
        dtypes = kwargs.get("tensor_dtypes")
        output_dtypes = dtypes[7:10] if dtypes else [x.dtype for x in initial_outputs]
        data_format = kwargs.get("tensor_formats", ["NCHW"])[0]
        outputs = golden_batch_norm_grad_v3(
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            training,
            eps,
            data_format,
        )
        outputs = tuple(
            (value if enabled else _as_cpu_tensor(initial).clone()).to(
                _torch_dtype(dtype)
            )
            for value, enabled, initial, dtype in zip(
                outputs, output_mask, initial_outputs, output_dtypes
            )
        )
        if kwargs.get("use_numpy", False):
            return tuple(_as_numpy(value) for value in outputs)
        return outputs


__spec__ = {"aclnnBatchNormBackward": "AclnnBatchNormBackwardTestSpec"}
