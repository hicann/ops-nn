# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import torch
import numpy as np
import ctypes
import random
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


def generate_group_index(G, max_tokens, device):
    """Generate group_index with sum <= max_tokens"""
    if G > max_tokens:
        G = max_tokens
    np.random.seed(G + max_tokens)
    tokens_list = []
    remaining = max_tokens
    for i in range(G - 1):
        max_val = max(1, remaining // (G - i))
        val = np.random.randint(1, max_val + 1)
        tokens_list.append(val)
        remaining -= val
    tokens_list.append(max(1, remaining))
    total = sum(tokens_list)
    if total > max_tokens:
        diff = total - max_tokens
        tokens_list[-1] -= diff
    group_index = torch.tensor(tokens_list, dtype=torch.int64).to(device)
    return group_index


def swiglu_group_quant_backward(
    grad_y, x, weight=None, y_origin=None, group_index=None, clamp_limit=0.0
):
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if grad_y.dtype not in supported_dtypes:
        raise ValueError(
            f"grad_y dtype must be float16/bfloat16/float32, got {grad_y.dtype}"
        )
    if x.dtype not in supported_dtypes:
        raise ValueError(f"x dtype must be float16/bfloat16/float32, got {x.dtype}")
    if weight is not None and weight.dtype != torch.float32:
        raise ValueError(f"weight dtype must be float32, got {weight.dtype}")
    if y_origin is not None and y_origin.dtype not in supported_dtypes:
        raise ValueError(
            f"y_origin dtype must be float16/bfloat16/float32, got {y_origin.dtype}"
        )

    input_dtype = x.dtype
    grad_y = grad_y.float()
    x = x.float()
    if y_origin is not None:
        y_origin = y_origin.float()

    grad_weight = None

    if group_index is not None:
        trunc = int(group_index.sum().item())

    if weight is not None:
        grad_weight = torch.sum(grad_y * y_origin, dim=-1, keepdim=True)
        if group_index is not None:
            original_gw_shape = grad_weight.shape
            grad_weight = grad_weight.reshape([-1, 1])
            num_rows = grad_weight.shape[0]
            row = torch.arange(num_rows, device=grad_weight.device)
            mask = (row < trunc).unsqueeze(-1).float()
            grad_weight = grad_weight * mask
            grad_weight = grad_weight.reshape(original_gw_shape)
        grad_y0 = grad_y * weight.float()
    else:
        grad_y0 = grad_y

    original_shape = x.shape
    x = x.reshape([-1, x.shape[-1]])
    H = x.shape[-1] // 2
    x0 = x[:, :H]
    x1 = x[:, H:]

    x0_truncated = x0
    x1_truncated = x1

    if clamp_limit != -1.0:
        x0 = torch.clamp(x0, max=clamp_limit)
        x1 = torch.clamp(x1, -clamp_limit, clamp_limit)

    sigmoid_x0 = torch.sigmoid(x0)
    silu_x0 = x0 * sigmoid_x0
    silu_grad_x0 = sigmoid_x0 * (1 + x0 * (1 - sigmoid_x0))

    grad_y0_flat = grad_y0.reshape([-1, H])
    grad_x0 = grad_y0_flat * x1 * silu_grad_x0
    grad_x1 = grad_y0_flat * silu_x0

    if clamp_limit != -1.0:
        mask_x0 = (x0_truncated < clamp_limit).float()
        mask_x1 = ((-clamp_limit < x1_truncated) & (x1_truncated < clamp_limit)).float()
        grad_x0 = grad_x0 * mask_x0
        grad_x1 = grad_x1 * mask_x1

    if group_index is not None:
        num_rows = grad_x0.shape[0]
        row = torch.arange(num_rows, device=grad_x0.device)
        mask = (row < trunc).unsqueeze(-1).float()
        grad_x0 = grad_x0 * mask
        grad_x1 = grad_x1 * mask

    grad_x = torch.cat([grad_x0, grad_x1], dim=-1)
    grad_x = grad_x.reshape(original_shape)
    grad_x = grad_x.to(input_dtype)

    return grad_x, grad_weight


@register("function_aclnnSwigluGroupQuantGrad")
class AclnnSwigluGroupQuantGradGoldenExecutor(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        group_index = input_data.kwargs["group_index"]
        if group_index is not None:
            if len(input_data.kwargs["grad_y"].shape) == 3:
                max_tokens = (
                    input_data.kwargs["grad_y"].shape[0]
                    * input_data.kwargs["grad_y"].shape[1]
                )
            else:
                max_tokens = input_data.kwargs["grad_y"].shape[0]
            device = input_data.kwargs["x"].device
            random.seed(max_tokens)
            G = random.choice([192, 256, 384])
            if G > max_tokens:
                for g in [384, 256, 192]:
                    if g <= max_tokens:
                        G = g
                        break
            group_index = generate_group_index(G, max_tokens, device)
            input_data.kwargs["group_index"] = group_index

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad_y = input_data.kwargs["grad_y"]
        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        y_origin = input_data.kwargs["y_origin"]
        group_index = input_data.kwargs["group_index"]
        clamp_limit = input_data.kwargs["clamp_limit"]

        grad_x, grad_weight = swiglu_group_quant_backward(
            grad_y,
            x,
            weight=weight,
            y_origin=y_origin,
            group_index=group_index,
            clamp_limit=clamp_limit,
        )

        return grad_x, grad_weight


@register("aclnnfunction_aclnnSwigluGroupQuantGrad")
class AclnnSwigluGroupQuantGradExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        group_index = input_data.kwargs["group_index"]
        if group_index is not None:
            if len(input_data.kwargs["grad_y"].shape) == 3:
                max_tokens = (
                    input_data.kwargs["grad_y"].shape[0]
                    * input_data.kwargs["grad_y"].shape[1]
                )
            else:
                max_tokens = input_data.kwargs["grad_y"].shape[0]
            device = input_data.kwargs["x"].device
            random.seed(max_tokens)
            G = random.choice([192, 256, 384])
            if G > max_tokens:
                for g in [384, 256, 192]:
                    if g <= max_tokens:
                        G = g
                        break

            group_index = generate_group_index(G, max_tokens, device)
            input_data.kwargs["group_index"] = group_index

        input_args, output_packages = super().init_by_input_data(input_data)

        weight = input_data.kwargs["weight"]
        y_origin = input_data.kwargs["y_origin"]
        group_index_kw = input_data.kwargs["group_index"]

        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)

        if weight is None:
            input_args[2] = null_tensor_ptr
        if y_origin is None:
            input_args[3] = null_tensor_ptr
        if group_index_kw is None:
            input_args[4] = null_tensor_ptr

        return input_args, output_packages
