#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat, TensorPtr
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def _construct_grad_y(input_data):
    input_x = input_data.kwargs["x"]
    dim = input_data.kwargs["dim"]
    normalized_dim = dim if dim >= 0 else dim + input_x.dim()
    grad_y_shape = list(input_x.shape)
    grad_y_shape[normalized_dim] = grad_y_shape[normalized_dim] // 2
    gen = torch.Generator(device="cpu").manual_seed(42)
    grad_y = (
        torch.rand(grad_y_shape, dtype=torch.float32, device="cpu", generator=gen) * 2
        - 1
    )
    grad_y = grad_y.to(input_x.dtype)
    input_data.kwargs["grad_y"] = grad_y.to(input_x.device)


def _construct_group_index(input_data):
    """为 range_values='active' 的 case 构造 group_index 数据。
    group_index 值 = batch_size，保证 kernel 和 golden 都处理全部行（与 isGroup_=0 行为一致）。"""
    input_x = input_data.kwargs["x"]
    dim = input_data.kwargs["dim"]
    normalized_dim = dim if dim >= 0 else dim + input_x.dim()
    shape = list(input_x.shape)
    if len(shape) > 1 and normalized_dim != 0:
        batch_size = 1
        for i in range(normalized_dim):
            batch_size *= shape[i]
    else:
        batch_size = 1
    group_index = torch.tensor([batch_size], dtype=torch.int64, device=input_x.device)
    input_data.kwargs["group_index"] = group_index


def _construct_group_index_partial(input_data):
    """为 range_values='partial' 的 case 构造 group_index 数据。
    group_index 值 = batch_size // 2，使 realBatchSize < dimBatchSize，触发 ZeroInvalidRows 分支。"""
    input_x = input_data.kwargs["x"]
    dim = input_data.kwargs["dim"]
    normalized_dim = dim if dim >= 0 else dim + input_x.dim()
    shape = list(input_x.shape)
    if len(shape) > 1 and normalized_dim != 0:
        batch_size = 1
        for i in range(normalized_dim):
            batch_size *= shape[i]
    else:
        batch_size = 1
    if batch_size > 1:
        group_sum = batch_size // 2
    else:
        group_sum = batch_size
    group_index = torch.tensor([group_sum], dtype=torch.int64, device=input_x.device)
    input_data.kwargs["group_index"] = group_index


def _get_group_index_mode(executor):
    """从 self.task_result.case_config.inputs 读取 group_index 的 range_values。"""
    case_config = getattr(getattr(executor, "task_result", None), "case_config", None)
    if case_config is None:
        return None
    inputs = getattr(case_config, "inputs", None) or []
    for inp in inputs:
        items = inp if isinstance(inp, list) else [inp]
        for item in items:
            if getattr(item, "name", None) == "group_index":
                return getattr(item, "range_values", None)
    return None


@register("aclnn_clipped_swiglu_grad")
class FunctionAclnnClippedSwigluGrad(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

        gi_mode = _get_group_index_mode(self)
        if gi_mode == "active":
            _construct_group_index(input_data)
        elif gi_mode == "partial":
            _construct_group_index_partial(input_data)
        elif gi_mode == "default":
            input_data.kwargs["group_index"] = None

        input_args, output_packages = super().init_by_input_data(input_data)

        # group_index 可选输入：c_void_p 时转为 null tensor 指针
        if type(input_args[2]).__name__ == "c_void_p":
            AclTensorPtr = ctypes.POINTER(AclTensor)
            null_void_ptr = ctypes.c_void_p(None)
            null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)
            input_args[2] = null_tensor_ptr

        return input_args, output_packages


@register("function_clipped_swiglu_grad")
class FunctionClippedSwigluGrad(BaseApi):
    def clipped_swiglu_grad_golden(
        self,
        input_x,
        input_dy,
        input_group_index,
        dim=-1,
        alpha=1.702,
        limit=7.0,
        bias=1.0,
        interleaved=True,
    ):
        dim = dim if dim >= 0 else dim + input_x.dim()
        dtype = input_x.dtype
        if input_x.dtype in [torch.bfloat16, torch.float16]:
            input_x = input_x.to(torch.float32)
        if input_dy.dtype in [torch.bfloat16, torch.float16]:
            input_dy = input_dy.to(torch.float32)
        shape = input_x.shape
        # 合轴：x -> [pre, dim2H], dy -> [pre, dim2H // 2]
        if input_x.ndim > 1:
            if dim != 0:
                dim1 = torch.prod(torch.tensor(shape[:dim])).item()
                dim2 = torch.prod(torch.tensor(shape[dim:])).item()
                x = input_x.reshape(dim1, dim2).clone()
                dy = input_dy.reshape(dim1, dim2 // 2).clone()
            else:
                dim2 = torch.prod(torch.tensor(shape[dim:])).item()
                x = input_x.reshape(1, dim2).clone()
                dy = input_dy.reshape(1, dim2 // 2).clone()
        else:
            x = input_x.reshape(1, shape[0]).clone()
            dy = input_dy.reshape(1, shape[0] // 2).clone()
        group = x.shape[0]
        if input_group_index is not None:
            group = min(torch.sum(input_group_index).item(), x.shape[0])
        x_tensor = x[:group]
        dy_tensor = dy[:group]
        h = x.shape[1] // 2
        # 切分（与正向一致）
        if interleaved:
            a = x_tensor[..., ::2]
            b = x_tensor[..., 1::2]
        else:
            out = torch.chunk(x_tensor, 2, dim=-1)
            a = out[0]
            b = out[1]
        # 重算正向中间量
        A = a.clamp(min=None, max=limit)
        B = b.clamp(min=-limit, max=limit)
        s = torch.sigmoid(alpha * A)
        # clamp 的 mask（基于原始值 a, b）
        maskA = (a <= limit).to(torch.float32)
        maskB = ((b >= -limit) & (b <= limit)).to(torch.float32)
        # 反向梯度公式
        # da = dy * (B + bias) * s * (1 + alpha * A * (1 - s)) * maskA
        # db = dy * A * s * maskB
        da = dy_tensor * (B + bias) * s * (1.0 + alpha * A * (1.0 - s)) * maskA
        db = dy_tensor * A * s * maskB
        # 散回 dx
        dx = torch.zeros_like(x)
        dx_tensor = dx[:group]
        if interleaved:
            dx_tensor[..., ::2] = da
            dx_tensor[..., 1::2] = db
        else:
            dx_tensor[..., :h] = da
            dx_tensor[..., h:] = db
        res_shape = list(shape)

        if dx.numel() != 0:
            dx = dx.reshape(res_shape)
        if dtype == torch.bfloat16:
            dx = dx.to(torch.bfloat16)
        elif dtype == torch.float16:
            dx = dx.to(torch.float16)
        return dx

    def init_by_input_data(self, input_data):
        gi_mode = _get_group_index_mode(self)
        if gi_mode == "active":
            _construct_group_index(input_data)
        elif gi_mode == "partial":
            _construct_group_index_partial(input_data)
        elif gi_mode == "default":
            input_data.kwargs["group_index"] = None
        return (
            super().init_by_input_data(input_data)
            if hasattr(super(), "init_by_input_data")
            else (None, None)
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs["x"]
        input_dy = input_data.kwargs["grad_y"]
        input_group_index = input_data.kwargs.get("group_index", None)
        dim = input_data.kwargs["dim"]
        alpha = input_data.kwargs["alpha"]
        limit = input_data.kwargs["limit"]
        bias = input_data.kwargs["bias"]
        interleaved = input_data.kwargs["interleaved"]
        dx = self.clipped_swiglu_grad_golden(
            input_x, input_dy, input_group_index, dim, alpha, limit, bias, interleaved
        )
        return dx


@register("aclnn_clipped_swiglu_grad_abnormal")
class FunctionAclnnClippedSwigluGradabnormal(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

        gi_mode = _get_group_index_mode(self)
        if gi_mode == "active":
            _construct_group_index(input_data)
        elif gi_mode == "partial":
            _construct_group_index_partial(input_data)
        elif gi_mode == "default":
            input_data.kwargs["group_index"] = None
        input_args, output_packages = super().init_by_input_data(input_data)

        # group_index 可选输入：c_void_p 时转为 null tensor 指针
        if type(input_args[2]).__name__ == "c_void_p":
            AclTensorPtr = ctypes.POINTER(AclTensor)
            null_void_ptr = ctypes.c_void_p(None)
            null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)
            input_args[2] = null_tensor_ptr

        id = self.task_result.case_config.id
        if id == 0:
            # 必选参数gradY/x/gradXOut为nullptr
            input_args[0] = TensorPtr()
        elif id == 1:
            # 必选参数gradY/x/gradXOut为nullptr
            input_args[1] = TensorPtr()
        elif id == 2:
            # 必选参数gradY/x/gradXOut为nullptr
            input_args[8] = TensorPtr()
        elif id == 3:
            # gradY的数据类型不在支持的范围之内
            new_tensor2 = torch.randn(input_args[0].pytensor.shape, dtype=torch.double)
            acl_tensor2 = self.backend.torch_tensor_to_acl(
                new_tensor2, AclFormat.ACL_FORMAT_ND
            )
            input_args[0] = acl_tensor2
        elif id == 4:
            # gradY的数据类型不在支持的范围之内
            new_tensor3 = torch.randn(input_args[8].pytensor.shape, dtype=torch.double)
            acl_tensor3 = self.backend.torch_tensor_to_acl(
                new_tensor3, AclFormat.ACL_FORMAT_ND
            )
            input_args[8] = acl_tensor3
        elif id == 5:
            # x在dim对应维度的大小必须为偶数
            new_tensor4 = torch.randn(
                1, 1, 2, 1, 3, 9, dtype=input_args[1].pytensor.dtype
            )
            acl_tensor4 = self.backend.torch_tensor_to_acl(
                new_tensor4, AclFormat.ACL_FORMAT_ND
            )
            input_args[1] = acl_tensor4
        elif id == 7:
            # dim取值范围为[-x.dim(), x.dim()-1]
            int64_arg_scale = ctypes.c_long(10)
            input_args[3] = int64_arg_scale
        elif id == 8:
            # gradY在dim对应维度的大小为x的一半，其他维度与x一致
            new_tensor5 = torch.randn(
                1, 1, 2, 1, 3, 9, dtype=input_args[0].pytensor.dtype
            )
            acl_tensor5 = self.backend.torch_tensor_to_acl(
                new_tensor5, AclFormat.ACL_FORMAT_ND
            )
            input_args[0] = acl_tensor5
        elif id == 9:
            # gradY在dim对应维度的大小为x的一半，其他维度与x一致
            new_tensor6 = torch.randn(
                1, 1, 2, 1, 3, 9, dtype=input_args[0].pytensor.dtype
            )
            acl_tensor6 = self.backend.torch_tensor_to_acl(
                new_tensor6, AclFormat.ACL_FORMAT_ND
            )
            input_args[8] = acl_tensor6
        return input_args, output_packages
