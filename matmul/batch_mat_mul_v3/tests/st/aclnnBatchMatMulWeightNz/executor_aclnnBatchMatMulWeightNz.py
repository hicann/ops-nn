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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import ctypes


@register("executor_batch_matmul_weight_nz")
class batchMatmulApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(batchMatmulApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆计算
        mat_self = input_data.kwargs["self"]
        mat2 = input_data.kwargs["mat2"]
        output = torch.matmul(mat_self, mat2)
        return output


@register("executor_aclnn_batch_matmul_weight_nz")
class aclnnBatchMatmulApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnBatchMatmulApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        # Store the padded weight as [batch..., N1, K1, K0, N0] while
        # retaining the logical [batch..., K, N] view expected by the API.
        weight = input_data.kwargs["mat2"]
        if weight.ndim != 3 or weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("mat2 must be a 3D float16 or bfloat16 tensor")
        k, n = weight.shape[-2:]
        padded = torch.nn.functional.pad(weight, (0, (-n) % 16, 0, (-k) % 16))
        packed = padded.reshape(
            *weight.shape[:-2], (k + 15) // 16, 16, (n + 15) // 16, 16
        )
        packed = packed.transpose(-3, -2).transpose(-4, -3).contiguous().flatten()
        batch_stride = ((k + 15) // 16) * ((n + 15) // 16) * 256
        input_data.kwargs["mat2"] = torch.as_strided(
            packed, weight.shape, (batch_stride, n, 1)
        )
        input_args, output_packages = super().init_by_input_data(input_data)
        # ATK appends a separate output package after the four JSON inputs.
        output_packages[:] = [input_args[2]]
        input_args.pop()
        # The API expects int8; ATK converts the JSON integer to c_long.
        input_args[3] = ctypes.c_int8(input_data.kwargs["cubeMathType"])
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if index == 1 or name == "mat2":
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
        if name == "mat2" or index == 1:
            shape = input_data.kwargs["mat2"].shape
            return (*shape[:-2], (shape[-1] + 15) // 16, (shape[-2] + 15) // 16, 16, 16)
        return None
