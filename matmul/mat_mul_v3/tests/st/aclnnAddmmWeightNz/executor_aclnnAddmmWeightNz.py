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


@register("execute_addmm_weight_nz")
class addmmApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(addmmApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆计算
        mat_input = input_data.kwargs["self"]
        mat1 = input_data.kwargs["mat1"]
        mat2 = input_data.kwargs["mat2"]
        beta = input_data.kwargs["beta"]
        alpha = input_data.kwargs["alpha"]
        out = input_data.kwargs["out"]
        if (
            mat_input.ndim == 2
            and (alpha != 1 or mat_input.shape[0] != 1)
            and beta != 0
        ):
            # Non-fused 2D bias with cubeMathType=0 rounds intermediates to out.dtype.
            output_dtype = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }[
                next(
                    arg.dtype
                    for arg in self.task_result.case_config.inputs
                    if arg.name == "out"
                )
            ]
            product = torch.mm(mat1.float(), mat2.float()).to(output_dtype)
            return product * alpha + mat_input.to(output_dtype) * beta
        return torch.addmm(mat_input, mat1, mat2, beta=beta, alpha=alpha, out=out)


@register("execute_aclnn_addmm_weight_nz")
class aclnnAddmmApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnAddmmApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        # Store the padded weight as [batch..., N1, K1, K0, N0] while
        # retaining the logical [batch..., K, N] view expected by the API.
        weight = input_data.kwargs["mat2"]
        if weight.ndim != 2 or weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("mat2 must be a 2D float16 or bfloat16 tensor")
        k, n = weight.shape[-2:]
        padded = torch.nn.functional.pad(weight, (0, (-n) % 16, 0, (-k) % 16))
        packed = padded.reshape(
            *weight.shape[:-2], (k + 15) // 16, 16, (n + 15) // 16, 16
        )
        packed = packed.transpose(-3, -2).transpose(-4, -3).contiguous().flatten()
        input_data.kwargs["mat2"] = torch.as_strided(
            packed, weight.shape, weight.stride()
        )
        input_args, output_packages = super().init_by_input_data(input_data)
        # ATK appends a separate output package after the seven JSON inputs.
        output_packages[:] = [input_args[5]]
        input_args.pop()
        # The API expects int8; ATK converts the JSON integer to c_long.
        import ctypes

        input_args[6] = ctypes.c_int8(input_data.kwargs["cubeMathType"])
        return input_args, output_packages

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if index == 2 or name == "mat2":
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
        if name == "mat2" or index == 2:
            shape = input_data.kwargs["mat2"].shape
            return (*shape[:-2], (shape[-1] + 15) // 16, (shape[-2] + 15) // 16, 16, 16)
        return None
