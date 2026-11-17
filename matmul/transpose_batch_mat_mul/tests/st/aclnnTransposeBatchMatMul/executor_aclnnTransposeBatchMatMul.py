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

import ctypes

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import Int64, TensorPtr, nnopbase

# perm 序列取值映射：0 -> 不转置，1 -> batch 维与 M 维交换 / K·N 维交换
PERM_X1 = {0: [0, 1, 2], 1: [1, 0, 2]}
PERM_X2 = {0: [0, 1, 2], 1: [0, 2, 1]}
PERM_Y = [1, 0, 2]


def make_int_array(values):
    return nnopbase.aclCreateIntArray((Int64 * len(values))(*values), len(values))


@register("execute_transpose_batch_mat_mul")
class TransposeBatchMatMulApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(TransposeBatchMatMulApi, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        self.x1 = input_data.kwargs["x1"].clone()
        self.x2 = input_data.kwargs["x2"].clone()
        self.perm_x1 = tuple(PERM_X1[int(input_data.kwargs["permX1Mode"])])
        self.perm_x2 = tuple(PERM_X2[int(input_data.kwargs["permX2Mode"])])
        self.cube_math_type = int(input_data.kwargs["cubeMathType"])
        self.batch_split_factor = int(input_data.kwargs["batchSplitFactor"])

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x1, x2 = self.x1, self.x2
        # cubeMathType==2 (USE_FP16) 时 fp32 输入降精度为 fp16 计算
        if self.cube_math_type == 2 and x1.dtype == torch.float32:
            x1 = x1.to(torch.float16).to(torch.float32)
            x2 = x2.to(torch.float16).to(torch.float32)
        if self.perm_x1 != (0, 1, 2):
            x1 = x1.permute(*self.perm_x1)
        if self.perm_x2 == (0, 2, 1):
            x2 = x2.transpose(-2, -1)
        # fp16/bf16 输入升精度到 fp32 计算后再转回输出精度
        y = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))
        if self.batch_split_factor > 1:
            batch, m, n = y.shape
            inner_batch = batch // self.batch_split_factor
            out = y.reshape(self.batch_split_factor, inner_batch, m, n).permute(
                0, 2, 1, 3
            )
            out = out.reshape(self.batch_split_factor, m, inner_batch * n)
        else:
            out = y.permute(1, 0, 2)
        return out.to(self.x1.dtype)


@register("execute_aclnn_transpose_batch_mat_mul")
class AclnnTransposeBatchMatMulApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(AclnnTransposeBatchMatMulApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        perm_x1 = PERM_X1[int(input_data.kwargs["permX1Mode"])]
        perm_x2 = PERM_X2[int(input_data.kwargs["permX2Mode"])]
        cube_math_type = int(input_data.kwargs["cubeMathType"])
        batch_split_factor = int(input_data.kwargs["batchSplitFactor"])

        input_args, output_packages = super().init_by_input_data(input_data)

        # bias / scale 为预留参数，当前置空
        input_args[2] = TensorPtr()
        input_args[3] = TensorPtr()
        # perm 序列以 aclIntArray 传入
        input_args[4] = make_int_array(perm_x1)
        input_args[5] = make_int_array(perm_x2)
        input_args[6] = make_int_array(PERM_Y)
        # cubeMathType 为 int8_t，batchSplitFactor 为 int32_t
        input_args[7] = ctypes.c_byte(cube_math_type)
        input_args[8] = ctypes.c_int(batch_split_factor)

        return input_args, output_packages

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(const aclTensor *x1, "
            "const aclTensor *x2, const aclTensor *bias, const aclTensor *scale, "
            "const aclIntArray *permX1, const aclIntArray *permX2, const aclIntArray *permY, "
            "int8_t cubeMathType, const int32_t batchSplitFactor, aclTensor *out, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )
