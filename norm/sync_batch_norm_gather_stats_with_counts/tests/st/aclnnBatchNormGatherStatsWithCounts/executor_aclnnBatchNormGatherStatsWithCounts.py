#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging
import torch
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

try:
    pass
except Exception:
    logging.warning("import torch_npu failed!!!")


@register("function_aclnnBatchNormGatherStatsWithCounts")
class FunctionaclnnBatchNormGatherStatsWithCounts(BaseApi):
    # 精度对比标杆
    def get_golden(
        self, input, mean, invstd, running_mean, running_var, momentum, eps, counts
    ):
        dtype = input.dtype
        mean = mean.float()
        invstd = invstd.float()
        running_mean = running_mean.float().unsqueeze(0)
        running_var = running_var.float().unsqueeze(0)
        counts = counts.float().unsqueeze(-1)
        count_broadcast = torch.broadcast_to(counts, invstd.shape)
        counts_all_sum = torch.sum(
            count_broadcast,
            [
                0,
            ],
            keepdim=True,
        )

        sum_out_broadcast = counts_all_sum.expand(count_broadcast.shape)
        x_count = mean.mul(count_broadcast)
        x_mean = x_count.div(sum_out_broadcast)
        mean_all = torch.sum(
            x_mean,
            [
                0,
            ],
            keepdim=True,
        )

        mean_broadcast = mean_all.expand(mean.shape)
        var_all = torch.reciprocal(invstd)
        var_all_square = torch.mul(var_all, var_all)
        var_all_square_epsilon = torch.add(var_all_square, -eps)
        mean_sub = torch.sub(mean, mean_broadcast)
        mean_var = torch.mul(mean_sub, mean_sub)
        mean_var_sum = torch.add(var_all_square_epsilon, mean_var)
        mean_var_count = torch.mul(mean_var_sum, count_broadcast)
        var_sum = torch.sum(
            mean_var_count,
            [
                0,
            ],
            keepdim=True,
        )
        var_sum_count = torch.div(var_sum, counts_all_sum)
        var_sum_count_epsilon = torch.add(var_sum_count, eps)
        var_sqrt = var_sum_count_epsilon.sqrt()
        invert_std = torch.reciprocal(var_sqrt)

        mean_all.squeeze_(0)
        invert_std.squeeze_(0)
        if dtype == torch.float16:
            return mean_all.to(dtype), invert_std.to(dtype)
        return mean_all, invert_std

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            input_data.kwargs["momentum"] = ctypes.c_double(
                input_data.kwargs["momentum"]
            )
            input_data.kwargs["eps"] = ctypes.c_double(input_data.kwargs["eps"])

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input = input_data.kwargs["input"]
        mean = input_data.kwargs["mean"]
        invstd = input_data.kwargs["invstd"]
        runningMean = input_data.kwargs["runningMean"]
        runningVar = input_data.kwargs["runningVar"]
        momentum = input_data.kwargs["momentum"]
        eps = input_data.kwargs["eps"]
        counts = input_data.kwargs["counts"]

        meanAll, invstdAll = self.get_golden(
            input, mean, invstd, runningMean, runningVar, momentum, eps, counts
        )
        return meanAll, invstdAll

    # 函数入参签名校验
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(const aclTensor* input, const aclTensor* mean, \
                                                                const aclTensor* invstd, aclTensor* runningMean, \
                                                                aclTensor* runningVar, double momentum, double eps, \
                                                                const aclTensor* counts, aclTensor* meanAll, \
                                                                aclTensor* invstdAll, uint64_t* workspaceSize, \
                                                                aclOpExecutor** executor)"
