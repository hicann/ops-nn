#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.function_api import FunctionApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
import numpy as np


@register("batch_norm_reduce_backward_cpu")
class AclnnBatchNormReduceBackward(FunctionApi):
    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "gradOut" or name == "input" or index == 0:
            gradOut = input_data.kwargs["gradOut"]
            dims = gradOut.dim()
            if dims == 2:
                return AclFormat.ACL_FORMAT_NC
            elif dims == 3:
                return AclFormat.ACL_FORMAT_NCL
            elif dims == 4:
                return AclFormat.ACL_FORMAT_NCHW
            elif dims == 5:
                return AclFormat.ACL_FORMAT_NCDHW
        return AclFormat.ACL_FORMAT_ND


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        oritype = input_data.kwargs["input"].dtype
        gradOut = input_data.kwargs["gradOut"].numpy()
        input = input_data.kwargs["input"].numpy()
        mean = input_data.kwargs["mean"].numpy()
        invstd = input_data.kwargs["invstd"].numpy()
        weight = input_data.kwargs["weight"].numpy()

        gradOut = gradOut.astype(np.float32)
        input = input.astype(np.float32)
        mean = mean.astype(np.float32)
        invstd = invstd.astype(np.float32)
        weight = weight.astype(np.float32)

        dim_len = len(input.shape)
        dim_list = [i for i in range(dim_len)]
        del dim_list[1]
        axis = tuple(dim_list)

        broadcast_shape = [1] * input.ndim
        broadcast_shape[1] = input.shape[1]

        mean_expanded = mean.reshape(tuple(broadcast_shape))
        invstd_expanded = invstd.reshape(tuple(broadcast_shape))

        # 计算 x_hat
        x_hat = (input - mean_expanded) * invstd_expanded
        # 计算 gradWeight
        grad_weight = np.sum(gradOut * x_hat, axis=axis)
        # 计算 gradBias
        grad_bias = np.sum(gradOut, axis=axis)
        # 计算 sumDy
        sum_dy = np.sum(gradOut, axis=axis)
        # 计算 sumDyXmu
        sum_dy_x_mu = np.sum(gradOut * (input - mean_expanded), axis=axis)

        sum_dy = torch.tensor(sum_dy, dtype=oritype)
        sum_dy_x_mu = torch.tensor(sum_dy_x_mu, dtype=oritype)
        grad_weight = torch.tensor(grad_weight, dtype=oritype)
        grad_bias = torch.tensor(grad_bias, dtype=oritype)

        return sum_dy, sum_dy_x_mu, grad_weight, grad_bias
