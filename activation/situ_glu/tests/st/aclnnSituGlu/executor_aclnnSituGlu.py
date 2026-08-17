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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


def _situ_glu_forward(x, dim=-1, beta=1.0, linear_beta=0.0, activate_left=True):
    dtype = x.dtype
    if x.dtype in [torch.float16, torch.bfloat16]:
        x = x.to(torch.float32)

    dim_pos = dim if dim >= 0 else dim + x.dim()
    pre = 1
    for i in range(dim_pos):
        pre *= x.shape[i]
    xf = x.reshape(pre, -1)
    h = xf.shape[1] // 2

    if activate_left:
        gate = xf[:, :h]
        up = xf[:, h:]
    else:
        gate = xf[:, h:]
        up = xf[:, :h]

    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta > 0.0:
        up = linear_beta * torch.tanh(up / linear_beta)
    y = situ_a * up

    out_shape = list(x.shape)
    out_shape[dim_pos] = out_shape[dim_pos] // 2
    y = y.reshape(out_shape)

    if dtype == torch.bfloat16:
        y = y.to(torch.bfloat16)
    elif dtype == torch.float16:
        y = y.to(torch.float16)
    return y


@register("function_situ_glu")
class FunctionSituGlu(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionSituGlu, self).__init__(task_result)
        self.run_modes = []

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        dim = int(input_data.kwargs.get("dim", -1))
        beta = float(input_data.kwargs.get("beta", 1.0))
        linear_beta = float(input_data.kwargs.get("linearBeta", 0.0))
        activate_left = bool(input_data.kwargs.get("activateLeft", True))

        return _situ_glu_forward(x, dim, beta, linear_beta, activate_left)

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND


@register("aclnn_situ_glu")
class AclnnSituGluApi(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super(AclnnSituGluApi, self).__init__(task_result, backend)
        self.run_modes = []

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        kw = dict(input_data.kwargs) if input_data.kwargs else {}
        input_args[2] = ctypes.c_double(float(kw.get("beta", 1.0)))
        input_args[3] = ctypes.c_double(float(kw.get("linearBeta", 0.0)))
        return input_args, output_packages

    @classmethod
    def get_cpp_func_signature_type(cls):
        return (
            "aclnnStatus aclnnSituGluGetWorkspaceSize("
            "const aclTensor* x, "
            "int64_t dim, "
            "double beta, "
            "double linearBeta, "
            "bool activateLeft, "
            "const aclTensor* out, "
            "uint64_t* workspaceSize, "
            "aclOpExecutor** executor)"
        )
