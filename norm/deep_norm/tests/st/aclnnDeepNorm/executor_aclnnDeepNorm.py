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

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

try:
    import torch_npu
except Exception as e:
    logging.warning("import torch_npu failed!!!")

alpha = 0.3
eps = 1e-6

def compute_golden(input_x, input_gx, input_beta, input_gamma, input_dtype, calc_alpha=0.3, eps=1e-06, reduce_axis=-1):
    # 计算标杆时，都需要先转成fp32计算标杆,最后再转回来
    calc_x = input_x
    calc_gx = input_gx
    calc_beta = input_beta
    calc_gamma = input_gamma

    x = calc_x * calc_alpha + calc_gx
    mean = np.mean(x, reduce_axis, keepdims=True)
    diff = x - mean
    variance = np.mean(np.power(diff, 2), reduce_axis, keepdims=True)
    rstd = 1 / np.sqrt(variance + eps)
    res = calc_gamma * (diff) * rstd + calc_beta
    if torch.bfloat16 == input_dtype:
        golden_res = torch.from_numpy(res.astype(np.float32)).to(torch.bfloat16)
    elif  torch.float16 == input_dtype:
        golden_res = torch.from_numpy(res.astype(np.float32)).to(torch.float16)
    else:
        golden_res = torch.from_numpy(res.astype(np.float32))
    mean = torch.from_numpy(mean.astype(np.float32))
    rstd = torch.from_numpy(rstd.astype(np.float32))
    return mean, rstd, golden_res


@register("aclnn_deep_norm")
class DeepNormFunctionApi(BaseApi):
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnDeepNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gx, const aclTensor *beta, const aclTensor *gamma, double alpha, double epsilon, const aclTensor *meanOut, const aclTensor *rstdOut, const aclTensor *yOut, uint64_t *workspaceSize, aclOpExecutor **executor)"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "npu":
            import torch_npu
            torch_npu.npu.set_compile_mode(jit_compile=False)

        calc_x = input_data.kwargs["x"]
        calc_gx = input_data.kwargs["gx"]
        calc_gamma = input_data.kwargs["gamma"]
        calc_beta = input_data.kwargs["beta"]
        calc_alpha = 0.3

        len_shape_x = len(calc_x.shape)
        len_shape_gamma = len(calc_gamma.shape)
        reduce_axis = tuple(range(len_shape_x - len_shape_gamma, len_shape_x, 1))

        dtype = input_data.kwargs["x"].dtype

        eps = 1e-6


        print(calc_x.dtype)
        if calc_x.dtype in [torch.bfloat16]:
            calc_x = calc_x.float()
            calc_gx = calc_gx.float()
            calc_beta = calc_beta.float()
            calc_gamma = calc_gamma.float()

        if self.device == "cpu" or self.device == "gpu":
            return compute_golden(calc_x.detach().clone().cpu().numpy(),
                                  calc_gx.detach().clone().cpu().numpy(),
                                  calc_beta.detach().clone().cpu().numpy(),
                                  calc_gamma.detach().clone().cpu().numpy(),
                                  dtype, calc_alpha, eps, reduce_axis)


@register("pyaclnn_deep_norm")
class SampleAclnnApi(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        output1, output2, output3, output4 = super().after_call(output_packages)
        return output1, output2, output3