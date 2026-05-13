#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("executor_quant_matmul_weight_nz")
class quantMatmulWeightNzApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(quantMatmulWeightNzApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆计算
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        x2Scale = input_data.kwargs["x2Scale"]
        bias = input_data.kwargs["bias"]
        transposeX1 = input_data.kwargs["transposeX1"]
        transposeX2 = input_data.kwargs["transposeX2"]
        if transposeX1:
            x1 = x1.transpose(1, 2)
        if transposeX2:
            x2 = x2.transpose(1, 2)
        output = torch.matmul(x1,x2) * x2Scale + bias
        output = output.to(torch.bfloat16)
        return output

@register("executor_aclnn_quant_matmul_weight_nz")
class aclnnQuantMatmulWeightNzApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnQuantMatmulWeightNzApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):

        input_data.kwargs["x1Scale"] = TensorPtr()
        input_data.kwargs["yScale"] = TensorPtr()
        input_data.kwargs["x1Offset"] = TensorPtr()
        input_data.kwargs["x2Offset"] = TensorPtr()
        input_data.kwargs["yOffset"] = TensorPtr()

        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output
    
    def get_format(self, input_data: InputDataset, index=None, name=None):
        if index == 1:
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND