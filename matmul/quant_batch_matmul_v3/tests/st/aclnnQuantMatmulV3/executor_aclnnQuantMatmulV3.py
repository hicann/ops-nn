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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

@register("execute_quant_matmul_v3")
class quantMatmulApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(quantMatmulApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆计算
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        scale = input_data.kwargs["scale"]
        transposeX1 = input_data.kwargs["transposeX1"]
        transposeX2 = input_data.kwargs["transposeX2"]
        x1 = x1.to(torch.int32)
        x2 = x2.to(torch.int32)
        if transposeX1:
            x1 = x1.t()
        if transposeX2:
            x2 = x2.t()
        output = torch.matmul(x1, x2)

        # case: output=[m, n], scale=[n]
        output = output * scale

        return output

@register("execute_aclnn_quant_matmul_v3")
class aclnnQuantMatmulV3Api(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnQuantMatmulV3Api, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):

        input_data.kwargs["offset"] = TensorPtr()
        input_data.kwargs["bias"] = TensorPtr()
        input_args, output_packages = super().init_by_input_data(input_data)

        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output