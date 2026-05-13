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
import ctypes

@register("executor_mv")
class mvApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(mvApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆计算
        mat_self = input_data.kwargs["self"]
        vec = input_data.kwargs["vec"]
        output = torch.mv(mat_self, vec)
        return output

@register("executor_aclnn_mv")
class aclnnMvApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnMvApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # 输出在输入前
        output_packages[:] = [input_args[2]]
        input_args.pop()
        # atk不支持int8 json输入，在此转换
        input_args[-1] = ctypes.c_int8(input_data.kwargs["cubeMathType"])
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output