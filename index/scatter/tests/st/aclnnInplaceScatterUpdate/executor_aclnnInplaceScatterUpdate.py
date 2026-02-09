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
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_scatter_update")
class FunctionIndexAddApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionIndexAddApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # inplace类型接口
        data = input_data.kwargs["data"]
        indices = input_data.kwargs["indices"]
        updates = input_data.kwargs["updates"]
        axis = input_data.kwargs["axis"]
        data_value = data.cpu().numpy()
        indices_value = indices.cpu().numpy()
        updates_value = updates.cpu().numpy()
        print(data_value.shape)
        print(indices_value)
        print(updates_value.shape)
        
        shape_0 = updates.shape[0]
        shape_1 = updates.shape[1]
        shape_2 = updates.shape[2]
        shape_3 = updates.shape[3]

        for i in range(shape_0):
            indices_key = indices_value[i]
            for j in range(shape_1):
                for k in range(shape_2):
                    for l in range(shape_3):
                        data_value[i][j][indices_key + k][l] = updates_value[i][j][k][l]

        return torch.from_numpy(data_value)


@register("pyaclnn_scatter_update")
class AddAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output


