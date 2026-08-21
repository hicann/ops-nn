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
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np


@register("function_aclnnUnique")
class aclnnUniqueExecutor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            input = input_data.kwargs["self"]
            sorted = input_data.kwargs["sorted"]
            returnInverse = input_data.kwargs["returnInverse"]
            if returnInverse == True:
                valueOut, inverseOut = torch.unique(input, sorted, returnInverse, False)
                return valueOut, inverseOut.zero_()
            else:
                valueOut = torch.unique(input, sorted, returnInverse, False)
                inverseOut = torch.zeros(input.shape, dtype=torch.int64)
            return valueOut, inverseOut
    
@register("aclnn_function")
class exec_unique2(AclnnBaseApi):
    def __call__(self):
        super().__call__()
    def after_call(self, output_packages):
        output = []
        value_tensor = torch.sort(self.acl_tensor_to_torch(output_packages[0]).cpu())[0]
        output.append(value_tensor)
        if len(output_packages) == 2:
            inverse_tensor = self.acl_tensor_to_torch(output_packages[1]).zero_()
            output.append(inverse_tensor)
        return output