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

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr


@register("function_aclnn_unique2")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            input = input_data.kwargs["self"]
            sorted = input_data.kwargs["sorted"]
            returnInverse = input_data.kwargs["returnInverse"]
            valueOut, inverseOut, countsOut = torch.unique(input, sorted, returnInverse, True)
        return valueOut, inverseOut, countsOut.zero_()


@register("aclnn_unique2")
class exec_unique2(AclnnBaseApi):
    def __call__(self):
        super().__call__()
    def after_call(self, output_packages):
        output = []
        value_tensor = self.acl_tensor_to_torch(output_packages[0])
        inverse_tensor = self.acl_tensor_to_torch(output_packages[1])
        output.append(value_tensor)
        output.append(inverse_tensor)
        shape = tuple(value_tensor.shape)
        dtype = inverse_tensor.dtype
        counts_tensor = torch.zeros(shape).to(dtype)
        output.append(counts_tensor)
        return output
    def get_cpp_func_signature_type(self):
        return "aclnnUnique2GetWorkspaceSize(const aclTensor* self, bool sorted, bool returnInverse, bool returnCounts, aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut, uint64_t* workspaceSize, aclOpExecutor** executor)"