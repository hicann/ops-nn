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
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_repeat_interleave_with_dim")
class AsceneFunitionRepeatInterleaveIntWithDimApi(BaseApi):  
    
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        OpsDataset.change_flag = None
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = torch.repeat_interleave(input_data.kwargs["self"], input_data.kwargs["repeats"],
                                         input_data.kwargs["dim"])
        return output
    
    def init_by_input_data(self, input_data: InputDataset):
        output = torch.repeat_interleave(input_data.kwargs["self"].cpu(), input_data.kwargs["repeats"].cpu(),
                                         input_data.kwargs["dim"])
        print("outputSize:")
        print(output.shape[input_data.kwargs["dim"]])
        input_data.kwargs["outputSize"] = output.shape[input_data.kwargs["dim"]]
    
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(const aclTensor* self, const aclTensor* repeats, int64_t dim, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)"
