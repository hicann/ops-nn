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
import time
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_repeat_interleave_int_with_dim")
class AsceneFunitionRepeatInterleaveIntWithDimApi(BaseApi):  
    
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        OpsDataset.change_flag = None
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        start = time.time()
        output = torch.repeat_interleave(input_data.kwargs["self"], input_data.kwargs["repeats"],
                                         input_data.kwargs["dim"])
        end = time.time()
        self_shape = input_data.kwargs["self"].shape
        repeats_v = input_data.kwargs["repeats"]
        dim_v = input_data.kwargs["dim"]
        print(f"标杆运行时常：{end - start} s; self.shape: {self_shape}, rpeats: {repeats_v}, dim: {dim_v}")
        return output
    
    def init_by_input_data(self, input_data: InputDataset):
        repeat = input_data.kwargs["repeats"]
        dim = input_data.kwargs["dim"]
        input_data.kwargs["outputSize"] = input_data.kwargs["self"].shape[dim] * repeat
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(const aclTensor* self, int64_t repeats, int64_t dim, int64_t outputSize, aclTensor *out, uint64_t* workspaceSize, aclOpExecutor** executor)"
