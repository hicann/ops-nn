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


@register("function_aclnnForeachRoundOffNumberV2")
class aclnnForeachRoundOffNumberV2Executor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        roundMode = input_data.kwargs["roundMode"]

        ROUND_MODE_FLOOR = 2
        ROUND_MODE_CEIL = 3
        ROUND_MODE_ROUND = 1
        ROUND_MODE_TRUNC = 5
        ROUND_MODE_FRAC = 7

        if roundMode == ROUND_MODE_FLOOR:
            output = torch._foreach_floor(x)
        elif roundMode == ROUND_MODE_CEIL:
            output = torch._foreach_ceil(x)
        elif roundMode == ROUND_MODE_ROUND:
            output = torch._foreach_round(x)
        elif roundMode == ROUND_MODE_TRUNC:
            output = torch._foreach_trunc(x)
        elif roundMode == ROUND_MODE_FRAC:
            output = torch._foreach_frac(x)
        return output

@register("aclnnfunction_aclnnForeachRoundOffNumberV2")
class aclnnfunctionExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.task_result.output_info_list = [self.task_result.output_info_list]
        return super().init_by_input_data(input_data)