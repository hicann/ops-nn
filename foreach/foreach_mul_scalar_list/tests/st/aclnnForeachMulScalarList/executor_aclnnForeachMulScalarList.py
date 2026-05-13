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


@register("function_aclnnForeachMulScalarList")
class aclnnForeachMulScalarListExecutor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["scalars"]

        output = torch._foreach_mul(x1, x2)

        return output

@register("aclnnfunction_aclnnForeachMulScalarList")
class aclnnfunctionExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.task_result.output_info_list = [self.task_result.output_info_list]
        return super().init_by_input_data(input_data)