#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("ascend_function_renorm")
class AscendFunctionRenorm(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AscendFunctionRenorm, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        input_self = input_data.kwargs["self"]
        p = input_data.kwargs["p"]
        dim = input_data.kwargs["dim"]
        maxNorm = input_data.kwargs["maxNorm"]

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        output = torch.renorm(input_self.to(device), p, dim, maxNorm)

        return output

