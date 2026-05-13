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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("ascend_function_embedding_renorm")
class AscendFunctionEmbeddingRenorm(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AscendFunctionEmbeddingRenorm, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        input_self = input_data.kwargs["selfRef"]
        indices = input_data.kwargs["indices"]
        maxNorm = input_data.kwargs["maxNorm"]
        normType = input_data.kwargs["normType"]

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        output = torch.embedding_renorm_(input_self.to(device), indices.to(device), maxNorm, normType)

        return output


@register("aclnn_embedding_renorm")
class AclnnEmbeddingRenorm(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages
    
    def after_call(self, output_packages):
        output = []
        for output_param in output_packages:
            output.append(self.acl_tensor_to_torch(output_param))
        return output