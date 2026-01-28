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
import torch.nn
import functools
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.common.log import Logger
logging = Logger().get_logger()


@register("ascend_aclnn_swish")
class AclnnSwishApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        ori_dtype = input_data.kwargs['self'].dtype

        x = input_data.kwargs['self'].to(torch.float32)
        scale = torch.tensor(input_data.kwargs['betaOptional'], dtype=torch.float32)

        output = x / (1 + torch.exp(-scale * x))
        
        return output.to(ori_dtype)
