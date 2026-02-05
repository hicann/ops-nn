#!/usr/bin/env python3
# -- coding: utf-8 --
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

@register("ascend_aclnn_binary_cross_entropy_with_logits")
class MethodAclnnBinaryCrossEntropyWithLogitsApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnBinaryCrossEntropyWithLogitsApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取yaml中所需参数
        self.self_ = input_data.kwargs['self']
        self.target_ = input_data.kwargs['target']
        self.weight_optional_ = input_data.kwargs['weightOptional']
        self.pos_weight_optional_ = input_data.kwargs['posWeightOptional']
        self.reduction_ = input_data.kwargs['reduction']

        output_data = torch.ops.aten.binary_cross_entropy_with_logits(self.self_, target=self.target_, weight=self.weight_optional_, pos_weight=self.pos_weight_optional_, reduction=self.reduction_)
        return output_data