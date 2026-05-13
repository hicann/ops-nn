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
import torch.nn as nn
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

from atk.common.log import Logger

logging = Logger().get_logger()

@register("ascend_functional_multilabel_margin_loss")
class FunctionalMultiLabelMarginLossApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionalMultiLabelMarginLossApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        torch_reduction = [ 'none', 'mean', 'sum']
        reduction = torch_reduction[input_data.kwargs['reduction']]
        loss = nn.MultiLabelMarginLoss(reduction=reduction)
        input=input_data.kwargs["self"]
        
        target=input_data.kwargs["target"]
        ori_input_dtype = input.dtype
        not_fp32 = ori_input_dtype != torch.float32
        if self.device == "cpu" and not_fp32:
            input = input.to(torch.float32)
        output = loss(input, target)
        if self.device == "cpu" and not_fp32:
            output = output.to(ori_input_dtype)
        return output


    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnMultilabelMarginLossGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction, aclTensor* out, aclTensor* isTarget, uint64_t* workspaceSize, aclOpExecutor** executor)"


from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("aclnn_ascend_functional_multilabel_margin_loss")
class AclnnFunctionalMultiLabelMarginLossApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args = []
        output_packages = []

        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)

        for name, kwarg in input_data.kwargs.items():
            if name == "self":
                self_dtype = kwarg.dtype
            if name == "self":
                value_to_insert = self.backend.convert_input_data(kwarg.to(self_dtype), name=name)[0]
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)

        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = str(input_data.kwargs["self"].dtype)
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)
        input_args.insert(4, value_to_insert)
        return input_args, output_packages
    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output
