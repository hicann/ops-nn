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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

def HNC_2_NCH(tensor):
    tensor = tensor.transpose(0, 1)
    tensor = tensor.transpose(1, 2)
    return tensor

def KCiCo_2_CoCiK(tensor):
    tensor = tensor.transpose(0, 2)
    return tensor

def NCH_2_HNC(tensor):
    tensor = tensor.transpose(0, 2)
    tensor = tensor.transpose(1, 2)
    return tensor

@register("golden_convtbc")
class ConvTbcGolden(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(ConvTbcGolden, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        input_data.kwargs['self'] = HNC_2_NCH(input_data.kwargs['self'])
        input_data.kwargs['weight'] = KCiCo_2_CoCiK(input_data.kwargs['weight'])

        if self.device == 'cpu':
            output = eval(self.api_name)(input_data.kwargs['self'].to(torch.float32),
                                        input_data.kwargs['weight'].to(torch.float32), 
                                        input_data.kwargs['bias'].to(torch.float32),
                                        padding = input_data.kwargs['pad']
                                        )

        return NCH_2_HNC(output.cpu()).to(input_data.kwargs['self'].dtype)


@register("aclnn_convTbc")
class exec_convTbc(AclnnBaseApi):

    def get_format(self, input_data:InputDataset, index= None, name=None):
        return AclFormat.ACL_FORMAT_NCL

    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)
        if sum(input_args[2].pytensor.shape) == 0:
            input_args[2] = TensorPtr()
        input_args[4], input_args[5] = input_args[5], input_args[4]
        output_packages[:] = [input_args[4]]
        return input_args, output_packages