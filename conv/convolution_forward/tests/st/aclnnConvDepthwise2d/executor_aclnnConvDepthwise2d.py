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

@register("golden_conv_depthwise_2d")
class ConvDepthwise2dGolden(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(ConvDepthwise2dGolden, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def change_padding(self, padding):
        if len(padding) == 2:
            return True
        return False
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None

        if self.device == 'cpu':
            padding = input_data.kwargs['padding']
            fmap = input_data.kwargs['self'].to(torch.float32)
            change_padding_flag = self.change_padding(padding)
            output = eval(self.api_name)(torch.nn.functional.pad(fmap, pad=padding) if change_padding_flag else fmap,
                                        input_data.kwargs['weight'].to(torch.float32), 
                                        input_data.kwargs['bias'].to(torch.float32),
                                        stride = input_data.kwargs['stride'],
                                        padding = 0 if change_padding_flag else padding,
                                        dilation = input_data.kwargs['dilation'],
                                        groups = input_data.kwargs['self'].shape[1]
                                        )

        return output.cpu().to(input_data.kwargs['self'].dtype)

@register("aclnn_conv_depthwise_2d")
class exec_convDepthwise2d(AclnnBaseApi):

    def get_format(self, input_data:InputDataset, index= None, name=None):
        return AclFormat.ACL_FORMAT_NCHW

    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args[7], input_args[8] = input_args[8], input_args[7]
        output_packages[:] = [input_args[7]]
        return input_args, output_packages