#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import random
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("aclnn_avgpool3d")
class AclnnAvgPool3dApi(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if self.device == "npu":
            torch.npu.set_compile_mode(jit_compile=False)
            device = f"npu:{self.device_id}"
            model = eval(self.api_name)(
                                        input_data.kwargs['kernel_size'], 
                                        input_data.kwargs['stride'],
                                        input_data.kwargs['padding'],
                                        input_data.kwargs['ceil_mode'],
                                        input_data.kwargs['count_include_pad'],
                                        input_data.kwargs['divisor_override']
                                        ).to(device)
            output = model(input_data.kwargs['input'].to(device))

        if self.device == 'cpu':
            model = eval(self.api_name)(
                                        input_data.kwargs['kernel_size'], 
                                        input_data.kwargs['stride'],
                                        input_data.kwargs['padding'],
                                        input_data.kwargs['ceil_mode'],
                                        input_data.kwargs['count_include_pad'],
                                        input_data.kwargs['divisor_override']
                                        )
            ori_type = input_data.kwargs['input'].dtype
            output = model(input_data.kwargs['input'].to(torch.float32))
            output = output.to(ori_type)
        
        if self.device == 'gpu':
            device = f"cuda:{self.device_id}"
            model = eval(self.api_name)(
                                        input_data.kwargs['kernel_size'], 
                                        input_data.kwargs['stride'],
                                        input_data.kwargs['padding'],
                                        input_data.kwargs['ceil_mode'],
                                        input_data.kwargs['count_include_pad'],
                                        input_data.kwargs['divisor_override']
                                        ).to(device)
            output = model(input_data.kwargs['input'].to(device))
            
        return output.cpu()
    
    def get_format(self, input_data: InputDataset, index=None, name=None):
        if len(input_data.kwargs['input'].shape) == 4:
            return AclFormat.ACL_FORMAT_ND
        return AclFormat.ACL_FORMAT_NCDHW
