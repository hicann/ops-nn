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

@register("aclnn_avgpool2dgrad")
class AclnnAvgPool2dApi(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if self.device == 'cpu':
            dtype = input_data.kwargs['gradOutput'].dtype
            out_put = torch.ops.aten.avg_pool2d_backward(input_data.kwargs['gradOutput'].to(torch.float32),
                                                         input_data.kwargs['self'].to(torch.float32),
                                                         input_data.kwargs['kernel_size'],
                                                         input_data.kwargs['stride'],
                                                         input_data.kwargs['padding'],
                                                         input_data.kwargs['ceil_mode'],
                                                         input_data.kwargs['count_include_pad'],
                                                         input_data.kwargs['divisor_override']) 
            return out_put.to(dtype)
        return output

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_NCHW
