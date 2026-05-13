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


@register("aclnn_adaptivemaxpool3dgrad")            
class AclnnAdaptiveMaxPool3dGradApi(BaseApi):  
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if self.device == 'cpu':
            dtype = input_data.kwargs['self'].dtype
            m = torch.nn.AdaptiveMaxPool3d(input_data.kwargs['output_size'], True)
            output , indices = m(input_data.kwargs['self'].to(torch.float32))

            gradout = torch.ops.aten.adaptive_max_pool3d_backward(output, input_data.kwargs['self'].to(torch.float32), indices)
            return gradout.to(dtype)
        return output
        
    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'pyaclnn':
            m = torch.nn.AdaptiveMaxPool3d(input_data.kwargs['output_size'], True)
            originself = input_data.kwargs['self']
            origindtype = input_data.kwargs['self'].dtype
            output , indices = m(input_data.kwargs['self'].cpu().to(torch.float32))
            input_data.kwargs['indices'] = indices.to(torch.int32).npu()
            input_data.kwargs['gradOutput'] = output.to(origindtype).npu()
            input_data.kwargs['self'] = originself
            input_data.kwargs.pop('output_size')