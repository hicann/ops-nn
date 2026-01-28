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
#

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

@register("aclnn_dynamic_quant")
class aclnnDynamicQuant(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        max_value = 127.0
        min_value = -128.0
        x = input_data.kwargs["x"]
        smooth = input_data.kwargs["smoothScaleOptional"]
        x_fp32 = x.to(torch.float)
        smooth_fp32 = smooth.to(torch.float)
        x_fp32 = x_fp32 * smooth_fp32
        input_max = torch.abs(x_fp32).max(dim = -1, keepdim = True)[0]
        scale = input_max / max_value
        out = x_fp32 * (1 / scale)
        out = torch.clamp(torch.round(out), min_value, max_value).to(torch.int8)
        quant_scale = scale.view(*scale.shape[:-1])
        return out, quant_scale