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
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("aclnn_logsoftmax")
class AclnnLogSoftmaxApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        origin_dtype = input_data.kwargs["self"].dtype
        dim = input_data.kwargs["dim"]
        input_d = input_data.kwargs["self"]

        m = torch.nn.LogSoftmax(dim=dim)
        if (input_d.dtype == torch.float16) or (input_d.dtype == torch.bfloat16):
            output = m(input_d.to(torch.float32)).to(origin_dtype)
        else:
            output = m(input_d)

        return output