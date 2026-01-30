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


@register("ascend_method_scatter_value")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        _self = input_data.kwargs["self"]

        dim = input_data.kwargs["dim"]

        index = input_data.kwargs["index"]

        value = input_data.kwargs["value"]

        reduce = input_data.kwargs["reduce"]

        index = index.to(torch.int64)

        if reduce == 0:
            out = torch.scatter(_self, dim, index, value)
        
        elif reduce == 1:
            reduce = "add"
            out = torch.scatter(_self, dim, index, value, reduce=reduce)
        
        elif reduce == 2:
            reduce = "multiply"
            out = torch.scatter(_self, dim, index, value, reduce=reduce)

        return out
    