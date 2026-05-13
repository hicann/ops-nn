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
import numpy as np


@register("function_ascend_gather")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):       
        self_tensor = input_data.kwargs["self"]
        dim = input_data.kwargs["dim"]
        index_tensor = input_data.kwargs["index"]

        if self.output is None:
            # 索引不支持int32，做cast处理      
            if index_tensor.dtype == torch.int32:
                index_tensor = index_tensor.to(torch.int64)           
            
            output = torch.gather(self_tensor, dim, index_tensor)
             
        else:
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
                
        return output        
