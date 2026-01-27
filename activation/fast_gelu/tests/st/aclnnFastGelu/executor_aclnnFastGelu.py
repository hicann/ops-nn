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

import os
import random
import copy
import numpy as np
import torch

import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

def fast_gelu(input_x: torch.Tensor):
    # print("输入 dtype:", input_x.dtype)

    muls_const_value = -1.702
    exp_x = torch.exp(muls_const_value * input_x)   # 用 torch.exp
    div_down = 1 + exp_x
    res = input_x / div_down

    # print("输出 dtype:", res.dtype)
    return res

@register("aclnn_fast_gelu")
class FastGeluDisApi(BaseApi):
    # # a = None
    # def init_by_input_data(self, input_data: InputDataset):
    #     self.Input_Data = copy.deepcopy(input_data)  # 备份一个做切分之前的原始参数
    #     input_data.kwargs["self"] = input_data.kwargs["self"][..., ::2]

    # def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
    #     if name is not None:
    #         return self.Input_Data.kwargs[name].shape  # 把原始参数里的shape作为storage shape给aclnn用
    #     else:
    #         return None

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        input_tensor = input_data.kwargs["self"]
        # if FastGeluDisApi.a == None:
        #     FastGeluDisApi.a = input_tensor.dtype
            # print(FastGeluDisApi.a)
        # if self.device == "cpu":
            # if FastGeluDisApi.a == torch.float16:
        y = fast_gelu(input_tensor)
        # input_tensor = input_tensor.half()
        # y = input_tensor * torch.sigmoid(1.702 * input_tensor)
    
    
        return y
        