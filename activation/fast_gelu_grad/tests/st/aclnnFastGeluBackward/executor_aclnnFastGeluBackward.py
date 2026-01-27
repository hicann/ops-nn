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

import numpy as np
import torch

import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("aclnn_fast_gelu_backward_fp16")
class FastGeluBackwardApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        grad_output = input_data.kwargs["gradOutput"]
        x = input_data.kwargs["self"]
        # if grad_output.dtype = float16
        #     grad_output = grad_output.half()
        #     x = x.half()
        #     print(grad_output.dtype)
        print(grad_output.dtype)
        input_dy = grad_output
        input_x = x
        const_2 = 1.702
        const_3 = 1.0
        const_1 = 0.0 - const_2
        const_one = const_3
        const_attr = const_1
        const_neg_one = 0.0 - const_3
        mul_x = x * const_attr  # -1.702 * x
        sig_mul_x = torch.exp(mul_x)  # exp(-1.702 * x)
        sig_mul_x = sig_mul_x + const_one  # exp(-1.702 * x) + 1
        sig_mul_x = 1 / sig_mul_x  # 1 / (exp(-1.702 * x) + 1)

        result_temp = sig_mul_x - const_3  # 1 / (exp(-1.702 * x) + 1) - 1
        result_temp = mul_x * result_temp  # -1.702 * x * (1 / (exp(-1.702 * x) + 1) - 1)
        result_temp = result_temp + const_one  # +1
        result_temp = result_temp * sig_mul_x  # * (1 / (exp(-1.702 * x) + 1))

        # 计算反向传播的梯度
        output_data = grad_output * result_temp
        # print(output_data.dtype)
        # output_data = torch.tensor(output_data)   
        return output_data