#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclFormat
import numpy as np


@register("aclnn_batch_norm_elemt_backward")
class FunctionRmsNormGradApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # print("001\n")
        gradOut = input_data.kwargs["gradOut"]
        input_input = input_data.kwargs["input"]
        input_mean = input_data.kwargs["mean"]
        input_invstd = input_data.kwargs["invstd"]
        input_weight = input_data.kwargs["weight"]
        input_sumDy = input_data.kwargs["sumDy"]
        input_sumDyXmu = input_data.kwargs["sumDyXmu"]
        input_counter = input_data.kwargs["counter"]
        if self.device == "cpu":
            output1 = self.calc_expect_func(gradOut, input_input, input_mean, input_invstd,
                                      input_weight, input_sumDy, input_sumDyXmu, input_counter)
        return output1

    def calc_expect_func(self, grad_output, save_input, mean, invstd, weight, sumDy, sumDyXmu, counter):
      """
      calc_expect_func
      """
      gradOut = grad_output.to(torch.float32)
      inputx = save_input.to(torch.float32)
      mean = mean.to(torch.float32)
      invstd = invstd.to(torch.float32)
      weight = weight.to(torch.float32)
      sumDy = sumDy.to(torch.float32)
      sumDyXmu = sumDyXmu.to(torch.float32)
     
      j=0
      if counter.dim() != 0:
          for i in counter:
              j += i.item()
          for i in range(len(counter)):
              counter[i] = j
      target_shape = inputx.shape
      mean_dy = sumDy / counter
      mean_dy_xmu = sumDyXmu / counter
      c = invstd * invstd
      a = c * mean_dy_xmu
      b = invstd * weight
     
      output_dy = gradOut - expand_self(mean_dy, target_shape)
      input_mean = inputx - expand_self(mean, target_shape)
      input_invstd = input_mean * expand_self(a, target_shape)
      output_input = output_dy - input_invstd
      sumoutorigin = output_input * expand_self(b, target_shape)
      return sumoutorigin


def expand_self(mytensor, target_shape):
    if mytensor.dim() == 0:
        return torch.Tensor([])
    C = mytensor.size(0)
    expanded = mytensor.view(1, C, *([1] * (len(target_shape) - 2)))
    return expanded.expand(target_shape)