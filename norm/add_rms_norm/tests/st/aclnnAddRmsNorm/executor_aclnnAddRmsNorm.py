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


@register("aclnn_add_rms_norm")
class FunctionRmsNormGradApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["epsilon"] = 0.000001

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x1 = input_data.kwargs["x1"]
        input_x2 = input_data.kwargs["x2"]
        input_gamma = input_data.kwargs["gamma"]
        epsilon = input_data.kwargs["epsilon"]
        kernelType = input_data.kwargs["kernelType"]
        if self.device == "cpu":
            output1, output2, output3 = self.npu_add_rms_norm_golden(input_x1, input_x2, input_gamma, kernelType, epsilon)
        return output1, output2, output3
    
    def npu_add_rms_norm_golden(self, input_x1, input_x2, input_gamma, kernelType, epsilon=0.000001):
        ori_x_shape = input_x1.shape
        ori_gamma_shape = input_gamma.shape
        xlength = len(ori_x_shape)
        gammaLength = len(ori_gamma_shape)
        torchType32 = torch.float32
        rstdShape = []
        
        for i in range(xlength):
          if i < (xlength - gammaLength):
            rstdShape.append(ori_x_shape[i])
          else:
            rstdShape.append(1)
        n = xlength - gammaLength
        input_gamma = input_gamma.reshape(np.multiply.reduce(np.array(ori_gamma_shape)))
        x1_shape = ori_x_shape[0:n] + input_gamma.shape
        input_x1 = input_x1.reshape(x1_shape)
        input_x2 = input_x2.reshape(x1_shape)
        
        if kernelType == 1:
          oriType = torch.float16
          xOut = (input_x1.to(oriType) + input_x2.to(oriType))
        elif kernelType == 2:
          oriType = torch.bfloat16
          x_fp32 = (input_x1.to(torchType32) + input_x2.to(torchType32))
          xOut = x_fp32.to(oriType)
        else:
          oriType = torch.float32
          xOut = (input_x1.to(torchType32) + input_x2.to(torchType32))
        x_fp32 = xOut.to(torchType32)
        # rmsNorm  
        variance = torch.mean(torch.pow(x_fp32, 2), axis=-1, keepdims=True)
        std = torch.sqrt(variance + epsilon)
        
        rstd = 1 / std
        result_mid = x_fp32 * rstd
        if kernelType == 1:
          result_mid_ori = result_mid.to(oriType)
          y_array = result_mid_ori.to(torchType32) * input_gamma.to(torchType32)
        elif kernelType == 2:
          result_mid_ori = result_mid.to(oriType)
          y_array = result_mid_ori * input_gamma.to(oriType)
        else:
          y_array = result_mid.to(torchType32) * input_gamma.to(torchType32)
        rstdOut = rstd.reshape(rstdShape).to(torchType32)
        yOut = y_array.reshape(ori_x_shape).to(oriType)
        xOut = x_fp32.reshape(ori_x_shape).to(oriType)
        return yOut, rstdOut, xOut
        

@register("sample_aclnn_api")
class SampleAclnnApi(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def init_by_input_data(self, input_data: InputDataset):
        del input_data.kwargs["kernelType"]
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages
    # pyaclnn的自定义api
    def after_call(self, output_packages):
        output1, output2, output3 = super().after_call(output_packages)
        return output1, output2, output3

