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
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

@register("execute_weight_quant_batch_matmul_v3")
class weightQuantBatchMatmulV3Api(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 标杆计算
        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        antiquantScale = input_data.kwargs["antiquantScale"]
        antiquantOffset = input_data.kwargs["antiquantOffsetOptional"]
        bias = input_data.kwargs["biasOptional"]
        groupSize = input_data.kwargs["antiquantGroupSize"]

        weight = weight.type(torch.float16)
        weight = weight.numpy()
        x = x.numpy()
        antiquantScale = antiquantScale.numpy()
        kSize = x.shape[-1]

        if antiquantOffset is not None:
            antiquantOffset = antiquantOffset.numpy()
        
        if groupSize:
            for i in range(kSize):
                group_size_dim = i // groupSize
                if (antiquantOffset is not None) and (len(antiquantOffset) != 0):
                    weight[...,i,:] = (weight[...,i,:] + antiquantOffset[group_size_dim, :]) * antiquantScale[group_size_dim, :]
                else:
                    weight[...,i,:] = weight[...,i,:] * antiquantScale[group_size_dim, :]
        else:
            if (antiquantOffset is not None) and (len(antiquantOffset) != 0):
                weight = (weight + antiquantOffset) * antiquantScale
        
        weight = torch.from_numpy(weight).to(torch.float32)
        x = torch.from_numpy(x).to(torch.float32)
        output = torch.matmul(x, weight)

        if bias is not None:
            output = output + bias.to(torch.float32)
        output = output.to(torch.float16)

        return output

@register("execute_aclnn_weight_quant_batch_matmul_v3")
class aclnnWeightQuantBatchMatmulV3Api(AclnnBaseApi):

    def init_by_input_data(self, input_data: InputDataset):
        # atk暂不支持quantScale的输入type：uint64，设为nullptr
        # quantScaleOptional is a nullptr
        input_data.kwargs["quantScaleOptional"] = TensorPtr()
        # quantOffsetOptional is a nullptr
        input_data.kwargs["quantOffsetOptional"] = TensorPtr()
        # change type c_long to c_int32
        input_data.kwargs["antiquantGroupSize"] = ctypes.c_int32(input_data.kwargs["antiquantGroupSize"])
        # change type c_long to c_int32
        input_data.kwargs["innerPrecise"] = ctypes.c_int32(input_data.kwargs["innerPrecise"])
        input_args, output_packages = super().init_by_input_data(input_data)

        return input_args, output_packages
    
    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output