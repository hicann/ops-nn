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
from atk.tasks.dataset.base_dataset import OpsDataset
import random

@register("ascend_method_max_unpool2d")
class MethodMaxUnpool2dApi(BaseApi):

    def __init__(self, task_result: TaskResult):
        super(MethodMaxUnpool2dApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
        if self.device == "cpu":
            self.data_device = torch.device("cpu")
        elif self.device == "pyaclnn":
            self.data_device = torch.device(f"npu:0")

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        # out = torch.ops.aten.max_unpool2d(input_data.kwargs["self"], input_data.kwargs["indices"], 
        #                                   input_data.kwargs["outputSize"])
        # return out
        _self = input_data.kwargs["self"]
        indices = input_data.kwargs["indices"]
        outputSize = input_data.kwargs["outputSize"]

        outShape = [ 1, 1, outputSize[-2], outputSize[-1] ]
        dim = len(_self.shape)
        if dim == 3:
            outShape[1] = _self.shape[0]
            is_3D = True
        else:
            outShape[0] = _self.shape[0]
            outShape[1] = _self.shape[1]
            is_3D = False

        output = torch.zeros(outShape, dtype=_self.dtype)
        output_flat =  output.reshape(outShape[0], outShape[1], -1)
        indices_flat = indices.reshape(outShape[0], outShape[1], -1) 
        self_flat = _self.reshape(outShape[0], outShape[1], -1)
        output_flat.scatter_(2, indices_flat, self_flat)

        if is_3D:
            return output_flat.reshape(outShape[1:])
        else:
            return output_flat.reshape(outShape)
    
    def init_by_input_data(self, input_data: InputDataset):
        
        _self = input_data.kwargs["self"]

        self_N = 1
        self_C = 1
        self_H = _self.shape[-2]
        self_W = _self.shape[-1]

        dim = len(_self.shape)
        if dim == 3:
            self_C = _self.shape[0]
            indices_new = torch.arange(self_H * self_W).reshape(1, self_H, self_W)
            indices_new = indices_new.expand(self_C, self_H, self_W)
        else:
            self_N = _self.shape[0]
            self_C = _self.shape[1]
            indices_new = torch.arange(self_H * self_W).reshape(1, 1, self_H, self_W)
            indices_new = indices_new.expand(self_N, self_C, self_H, self_W)

        input_data.kwargs["outputSize"][0] = self_H
        input_data.kwargs["outputSize"][1] = self_W

        if self.device == "pyaclnn":
            import torch_npu
            input_data.kwargs["indices"] = indices_new.npu()
        else:
            input_data.kwargs["indices"] = indices_new.to(self.data_device)