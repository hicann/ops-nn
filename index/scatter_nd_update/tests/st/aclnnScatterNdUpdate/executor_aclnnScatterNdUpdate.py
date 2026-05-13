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
import tensorflow as tf
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset



@register("ascend_scatter_nd_update")
class ScatterNdUpdate(BaseApi):

    def __init__(self, task_result: TaskResult):
        super(ScatterNdUpdate, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
        if self.device == "cpu":
            self.data_device = torch.device("cpu")
        elif self.device == "pyaclnn":
            self.data_device = torch.device(f"npu:0")

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        varRef = input_data.kwargs["varRef"]
        indices = input_data.kwargs["indices"]
        updates = input_data.kwargs["updates"]

        out = tf.tensor_scatter_nd_update(tensor=varRef, indices=indices, updates=updates).numpy()
        return torch.from_numpy(out)
        
    def init_by_input_data(self, input_data):

        varRef = input_data.kwargs["varRef"]
        indices = input_data.kwargs["indices"]
        updates = input_data.kwargs["updates"]
    
        for i in range(len(indices.shape)):
            indices = torch.unique(indices, dim=i)

        part1 = list(tuple(indices.shape)[:-1])  
        idx = indices.shape[-1]
        part2 = list(varRef.shape[idx:])
        updates_shape = part1 + part2

        if updates.dtype == torch.float16:
            new_updates = torch.randn(updates_shape).to(torch.float16)
        elif updates.dtype == torch.float32:
            new_updates = torch.randn(updates_shape).to(torch.float32)
        elif updates.dtype == torch.int64:
            new_updates = torch.randint(0, 100, updates_shape).to(torch.int64)
        elif updates.dtype == torch.bool:
            new_updates = torch.randint(0, 2, updates_shape).bool()
        elif updates.dtype == torch.int8:
            new_updates = torch.randint(-127, 128, updates_shape).to(torch.int8)
        
        if self.device == "pyaclnn":
            import torch_npu
            input_data.kwargs["indices"] = indices.npu()
            input_data.kwargs["updates"] = new_updates.npu()
        else:
            input_data.kwargs["indices"] = indices.to(self.data_device)
            input_data.kwargs["updates"] = new_updates.to(self.data_device)





    
@register("pyaclnn_scatter_nd_update")
class ScatterNdUpdatePyaclnn(AclnnBaseApi):

    def init_by_input_data(self, input_data): 

        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages



