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
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset



@register("ascend_scatter_add")
class ScatterAdd(BaseApi):

    def __init__(self, task_result: TaskResult):
        super(ScatterAdd, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        varRef = input_data.kwargs["varRef"]
        indices = input_data.kwargs["indices"]
        updates = input_data.kwargs["updates"]

        indices_len = len(indices.shape)
        indices_new = indices.reshape(-1)
        updates_new_shape = (-1, *updates.shape[indices_len:])
        updates_new = updates.reshape(updates_new_shape)

        varRef_dtype = varRef.dtype
        if varRef_dtype == torch.float32:
            out = varRef.index_add(0, indices_new, updates_new)
            return out
        elif varRef_dtype == torch.bfloat16:
            varRef = varRef.to(torch.float32)
            updates_new = updates_new.to(torch.float32)
            out = varRef.index_add(0, indices_new, updates_new).to(torch.bfloat16)
            return out
        else :
            varRef_np = varRef.cpu().numpy()
            indices_np = indices_new.cpu().numpy()
            updates_np = updates_new.cpu().numpy()
            for i, index in enumerate(indices_np):
                varRef_np[index] += updates_np[i]
            return torch.from_numpy(varRef_np)
           

    
@register("pyaclnn_scatter_add")
class ScatterNdUpdatePyaclnn(AclnnBaseApi):

    def init_by_input_data(self, input_data): 

        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages



