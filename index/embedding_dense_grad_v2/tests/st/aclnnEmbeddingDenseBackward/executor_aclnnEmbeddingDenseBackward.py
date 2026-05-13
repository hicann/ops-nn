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
import numpy as np

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_embedding_dense_backward")
class AsceneEmbeddingDenseBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AsceneEmbeddingDenseBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        if self.device == "cpu":
            self.data_device = torch.device("cpu")
        elif self.device == "pyaclnn":
            self.data_device = torch.device(f"npu:0")

    def init_by_input_data(self, input_data: InputDataset):

        indices_new = torch.from_numpy(np.random.randint(0, input_data.kwargs["numWeights"],
                                                                          size=input_data.kwargs["indices"].shape))
        if self.device == "pyaclnn":
            import torch_npu
            input_data.kwargs["indices"] = indices_new.npu()
        else:
            input_data.kwargs["indices"] = indices_new.to(self.data_device)
       
    def __call__(self, input_data: InputDataset, with_output: bool = False):

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        self.grad = input_data.kwargs["grad"].to(device)
        self.indices = input_data.kwargs["indices"].to(device)
        output = torch.ops.aten.embedding_dense_backward(self.grad, self.indices, input_data.kwargs["numWeights"],
                                                         input_data.kwargs["paddingIdx"],
                                                         input_data.kwargs["scaleGradByFreq"])

        return output