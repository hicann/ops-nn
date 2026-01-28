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
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_group_norm_backward")
class AsceneGroupNormBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AsceneGroupNormBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def init_by_input_data(self, input_data: InputDataset):

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
 
        n_of_grad_out = input_data.kwargs["N"]
        c_of_grad_out = input_data.kwargs["C"]
        hxw_of_grad_out = input_data.kwargs["HxW"]
        group = input_data.kwargs["group"]

        # 正向输入的beta
        beta = torch.rand(c_of_grad_out).to(input_data.kwargs["input"].dtype).to(device)
        input_data.kwargs["input"] = input_data.kwargs["input"].to(device)
        input_data.kwargs["gamma"] = input_data.kwargs["gamma"].to(device)
        out, mean_out, rstd_out = torch.ops.aten.native_group_norm(input_data.kwargs["input"],
                                                                 input_data.kwargs["gamma"],
                                                                 beta, n_of_grad_out, c_of_grad_out, hxw_of_grad_out,
                                                                 group, 1e-5)

        input_data.kwargs["mean"] = mean_out.to(device)
        input_data.kwargs["rstd"] = rstd_out.to(device)

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            if self.device_id == 0:
                torch.npu.set_compile_mode(jit_compile=False)
            elif self.device_id == 1:
                torch.npu.set_compile_mode(jit_compile=True)
        else:
            device = "cpu"

        # #开启确定性计算
        torch.use_deterministic_algorithms(True)
        input_data.kwargs["gradOut"] = input_data.kwargs["gradOut"].to(device)
        output = torch.ops.aten.native_group_norm_backward(input_data.kwargs["gradOut"], input_data.kwargs["input"],
                                                           input_data.kwargs["mean"], input_data.kwargs["rstd"],
                                                           input_data.kwargs["gamma"], input_data.kwargs["N"],
                                                           input_data.kwargs["C"],
                                                           input_data.kwargs["HxW"], input_data.kwargs["group"],
                                                           input_data.kwargs["outputMask"])
        return output[0], output[1], output[2]