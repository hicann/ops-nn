#!/usr/bin/env python3
# -- coding: utf-8 --
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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_method_aclnnChamferDistanceBackward")
class ChamferDistanceBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(ChamferDistanceBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        xyz1 = input_data.kwargs["xyz1"]

        xyz2 = input_data.kwargs["xyz2"]

        idx1 = input_data.kwargs["idx1"]

        idx2 = input_data.kwargs["idx2"]

        gradDist1 = input_data.kwargs["gradDist1"]
        gradDist2 = input_data.kwargs["gradDist2"]
        gradDist1_shape = gradDist1.shape


        batch = gradDist1_shape[0]
        num = gradDist1_shape[1]

        master_dtype = gradDist1.dtype
        grad_xyz1_cpu = torch.zeros(batch, num, 2, dtype=master_dtype)
        grad_xyz2_cpu = torch.zeros(batch, num, 2, dtype=master_dtype)

        for b in range(batch):
            for n in range(num):
                cur_ind = idx1[b][n]

                x1, y1 = xyz1[b][n]

                x2, y2 = xyz2[b][cur_ind]

                g1 = gradDist1[b][n] * 2

                grad_xyz1_cpu[b][n][0] += (x1 - x2) * g1

                grad_xyz1_cpu[b][n][1] += (y1 - y2) * g1
                grad_xyz2_cpu[b][cur_ind][0] -= (x1 - x2) * g1
                grad_xyz2_cpu[b][cur_ind][1] -= (y1 - y2) * g1

        for b in range(batch):
            for n in range(num):
                cur_ind = idx2[b][n]
                x1, y1 = xyz2[b][n]
                x2, y2 = xyz1[b][cur_ind]
                g2 = gradDist2[b][n] * 2
                grad_xyz2_cpu[b][n][0] += (x1 - x2) * g2
                grad_xyz2_cpu[b][n][1] += (y1 - y2) * g2
                grad_xyz1_cpu[b][cur_ind][0] -= (x1 - x2) * g2
                grad_xyz1_cpu[b][cur_ind][1] -= (y1 - y2) * g2

        return grad_xyz1_cpu, grad_xyz2_cpu