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
#

import random
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("function_quantize")
class FunctionQuantizeApi(BaseApi):

    def __init__(self, task_result: TaskResult):
        super(FunctionQuantizeApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        x_value = input_data.kwargs["x"]
        x_dim = len(x_value.shape)
        axis = input_data.kwargs["axis"]
        scales = input_data.kwargs["scales"]
        zero_points = input_data.kwargs["zeroPoints"]
        dtype = input_data.kwargs["dtype"]

        x_value = x_value.to(torch.float32)
        scales = scales.to(torch.float32)
        if axis < 0:
            axis = axis + x_dim
        dtype_map = {
            torch.int8: torch.qint8,
            torch.int32: torch.qint32,
            torch.uint8: torch.quint8
        }
        dtype = dtype_map.get(dtype)
        if len(scales.shape) == 1 and scales.shape[0] == 1:
            repeat_num = x_value.shape[axis]
            if repeat_num != 1:
                scales = scales.repeat(repeat_num)
                zero_points = zero_points.repeat(repeat_num)
        output = torch.quantize_per_channel(x_value, scales, zero_points, axis, dtype).int_repr()
        return output
    
from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare

@ACCURACY_REGISTRY.register("stand_quantize")
class SingleBenchmarkAccuracyCompare(BaseAccuracyCompare):
    @staticmethod
    def compute_quantize_accuracy(local_output, remote_output, data_file):
        diff_value = torch.subtract(local_output.to(torch.int64), remote_output)
        diff_abs = torch.abs(diff_value)

        flat_diff_abs = diff_abs.view(-1)
        max_diff, max_diff_idx = torch.max(flat_diff_abs, dim = 0)
        result = torch.all(diff_abs <= 1)

        acc_result = AccuracyConfig(
            result = result.item(),
            filename = data_file,
            max_diff = max_diff.item(),
            max_diff_idx = max_diff_idx.item(),
        )
        return acc_result
    def compute_accuracy_result(self, local_output, remote_output, data_file):
        acc_ret = self.compute_quantize_accuracy(local_output, remote_output, data_file)
        return acc_ret