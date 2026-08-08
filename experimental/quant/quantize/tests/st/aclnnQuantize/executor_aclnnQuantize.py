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

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import AccuracyConfig, TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare

# ascend910b (DAV_2201) quantize output dtypes: INT8 / UINT8 / INT32 (no fp8 on this SoC).
_QDTYPE_MAP = {
    torch.int8: torch.qint8,
    torch.int32: torch.qint32,
    torch.uint8: torch.quint8,
}
_OUT_NP_DTYPE = {
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
}


@register("function_quantize")
class FunctionQuantizeApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionQuantizeApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # CPU golden: y = saturating_cast_to(dtype)( rint( x / scales + zero_points ) ).
        # torch.quantize_per_channel computes round(x/scale) + zero_point then saturates,
        # which is identical to rint(x/scale + zero_point) for integer zero_points and
        # stays within the stand_quantize tolerance (|diff| <= 1) for bf16 zero_points.
        x_value = input_data.kwargs["x"]
        x_dim = len(x_value.shape)
        axis = input_data.kwargs["axis"]
        scales = input_data.kwargs["scales"]
        # GUARD: zeroPoints is an optional input; when absent/None the op treats it as 0.
        zero_points = input_data.kwargs.get("zeroPoints", None)
        dtype = input_data.kwargs["dtype"]

        if axis < 0:
            axis = axis + x_dim

        # Empty tensor (any 0-sized dim): workspaceSize=0, no compute. Produce an empty
        # output with the requested integer dtype/shape; torch.quantize_per_channel cannot
        # run on an empty channel axis, so handle it directly.
        if 0 in tuple(x_value.shape):
            return torch.from_numpy(
                np.empty(tuple(x_value.shape), dtype=_OUT_NP_DTYPE[dtype])
            )

        x_value = x_value.to(torch.float32)
        scales = scales.to(torch.float32)
        qdtype = _QDTYPE_MAP.get(dtype)

        # zeroPoints absent => offset 0 (match scales element count for per-channel repeat below).
        if zero_points is None:
            zero_points = torch.zeros(
                scales.shape if scales.dim() > 0 else (1,), dtype=torch.int64
            )

        if len(scales.shape) == 1 and scales.shape[0] == 1:
            repeat_num = x_value.shape[axis]
            if repeat_num != 1:
                scales = scales.repeat(repeat_num)
                zero_points = zero_points.repeat(repeat_num)
        output = torch.quantize_per_channel(
            x_value, scales, zero_points, axis, qdtype
        ).int_repr()
        return output


@ACCURACY_REGISTRY.register("stand_quantize")
class SingleBenchmarkAccuracyCompare(BaseAccuracyCompare):
    @staticmethod
    def compute_quantize_accuracy(local_output, remote_output, data_file):
        # Empty tensor: no elements to compare. Empty golden vs empty NPU output is an
        # exact match by definition; torch.max over a 0-size dim would otherwise raise.
        if local_output.numel() == 0 and remote_output.numel() == 0:
            return AccuracyConfig(
                result=True,
                filename=data_file,
                max_diff=0,
                max_diff_idx=0,
            )

        diff_value = torch.subtract(local_output.to(torch.int64), remote_output)
        diff_abs = torch.abs(diff_value)

        flat_diff_abs = diff_abs.view(-1)
        max_diff, max_diff_idx = torch.max(flat_diff_abs, dim=0)
        result = torch.all(diff_abs <= 1)

        acc_result = AccuracyConfig(
            result=result.item(),
            filename=data_file,
            max_diff=max_diff.item(),
            max_diff_idx=max_diff_idx.item(),
        )
        return acc_result

    def compute_accuracy_result(self, local_output, remote_output, data_file):
        acc_ret = self.compute_quantize_accuracy(local_output, remote_output, data_file)
        return acc_ret
