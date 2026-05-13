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


@register("ascend_function_expand_into_jagged_permute")
class FunctionExpandIntoJaggedPermuteApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionExpandIntoJaggedPermuteApi, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        print(f'output_offsets:{input_data.kwargs["output_offsets"]}')
        print(f'permute:{input_data.kwargs["permute"]}')
        OpsDataset.seed_everything()

        def generate_strictly_increasing_tensor_torch(n, k):
            if k < 2 or n < k - 1:
                raise ValueError("无效的n或k")
            # 生成1到n-1的随机排列，取前k-2个并排序
            random_points = torch.randperm(n - 1, dtype=torch.int32)[:k - 2] + 1
            random_points, _ = torch.sort(random_points)
            # 加入首尾
            tensor = torch.cat([torch.tensor([0], dtype=torch.int32),
                                random_points,
                                torch.tensor([n], dtype=torch.int32)])
            return tensor

        n = input_data.kwargs['output_size']
        k = input_data.kwargs['output_offsets'].shape[0]
        if k == 23:
            input_data.kwargs['permute'] = torch.tensor(
                [0, 11, 1, 12, 2, 13, 3, 14, 4, 15, 5, 16, 6, 17, 7, 18, 8, 19, 9, 20, 10, 21]).to(
                input_data.kwargs['permute'].device).to(input_data.kwargs['permute'].dtype)
            input_data.kwargs["input_offsets"] = torch.tensor(
                [0, 4860, 9720, 14580, 19440, 24300, 29160, 34020, 38880, 43740, 48600, 53460, 58899, 64338, 69777,
                 75216, 80655, 86094, 91533, 96972, 102411, 107850, 113289]).to(
                input_data.kwargs['input_offsets'].device).to(input_data.kwargs['input_offsets'].dtype)
            input_data.kwargs["output_offsets"] = torch.tensor(
                [0, 4860, 10299, 15159, 20598, 25458, 30897, 35757, 41196, 46056, 51495, 56355, 61794, 66654, 72093,
                 76953, 82392, 87252, 92691, 97551, 102990, 107850, 113289]).to(
                device=input_data.kwargs['output_offsets'].device).to(dtype=input_data.kwargs['output_offsets'].dtype)
            input_data.kwargs["output_size"] = 113289
        else:
            input_data.kwargs["output_offsets"] = generate_strictly_increasing_tensor_torch(n, k).to(
                device=input_data.kwargs['output_offsets'].device).to(dtype=input_data.kwargs['output_offsets'].dtype)
            input_data.kwargs['permute'] = torch.randperm(k - 1).to(
                input_data.kwargs['permute'].device).to(input_data.kwargs['permute'].dtype)
        print(f'output_offsets:{input_data.kwargs["output_offsets"]}')
        print(f'permute:{input_data.kwargs["permute"]}')
    def expand_into_jagged_permute(self, permute, input_offsets, output_offsets, output_size):
        output2 = torch.empty(output_size, dtype=permute.dtype, device=permute.device)
        leni = output_offsets[1:] - output_offsets[:-1]
        permute_length = permute.shape[0]
        st = input_offsets[permute]
        stl = st + leni
        for i in range(permute_length):
            output2[output_offsets[i]:output_offsets[i + 1]] = torch.arange(st[i], stl[i], dtype=permute.dtype,
                                                                            device=output_offsets.device)
            output2.contiguous()
        return output2.to(dtype=permute.dtype)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        permute = input_data.kwargs["permute"].to(torch.int32)
        input_offsets = input_data.kwargs["input_offsets"]
        output_offsets = input_data.kwargs["output_offsets"]
        output_size = input_data.kwargs["output_size"]
        cpu_res = self.expand_into_jagged_permute(permute, input_offsets, output_offsets, output_size)

        return cpu_res
