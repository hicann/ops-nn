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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_expand_into_jagged_permute")
class FunctionExpandIntoJaggedPermuteApi(BaseApi):
    """Generate legal inputs for the ACLNN standard.acc positive suite.

    Expected-error L2 cases use their own assets and do not enter this executor.
    """

    def __init__(self, task_result: TaskResult):
        super(FunctionExpandIntoJaggedPermuteApi, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        print(f"output_offsets:{input_data.kwargs['output_offsets']}")
        print(f"permute:{input_data.kwargs['permute']}")
        OpsDataset.seed_everything()

        def generate_strictly_increasing_tensor_torch(n, k):
            if k < 2 or n < k - 1:
                raise ValueError("无效的n或k")
            # 生成1到n-1的随机排列，取前k-2个并排序
            random_points = torch.randperm(n - 1, dtype=torch.int32)[: k - 2] + 1
            random_points, _ = torch.sort(random_points)
            # 加入首尾
            tensor = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int32),
                    random_points,
                    torch.tensor([n], dtype=torch.int32),
                ]
            )
            return tensor

        k = input_data.kwargs["output_offsets"].shape[0]
        n = max(int(input_data.kwargs["output_size"]), k - 1)
        if k == 1:
            input_data.kwargs["output_size"] = 0
            input_data.kwargs["output_offsets"] = torch.zeros(
                1,
                dtype=input_data.kwargs["output_offsets"].dtype,
                device=input_data.kwargs["output_offsets"].device,
            )
            input_data.kwargs["input_offsets"] = torch.zeros(
                1,
                dtype=input_data.kwargs["input_offsets"].dtype,
                device=input_data.kwargs["input_offsets"].device,
            )
            input_data.kwargs["permute"] = torch.empty(
                0,
                dtype=input_data.kwargs["permute"].dtype,
                device=input_data.kwargs["permute"].device,
            )
        else:
            input_data.kwargs["output_size"] = n
            output_offsets = generate_strictly_increasing_tensor_torch(n, k).to(
                device=input_data.kwargs["output_offsets"].device,
                dtype=input_data.kwargs["output_offsets"].dtype,
            )
            permute = torch.arange(
                k - 1,
                device=input_data.kwargs["permute"].device,
                dtype=input_data.kwargs["permute"].dtype,
            )
            if k > 2:
                permute = permute.roll(1)
            target_lengths = output_offsets[1:] - output_offsets[:-1]
            source_lengths = torch.empty_like(target_lengths)
            source_lengths[permute] = target_lengths
            input_offsets = torch.cat(
                [
                    torch.zeros(
                        1,
                        dtype=input_data.kwargs["input_offsets"].dtype,
                        device=input_data.kwargs["input_offsets"].device,
                    ),
                    torch.cumsum(source_lengths, dim=0).to(
                        device=input_data.kwargs["input_offsets"].device,
                        dtype=input_data.kwargs["input_offsets"].dtype,
                    ),
                ]
            )
            input_data.kwargs["output_offsets"] = output_offsets
            input_data.kwargs["input_offsets"] = input_offsets
            input_data.kwargs["permute"] = permute
        input_offsets = input_data.kwargs["input_offsets"]
        output_offsets = input_data.kwargs["output_offsets"]
        permute = input_data.kwargs["permute"]
        if (
            input_offsets.ndim != 1
            or output_offsets.ndim != 1
            or permute.ndim != 1
            or input_offsets.shape != output_offsets.shape
            or input_offsets.shape[0] != permute.shape[0] + 1
            or input_offsets.numel() == 0
            or int(input_offsets[0]) != 0
            or int(output_offsets[0]) != 0
            or not torch.all(input_offsets[1:] > input_offsets[:-1])
            or not torch.all(output_offsets[1:] > output_offsets[:-1])
            or int(output_offsets[-1]) != int(input_data.kwargs["output_size"])
            or not torch.all((permute >= 0) & (permute < permute.shape[0]))
        ):
            raise ValueError(
                "generated ACLNN inputs violate ExpandIntoJaggedPermute constraints"
            )
        print(f"output_offsets:{input_data.kwargs['output_offsets']}")
        print(f"permute:{input_data.kwargs['permute']}")

    def expand_into_jagged_permute(
        self, permute, input_offsets, output_offsets, output_size
    ):
        output2 = torch.empty(output_size, dtype=permute.dtype, device=permute.device)
        leni = output_offsets[1:] - output_offsets[:-1]
        permute_length = permute.shape[0]
        st = input_offsets[permute]
        stl = st + leni
        for i in range(permute_length):
            output2[output_offsets[i] : output_offsets[i + 1]] = torch.arange(
                st[i], stl[i], dtype=permute.dtype, device=output_offsets.device
            )
            output2.contiguous()
        return output2.to(dtype=permute.dtype)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        permute = input_data.kwargs["permute"].to(torch.int32)
        input_offsets = input_data.kwargs["input_offsets"]
        output_offsets = input_data.kwargs["output_offsets"]
        output_size = input_data.kwargs["output_size"]
        cpu_res = self.expand_into_jagged_permute(
            permute, input_offsets, output_offsets, output_size
        )

        return cpu_res
