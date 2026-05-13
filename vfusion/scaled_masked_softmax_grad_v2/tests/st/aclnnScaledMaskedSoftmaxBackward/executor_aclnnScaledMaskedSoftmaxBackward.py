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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_function_scaled_masked_softmax_backward")
class AscendFunctionScaledMaskedSoftmaxBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AscendFunctionScaledMaskedSoftmaxBackwardApi, self).__init__(task_result)

    def reduce_sum(input):
        block_size = 64
        shape = input.shape
        line_num = shape[0] * shape[1] * shape[2]
        one_line_len = int(shape[3])

        if one_line_len <= 1024:
            block_size = 8
            align_line_len = (one_line_len + 63) // 64 * 64
            pad_num = align_line_len - one_line_len
            input = torch.cat([input.reshape(line_num, one_line_len),
                            torch.zeros((line_num, pad_num), device=input.device)], dim = 1)
            one_line_len = align_line_len
        else:
            input = input.reshape(line_num, one_line_len)

        blocks = (one_line_len + block_size - 1) // block_size
        output = torch.zeros((line_num, blocks), device=input.device)

        for i in range(blocks):
            start = i * block_size
            end = min((i + 1) * block_size, one_line_len)
            block_data = input[:, start:end]

            while block_data.size(1) > 1:
                if block_data.size(1) % 2 != 0:
                    block_data = torch.cat([block_data,
                                        torch.zeros((line_num, 1), device=block_data.device)], dim=1)
                block_data = block_data.reshape(line_num, -1, 2).sum(dim=2)
            output[:, i] = block_data.squeeze(1)
        if one_line_len <= 1024:
            add_repeats = blocks // block_size
            for i in range(1, add_repeats):
                output[:, :8] += output[:, i*8 : (i+1)*8]
            final_output = output[:, :8]
        else:
            final_output = output
        
        # 全局二叉累加
        while final_output.size(1) > 1:
            if final_output.size(1) % 2 != 0:
                final_output = torch.cat([final_output,
                                        torch.zeros((line_num, 1), device=final_output.device)], dim=1)
            final_output = final_output.reshape(line_num, -1, 2).sum(dim=2)
        
        return final_output.reshape(shape[0], shape[1], shape[2], 1)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        y_grad = input_data.kwargs["y_grad"]
        y = input_data.kwargs["y"]
        mask = input_data.kwargs["mask"]
        scale = input_data.kwargs["scale"]
        fixed_triu_mask = input_data.kwargs["fixed_triu_mask"]
        dtype = y_grad.dtype

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        y_grad = y_grad.to(device).float()
        y = y.to(device).float()
        mask = mask.to(device)

        x_grad = y_grad * y
        tmp_sum = AscendFunctionScaledMaskedSoftmaxBackwardApi.reduce_sum(x_grad)
        x_grad = x_grad - tmp_sum * y
        x_grad = x_grad * scale
        x_grad = x_grad.masked_fill(mask, value = 0)

        return x_grad.to(dtype)

@register("aclnn_scaled_masked_softmax_backward")
class AclnnScaledMaskedSoftmaxBackward(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages
