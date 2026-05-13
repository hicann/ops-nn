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


@register("function_einsum")
class FunctionEinsumApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionEinsumApi, self).__init__(task_result)
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self.change_flag = False
        # 官方的场景在以下场景测试时， 会调用torch.bmm 该接口不支持fp16， 统一转成pf32， 结果在切回fp16
        if input_data.kwargs["equation"] == "bhid,bhijd->bhij" and self.device != "npu":
            input_data.args = self.change_data_dtype(
                input_data.args, torch.float16, torch.float32
            )
            input_data.kwargs = self.change_data_dtype(
                input_data.kwargs, torch.float16, torch.float32
            )

        if not with_output:
            output = eval(self.api_name)(input_data.kwargs["equation"], input_data.kwargs["tensors"])
            return output
        if self.output is None:
            output = eval(self.api_name)(
                input_data.kwargs["equation"], input_data.kwargs["tensors"]
            )
        else:
            eval(self.api_name)(input_data.kwargs["equation"], input_data.kwargs["tensors"])
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        if self.change_flag:
            output = self.change_data_dtype(
                output, torch.float32, torch.float16
            )
        return output