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
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_scatter_for_aclnn_scatter")
class FunctionScatterApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        torch.use_deterministic_algorithms(True)
        if "self" in input_data.kwargs.keys():
            input_data.kwargs["input"] = input_data.kwargs["self"]
            del input_data.kwargs["self"]
        if input_data.kwargs["reduce"] == 0:
            del input_data.kwargs["reduce"]
        elif input_data.kwargs["reduce"] == 1:
            input_data.kwargs["reduce"] = "add"
        if not with_output:
            output = eval(self.api_name)(*input_data.args, **input_data.kwargs)
            if "reduce" not in input_data.kwargs.keys():
                input_data.kwargs["reduce"] = 0
            return output
        if self.output is None:
            output = eval(self.api_name)(
                *input_data.args, **input_data.kwargs
            )
        else:
            eval(self.api_name)(*input_data.args, **input_data.kwargs)
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        if "reduce" not in input_data.kwargs.keys():
            input_data.kwargs["reduce"] = 0
        return output
