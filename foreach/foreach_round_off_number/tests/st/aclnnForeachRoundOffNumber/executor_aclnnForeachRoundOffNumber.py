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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("function_aclnnForeachRoundOffNumber")
class aclnnForeachRoundOffNumberExecutor(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        roundMode = input_data.kwargs["roundMode"][0].item()

        ROUND_MODE_FLOOR = 2
        ROUND_MODE_CEIL = 3
        ROUND_MODE_ROUND = 1
        ROUND_MODE_TRUNC = 5
        ROUND_MODE_FRAC = 7

        if roundMode == ROUND_MODE_FLOOR:
            output = torch._foreach_floor(x)
        elif roundMode == ROUND_MODE_CEIL:
            output = torch._foreach_ceil(x)
        elif roundMode == ROUND_MODE_ROUND:
            output = torch._foreach_round(x)
        elif roundMode == ROUND_MODE_TRUNC:
            output = torch._foreach_trunc(x)
        elif roundMode == ROUND_MODE_FRAC:
            if x[0].dtype in [
                torch.int16,
            ]:
                output = [torch.zeros_like(t) for t in x]
            else:
                output = torch._foreach_frac(x)
        return output


@register("aclnnfunction_aclnnForeachRoundOffNumber")
class aclnnfunctionExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.task_result.output_info_list = [self.task_result.output_info_list]
        return super().init_by_input_data(input_data)
