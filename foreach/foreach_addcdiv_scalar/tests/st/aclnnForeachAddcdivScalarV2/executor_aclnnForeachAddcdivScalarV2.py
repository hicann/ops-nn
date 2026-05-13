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
import numpy as np


@register("function_aclnnForeachAddcdivScalarV2")
class aclnnForeachAddcdivScalarV2Executor(BaseApi):

    def converter(self, func, *args, **kwargs):
        def _convert(data, dtype):
            if isinstance(data, torch.Tensor):
                return data.to(dtype)
            elif isinstance(data, (list, tuple)):
                return type(data)(_convert(x, dtype) for x in data)
            return data

        def _has_float16(data):
            if isinstance(data, torch.Tensor):
                return data.dtype == torch.float16
            elif isinstance(data, (list, tuple)):
                return any(_has_float16(x) for x in data)
            return False

        if any(_has_float16(x) for x in args) or any(_has_float16(v) for v in kwargs.values()):
            promoted_args = [_convert(x, torch.float32) for x in args]
            promoted_kwargs = {k: _convert(v, torch.float32) for k, v in kwargs.items()}
            result = func(*promoted_args, **promoted_kwargs)
            return _convert(result, torch.float16)
        return func(*args, **kwargs)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        x3 = input_data.kwargs["x3"]
        scalar = input_data.kwargs["scalar"]

        output = None
        if self.device == 'cpu':
            output = self.converter(torch._foreach_addcdiv, x1, x2, x3, scalar)
        else:
            output = torch._foreach_addcdiv(x1, x2, x3, scalar)

        return output

@register("aclnnfunction_aclnnForeachAddcdivScalarV2")
class aclnnfunctionExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.task_result.output_info_list = [self.task_result.output_info_list]
        return super().init_by_input_data(input_data)