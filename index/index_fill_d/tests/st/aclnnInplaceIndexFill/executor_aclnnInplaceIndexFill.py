#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi


@register("function_aclnnInplaceIndexFill")
class FunctionInplaceIndexFillApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        x = input_data.kwargs["self"]
        return x.index_fill_(
            input_data.kwargs["dim"],
            input_data.kwargs["index"].to(torch.int64),
            input_data.kwargs["value"],
        )


@register("pyaclnn_inplace_index_fill")
class InplaceIndexFillAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        input_args, _ = super().init_by_input_data(input_data)
        # The result aliases self; the C API has no separate output tensor argument.
        return input_args[:-1], [input_args[0]]
