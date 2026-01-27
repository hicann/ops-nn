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
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

@register("aclnn_silu_backward")
class AscendSiluBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AscendSiluBackwardApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradInput = torch.ops.aten.silu_backward(
            input_data.kwargs["gradOutput"],
            input_data.kwargs["self"]
        )
        return gradInput

    def get_format(self, input_data: InputDataset, index=None, name=None):
            """
            :param input_data: 参数列表
            :param index: 参数位置
            :param name: 参数名字
            :return:
            format at this index or name
            """
            return AclFormat.ACL_FORMAT_ND