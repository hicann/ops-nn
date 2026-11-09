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


@register("function_aclnnIndexAddV2")
class FunctionIndexAddApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionIndexAddApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # torch 参考路径（cpu 标杆 / npu 派发链回归共用）：mode 仅取 0/非 0 两态；
        # mode=0 时 V2 语义 alpha 不参与计算，torch 参考按 alpha=1 折算。
        x = input_data.kwargs["self"]
        dim = input_data.kwargs["dim"]
        index = input_data.kwargs["index"].to(x.device)
        source = input_data.kwargs["source"]
        mode = input_data.kwargs["mode"]
        if mode == 0:
            alpha_value = 1
        else:
            alpha_value = input_data.kwargs["alpha"]
        output = torch.index_add(x, dim, index, source, alpha=alpha_value)
        return output

    def init_by_input_data(self, input_data: InputDataset):
        # pyaclnn 直达路径编组前钩子：mode=0 时 alpha 置 1（V2 语义 alpha 不参与计算）。
        mode = input_data.kwargs["mode"]
        if mode == 0:
            input_data.kwargs["alpha"] = 1

    def get_cpp_func_signature_type(self):
        # aclnnIndexAddV2 直达调用签名声明：mode 为 V2 独有入参，经用例 kwargs 编组直达 V2 入口，
        # 使 ST 用例真实覆盖 V2 mode=0（AICore 路径）/ mode≠0（AICPU+alpha 路径）分支。
        # 框架按本签名从 input_data.kwargs 依参数名编组：tensor→aclTensor、attr int→int64_t、
        # scalar→aclScalar；输出 out 由框架按 output_info 分配（shape/dtype 与 self 一致）。
        # 符号名由用例 aclnn_name（"IndexAddV2"）派生：aclnnIndexAddV2{GetWorkspaceSize,}。
        return (
            "aclnnStatus aclnnIndexAddV2GetWorkspaceSize(const aclTensor* self, const int64_t dim, "
            "const aclTensor* index, const aclTensor* source, const aclScalar* alpha, int64_t mode, "
            "aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)"
        )
