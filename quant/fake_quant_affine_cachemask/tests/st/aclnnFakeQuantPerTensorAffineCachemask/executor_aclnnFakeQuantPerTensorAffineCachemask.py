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


@register("aclnn_fake_quant_per_tensor_affine_cachemask")
class MethodTorchNnFakeQuantAffineCachemaskApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchNnFakeQuantAffineCachemaskApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        func = eval(self.api_name)

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
        input_data.kwargs["self"] = input_data.kwargs["self"].to(device)
        input_data.kwargs["scale"] = input_data.kwargs["scale"].to(device)
        input_data.kwargs["zeroPoint"] = input_data.kwargs["zeroPoint"].to(device)
        # 按文档语义：fakeQuantEnabled >= 1 启用伪量化，小于 1 时输出 identity；
        # 先转 bool 会使 0<值<1 被误判为启用
        fqe = input_data.kwargs["fakeQuantEnbled"]
        if not torch.is_tensor(fqe):
            fqe = torch.tensor(fqe)
        input_data.kwargs["fakeQuantEnbled"] = fqe.to(device)

        output = mask = None
        if with_output:
            output, mask = func(
                input_data.kwargs["self"],
                input_data.kwargs["scale"],
                input_data.kwargs["zeroPoint"],
                input_data.kwargs["fakeQuantEnbled"],
                input_data.kwargs["quantMin"],
                input_data.kwargs["quantMax"],
            )
        else:
            func(
                input_data.kwargs["self"],
                input_data.kwargs["scale"],
                input_data.kwargs["zeroPoint"],
                input_data.kwargs["fakeQuantEnbled"],
                input_data.kwargs["quantMin"],
                input_data.kwargs["quantMax"],
            )

        return output, mask
