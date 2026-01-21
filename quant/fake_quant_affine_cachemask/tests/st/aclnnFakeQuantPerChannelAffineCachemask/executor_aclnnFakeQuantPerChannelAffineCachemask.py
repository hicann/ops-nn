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
from atk.configs.base_config import TaskType


@register("aclnn_fake_quant_per_channel_affine_cachemask")
class MethodTorchNnFakeQuantAffineCachemaskApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchNnFakeQuantAffineCachemaskApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        # device处理 pyaclnn仅在此阶段区分 后续调用仍走cpu获取输出格式，此处转device仅构造输入
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device in ("npu", "pyaclnn"):
            device = f"npu:{self.device_id}"
        else:
            device = "cpu"
        
        if self.device == "cpu":
            # 标杆的scale仅支持fp32
            input_data.kwargs['scale'] = input_data.kwargs['scale'].to(torch.float32)
        input_data.kwargs['self'] = input_data.kwargs['self'].to(device)
        input_data.kwargs['scale'] = input_data.kwargs['scale'].to(device)
        input_data.kwargs['zeroPoint'] = input_data.kwargs['zeroPoint'].to(device)
        
    def __call__(self, input_data: InputDataset, with_output: bool = False):

        func = eval(self.api_name)
        output = mask = None
        if with_output:
            output, mask = func(input_data.kwargs['self'],
                                input_data.kwargs['scale'],
                                input_data.kwargs['zeroPoint'],
                                input_data.kwargs['axis'],
                                input_data.kwargs['quantMin'],
                                input_data.kwargs['quantMax'])
        else:
            func(input_data.kwargs['self'],
                 input_data.kwargs['scale'],
                 input_data.kwargs['zeroPoint'],
                 input_data.kwargs['axis'],
                 input_data.kwargs['quantMin'],
                 input_data.kwargs['quantMax'])

        return output, mask