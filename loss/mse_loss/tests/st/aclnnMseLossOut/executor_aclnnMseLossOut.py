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
import torch.nn
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("executor_mse_loss_out")
class FunctionApi(BaseApi):
    '''
    aclnnMseLossOut和aclnnMseLoss区别:
    (1) aclnnMseLoss直接调l0接口且传对应的reduction模式计算mseloss
    (2) aclnnMseLossOut调l0接口固定传reduction=0计算mseloss,再根据输入真实reduction做reduce处理
    (3) reduce为0时两者结果的shape均为self和target广播后的shape
    (4) reduce为1或2时aclnnMseLoss为0维,aclnnMseLossOut仅reduce了0轴
    '''
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        input_self = input_data.kwargs['self']
        input_target = input_data.kwargs['target']
        reduction = input_data.kwargs['reduction']

        # torch原生接口同样支持混合精度,(f16,f16)(f32,f32)(bf16,f32)(bf16,f16)
        # 任意混合精度时均会升到f32,和aclnn接口精度转换一致
        # 算子不支持bf16,因此同时bf16会提示["mse_cpu" not implemented for 'BFloat16']
        loss = torch.nn.MSELoss(reduction = "none")
        output = loss(input_self, input_target)

        if reduction == 2:
            return output.sum(dim=0)
        elif reduction == 1:
            return output.mean(dim=0)
        else:
            return output

