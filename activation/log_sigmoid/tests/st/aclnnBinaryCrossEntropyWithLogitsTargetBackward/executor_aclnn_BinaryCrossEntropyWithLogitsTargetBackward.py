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
from torch.nn import functional as F

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("BinaryCrossEntropyWithLogitsTargetBackward")
class AclnnBinaryCrossEntropyWithLogitsTargetBackwardApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        reduction_map = {
            0: "none",
            1: "mean",
            2: "sum",
        }

        gradOut = input_data.kwargs["gradOut"]
        input = input_data.kwargs["self"]
        target = input_data.kwargs["target"]
        weight = input_data.kwargs["weight"]
        postWeight = input_data.kwargs["postWeight"]
        reduction = reduction_map[input_data.kwargs["reduction"]]
        output = None
        org_dtype = gradOut.dtype

        if self.device == "cpu":
            dtype = torch.float
            input = input.to(dtype)
            target = target.to(dtype)
            weight = weight.to(dtype)
            postWeight = postWeight.to(dtype)
            gradOut = gradOut.to(dtype)
            target.requires_grad = True
            loss = F.binary_cross_entropy_with_logits(
                input, target, weight=weight, pos_weight=postWeight, reduction=reduction
            )
            loss.backward(gradOut.reshape(loss.shape))
            output = target.grad.to(org_dtype)

        return output
