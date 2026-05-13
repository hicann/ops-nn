#!/usr/bin/env python3
# -- coding: utf-8 --
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
import numpy as np

@register("function_aclnn_ctc_loss_backward")
class CtcLossBackward(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradOut = input_data.kwargs["gradOutput"]
        logProbs = input_data.kwargs["logProbs"]
        targets = input_data.kwargs["targets"]
        inputLengths = input_data.kwargs["inputLengths"]
        targetLengths = input_data.kwargs["targetLengths"]
        negLogLikelihood = input_data.kwargs["negLogLikelihood"]
        logAlpha = input_data.kwargs["logAlpha"]
        blank = input_data.kwargs["blank"]
        zeroInfinity = input_data.kwargs["zeroInfinity"]
        output = torch.ops.aten._ctc_loss_backward(gradOut, logProbs,
                    targets.type(torch.int64), inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity)
        return output

