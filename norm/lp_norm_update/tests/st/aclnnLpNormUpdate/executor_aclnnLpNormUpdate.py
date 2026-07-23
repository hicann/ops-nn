#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import sys

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
from golden import lp_norm_update_golden


@register("aclnn_lp_norm_update")
class LpNormUpdateApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        pass

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        p = input_data.kwargs["p"]
        epsilon = input_data.kwargs["epsilon"]

        if self.device == "cpu":
            output = lp_norm_update_golden(x, p, epsilon)
            return output
        else:
            raise NotImplementedError(
                "NPU execution via GEIR graph mode is not supported in ATK executor. "
                "Use test_geir_lp_norm_update.cpp for NPU verification."
            )
