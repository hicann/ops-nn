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
#

import logging
import torch
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import OutputData
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.common.utils import get_output_data_infos
import numpy as np

try:
    import torch_npu
    import torchvision
except Exception as e:
    logging.warning("import torch_npu failed!!!")

@register("function_aclnnFlatQuant")
class FunctionaclnnFlatQuant(BaseApi):
    # 精度对比标杆
    def get_golden(self, x, p1, p2, clipRatio):
        x = x.to(torch.float)
        p1 = p1.to(torch.float)
        p2 = p2.to(torch.float)
        if (clipRatio is None):
            clipRatio = 1.0

        x_shape = x.shape
        x1 = torch.matmul(x, p2)
        x2 = torch.matmul(p1, x1).to(torch.half)
        x2 = x2.flatten(-2, -1)
        qscale = torch.abs(x2).max(dim=-1, keepdim=True)[0].to(torch.float)
        ratio = torch.ones_like(qscale) * 7 / clipRatio
        return torch.flatten(qscale / ratio)

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == 'pyaclnn':
            input_data.kwargs["clipRatio"] = ctypes.c_double(input_data.kwargs["clipRatio"])

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        out = None
        x = input_data.kwargs["x"]
        kroneckerP1 = input_data.kwargs["kroneckerP1"]
        kroneckerP2 = input_data.kwargs["kroneckerP2"]
        clipRatio = input_data.kwargs["clipRatio"]

        if self.output is None:
            out = self.get_golden(x, kroneckerP1, kroneckerP2, clipRatio)
        else:
            run()
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        return out

    # 函数入参签名校验
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnFlatQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *kroneckerP1, \
                const aclTensor *kroneckerP2, double clipRatio, aclTensor *out, aclTensor *quantScale, \
                uint64_t *workspaceSize,aclOpExecutor **executor)"


@register("function_aclnnFlatQuant_pgacl")
class FunctionaclnnFlatQuant_pgacl(AclnnBaseApi):

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        
        x = input_data.kwargs["x"]
        input_args.extend([input_args[4]])
        out = torch.zeros(x.shape[0], x.shape[1], x.shape[2]//8, dtype=torch.int32).npu()
        input_args[4] = self.backend.convert_input_data(out, name="out")[0]
        return input_args, output_packages