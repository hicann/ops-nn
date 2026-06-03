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
import torch_npu
import torch.nn.functional as F

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("aclnn_masked_softmax_with_rel_pos_bias")
class AclnnMaskedSoftmaxWithRelPosBiasApi(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None

        if self.device == 'npu':
            torch.npu.set_compile_mode(jit_compile=False)
            device = f"npu:{self.device_id}"
            x = input_data.kwargs['x'].to(device)
            atten_mask = input_data.kwargs.get('attenMaskOptional')
            atten_mask = atten_mask.to(device) if atten_mask is not None else None
            bias = input_data.kwargs['relativePosBias'].to(device)
            scale_value = float(input_data.kwargs['scaleValue'])
            inner_mode = int(input_data.kwargs.get('innerPrecisionMode', 0))

            output = torch_npu.npu_masked_softmax_with_rel_pos_bias(
                x, atten_mask, bias,
                scale_value=scale_value,
                inner_precision_mode=inner_mode
            )

        if self.device == 'cpu':
            x = input_data.kwargs['x']
            atten_mask = input_data.kwargs.get('attenMaskOptional')
            bias = input_data.kwargs['relativePosBias']
            scale_value = float(input_data.kwargs['scaleValue'])

            orig_dtype = x.dtype
            x_f32 = x.to(torch.float32)
            y = torch.mul(x_f32, scale_value)
            y = torch.add(y, bias.to(torch.float32))
            if atten_mask is not None and atten_mask.numel() > 0:
                y = torch.add(y, atten_mask.to(torch.float32))

            output = F.softmax(y, dim=-1)
            output = output.to(orig_dtype)

        return output.cpu() if output is not None else None

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND
