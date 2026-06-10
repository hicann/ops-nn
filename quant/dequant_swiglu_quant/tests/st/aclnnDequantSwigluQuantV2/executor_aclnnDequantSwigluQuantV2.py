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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import torchair
import random
import numpy as np
import os
import json
import uuid


@register("ascend_method_dequant_swiglu_quant")
class MethodDequantSwigluQuant(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodDequantSwigluQuant, self).__init__(task_result)
        self.session = None

    def golden(
        self,
        x,
        weight_scale,
        activate_scale,
        bias,
        quant_scale,
        quant_offset,
        group_index,
        activate_left,
        quant_mode,
        swiglu_mode=0,
        clamp_limit=7.0,
        glu_alpha=1.702,
        glu_bias=1.0,
    ):
        x_dtype = x.dtype
        if len(x.shape) > 2:
            x = x.reshape(-1, x.shape[-1])

        if weight_scale is not None and len(weight_scale.shape) == 1:
            weight_scale = weight_scale.reshape(1, -1)

        if activate_scale is not None and len(activate_scale.shape) >= 1:
            activate_scale = activate_scale.reshape(-1, 1)
        if quant_mode == 1:
            if quant_scale is not None and len(quant_scale.shape) == 1:
                quant_scale = quant_scale.reshape(1, -1)

        if group_index is None:
            group_index = torch.randn([x.shape[0]])

        if quant_mode == 1:
            quant_mode = "dynamic"
        elif quant_mode == 0:
            quant_mode = "static"

        res_y = torch.zeros([x.shape[0], x.shape[1] // 2], dtype=torch.float32)
        res_scale = torch.zeros([x.shape[0]], dtype=torch.float32)
        scale_out = torch.tensor(0.0)
        offset = 0
        for g_idx in range(group_index.shape[0]):
            groupIdx = group_index[g_idx].to(torch.int32)
            if groupIdx == 0:
                continue
            x_tensor = x[offset: (offset + groupIdx)].to(torch.float32)
            if "int32" in str(x_dtype):
                if bias is not None and bias.dtype is torch.int32:
                    x_tensor = torch.add(x_tensor, bias[g_idx])
                res = torch.mul(x_tensor, weight_scale[g_idx].to(torch.float32))

                if activate_scale is not None:
                    res = torch.mul(res, activate_scale[offset: (offset + groupIdx)].to(torch.float32))

                if bias is not None and bias.dtype is not torch.int32:
                    res = torch.add(res, bias[g_idx].to(torch.float32))
            else:
                res = x_tensor

            if swiglu_mode == 1:
                self_tensor = res[..., ::2]
                other = res[..., 1::2]
            else:
                out = torch.chunk(res, 2, dim=-1)
                if activate_left:
                    self_tensor = out[0]
                    other = out[1]
                else:
                    self_tensor = out[1]
                    other = out[0]

            if swiglu_mode == 1:
                self_tensor = self_tensor.clamp(min=None, max=clamp_limit)
                other = other.clamp(min=-clamp_limit, max=clamp_limit)
                self_tensor = self_tensor * torch.sigmoid(glu_alpha * self_tensor)
                output = self_tensor * (other + glu_bias)
            else:
                output = torch.nn.functional.silu(self_tensor) * other

            if quant_scale is not None:
                if quant_mode == "static":
                    if(len(quant_scale.shape)==1):
                        quant_scale = quant_scale.unsqueeze(1).expand(-1, output.shape[1])
                    if(quant_offset is not None and len(quant_offset.shape)==1):
                        quant_offset = quant_offset.unsqueeze(1).expand(-1, output.shape[1])
                    output = torch.div(output, quant_scale[g_idx].to(torch.float32))
                    output = torch.add(output, quant_offset[g_idx].to(torch.float32))

                    scale_out = torch.tensor(0.0)
                else:
                    output = torch.mul(output, quant_scale[g_idx].to(torch.float32))
                    absd = torch.abs(output)
                    max_values = torch.amax(absd, dim=-1)
                    scale_out = max_values / 127
                    max_values = 127 / max_values
                    output = output * max_values.unsqueeze(1)

            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset: (offset + groupIdx)] = output
            res_scale[offset: (offset + groupIdx)] = scale_out
            offset = offset + groupIdx

        return res_y.to(torch.int8), res_scale

    def generate_random_list(self,length, total_sum):
        if length <= 0:
            return []
        if total_sum < length:
            raise ValueError("总和必须至少等于列表长度，以确保每个元素至少为1")

        result = [0] * length

        for i in range(total_sum):
            result[random.randint(0, length - 1)] += 1

        return result

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        quant_mode = input_data.kwargs.get("quant_mode")
        if quant_mode == "static":
            quant_mode = 0
        else:
            quant_mode = 1
        x = input_data.kwargs.get("x")
        quant_offset = input_data.kwargs.get("quant_offset", None) if quant_mode == 0 else None
        group_index = json.loads(input_data.kwargs.get("group_index", None))
        group_index = torch.tensor(group_index, dtype = torch.int64)
        weight_scale = input_data.kwargs.get("weight_scale")
        activation_scale = input_data.kwargs.get("activation_scale")
        if self.device == "cpu":
            y, scale = self.golden(
                x,
                weight_scale,
                activation_scale,
                input_data.kwargs.get("bias", None),
                input_data.kwargs["quant_scale"],
                quant_offset,
                group_index,
                input_data.kwargs.get("activate_left", True),
                quant_mode,
                input_data.kwargs.get("swiglu_mode", 0),
                input_data.kwargs.get("clamp_limit", 7.0),
                input_data.kwargs.get("glu_alpha", 1.702),
                input_data.kwargs.get("glu_bias", 1.0),
            )

            if quant_mode == 0:
                scale = torch.zeros(10, dtype=torch.float)
            return y, scale

        if self.device == "npu":
            model = Model()
            y, scale = model.forward(
                x.npu(),
                weight_scale.npu(),
                activation_scale.npu(),
                input_data.kwargs.get("bias", None),
                input_data.kwargs["quant_scale"],
                quant_offset.npu() if quant_offset is not None else None,
                group_index.npu(),
                input_data.kwargs.get("activate_left", True),
                quant_mode,
                input_data.kwargs.get("swiglu_mode", 0),
                input_data.kwargs.get("clamp_limit", 7.0),
                input_data.kwargs.get("glu_alpha", 1.702),
                input_data.kwargs.get("glu_bias", 1.0),
            )
            if quant_mode == 0:
                scale = torch.zeros(10, dtype=torch.float)
            return y, scale


@register("aclnn_dequant_swiglu_quant")
class DequantSwigluQuantAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def pre_process_data(self, input_data: InputDataset):
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase, AclDataType, AclFormat, AclTensor
        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)
        if input_data.kwargs.get("bias", None) == None:
            input_data.kwargs["bias"] = null_tensor_ptr
        if input_data.kwargs.get("quant_scale", None) == None:
            input_data.kwargs["quant_scale"] = null_tensor_ptr
        self.quant_mode = input_data.kwargs.get("quant_mode")
        if self.quant_mode == "static":
            input_data.kwargs["quant_offset"] = input_data.kwargs.get("quant_offset", None)
        else:
            input_data.kwargs["quant_offset"] = null_tensor_ptr
        x = input_data.kwargs.get("x")
        group_index_list = json.loads(input_data.kwargs.get("group_index", "[]"))
        input_data.kwargs["group_index"] = torch.tensor(
            group_index_list, dtype=torch.int64
        ).npu()

    def init_by_input_data(self, input_data: InputDataset):

        self.pre_process_data(input_data)
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        outputs = []
        for output_pack in output_packages:
            outputs.append(self.acl_tensor_to_torch(output_pack))
        if self.quant_mode == "static":
            outputs[1] = torch.zeros(10, dtype=torch.float)
        return outputs[0], outputs[1]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_index,
        activate_left,
        quant_mode,
        swiglu_mode,
        clamp_limit,
        glu_alpha,
        glu_bias,
    ):
        y, scale = torch_npu.npu_dequant_swiglu_quant(
            x,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            bias=bias,
            quant_scale=quant_scale,
            quant_offset=quant_offset,
            group_index=group_index,
            activate_left=activate_left,
            quant_mode=quant_mode,
            swiglu_mode=swiglu_mode,
            clamp_limit=clamp_limit,
            glu_alpha=glu_alpha,
            glu_bias=glu_bias,
        )
        return y, scale