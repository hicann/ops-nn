#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import logging
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

try:
    import torchvision
except Exception:
    logging.warning("import torch_npu failed!!!")


@register("function_aclnnDeformableConv2d")
class FunctionaclnnDeformableConv2d(BaseApi):
    # 精度对比标杆
    def get_golden(
        self,
        x,
        weight,
        offset,
        biasOptional,
        kernelSize,
        stride,
        padding,
        dilation,
        groups,
        deformableGroups,
    ):
        dtype = x.dtype
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(torch.float32)
            weight = weight.to(torch.float32)
            offset = offset.to(torch.float32)
            biasOptional = (
                None if biasOptional is None else biasOptional.to(torch.float32)
            )
        offset_tensors = torch.split(offset, offset.shape[1] // 3, dim=1)  # W, H, mask
        offsetHW = torch.cat([offset_tensors[1], offset_tensors[0]], dim=1)
        out = torchvision.ops.deform_conv2d(
            x,
            offsetHW,
            weight,
            biasOptional,
            stride=[stride[2], stride[3]],
            padding=[padding[0], padding[2]],
            dilation=[dilation[2], dilation[3]],
            mask=offset_tensors[2],
        )
        return out.to(dtype)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        offset = input_data.kwargs["offset"]
        biasOptional = input_data.kwargs["biasOptional"]
        kernelSize = input_data.kwargs["kernelSize"]
        stride = input_data.kwargs["stride"]
        padding = input_data.kwargs["padding"]
        dilation = input_data.kwargs["dilation"]
        groups = input_data.kwargs["groups"]
        deformableGroups = input_data.kwargs["deformableGroups"]

        out = self.get_golden(
            x,
            weight,
            offset,
            biasOptional,
            kernelSize,
            stride,
            padding,
            dilation,
            groups,
            deformableGroups,
        )
        return out

    # 函数入参签名校验
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(const aclTensor* x, const aclTensor* weight, const aclTensor* offset, \
                                                  const aclTensor* biasOptional, const aclIntArray* kernelSize, \
                                                  const aclIntArray* stride, const aclIntArray* padding, \
                                                  const aclIntArray* dilation, int64_t groups, int64_t deformableGroups, \
                                                  bool modulated, aclTensor* out, aclTensor* deformOutOptional, \
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)"


@register("function_aclnnDeformableConv2d_pgacl")
class FunctionaclnnDeformableConv2d_pgacl(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)

        x = input_data.kwargs["x"]
        weight = input_data.kwargs["weight"]
        offset = input_data.kwargs["offset"]

        input_args.extend(
            self.backend.convert_input_data(
                torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    offset.shape[2] * weight.shape[2],
                    offset.shape[3] * weight.shape[3],
                    dtype=x.dtype,
                ),
                name="deformOut",
            )
        )  # deformOut

        return input_args, output_packages
