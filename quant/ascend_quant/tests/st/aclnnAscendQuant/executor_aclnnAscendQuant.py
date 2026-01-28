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

import torch
import numpy as np
import os
import torch_npu
import sys

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("aclnn_ascend_quant")
class AclnnAscendQuant(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        scale = input_data.kwargs["Scale"]
        offset = input_data.kwargs["offset"]
        sqrt_mode = input_data.kwargs["sqrtMode"]
        round_mode = input_data.kwargs["roundMode"]
        dst_type = input_data.kwargs["dstDtype"]
        axis = -1

        output_data = None

        x = x.to(torch.float32)

        if len(scale.shape) == 1:
            scale_new_shape = [1] * len(x.shape)
            scale_new_shape[axis] = scale.shape[0]
            scale = torch.reshape(scale, scale_new_shape)
            if offset is not None:
                offset = torch.reshape(offset, scale_new_shape)

        scale = scale.to(torch.float32)
        if offset is not None:
            offset = offset.to(torch.float32)
        x = x.numpy()
        scale = scale.numpy()
        if offset is not None:
            offset = offset.numpy()
        if sqrt_mode:
            scale_sqrt = x * scale
            scale_rst = scale_sqrt * scale
        else:
            scale_rst = x * scale
        if offset is not None:
            add_offset = scale_rst + offset
        else:
            add_offset = scale_rst
        if round_mode == "round":
            round_data = np.round(add_offset, 0)
        elif round_mode == "floor":
            round_data = np.floor(add_offset)
        elif round_mode == "ceil":
            round_data = np.ceil(add_offset)
        else:
            round_data = np.trunc(add_offset)
        if dst_type == 2 or dst_type == "torch.quint8":
            round_data = np.clip(round_data, -128, 127)
            output_data = torch.from_numpy(round_data.astype("int8"))
        if dst_type == "torch.quint4x2":
            from ml_dtypes import int4
            round_data = np.clip(round_data, -8, 7).astype("int8")
            output_data = round_data.astype("int8")
            output_data = torch.from_numpy(output_data)
        if  dst_type == 29:
            from ml_dtypes import int4
            round_data = np.clip(round_data, -8, 7).astype("int8")
            output_data = round_data.astype("int8")
            # params['need_int4']=True
            output_data = torch.from_numpy(output_data)
        if dst_type == 3:
            from ml_dtypes import int4
            round_data = np.clip(round_data, -8, 7).astype(int4)
            output_data = pack_int4_to_int32(round_data)
            output_data = torch.from_numpy(output_data)

        return output_data

def pack_int4_to_int32(int4arr):
    int4arr = int4arr.astype(np.int8)

    original_shape = int4arr.shape
    if len(original_shape) == 0:
        original_shape = (8,)
    dst_shape = (*(original_shape[:-1]), original_shape[-1] // 8)

    int4arr = int4arr & 0b00001111

    shifts = np.arange(8) * 4

    int4arr = int4arr.reshape(-1, 8).astype(np.int32)

    int32arr = np.zeros(int4arr.shape[0], dtype=np.int32)
    for i in range(8):
        int32arr |= int4arr[:, i] << shifts[i]

    result = int32arr.reshape(dst_shape)

    return result


def unpack_int4(s32arr):
    from ml_dtypes import int4

    dst_shape = s32arr.shape
    if len(dst_shape) == 0:
        dst_shape = (8,)
    else:
        dst_shape = (*(dst_shape[:-1]), dst_shape[-1] * 8)

    sa1 = s32arr.astype(np.int32)
    sa2 = sa1.tobytes()
    sa3 = np.frombuffer(sa2, dtype=np.uint8)
    shift = np.array([0, 4], dtype=np.uint8)
    sa4 = np.bitwise_and(sa3.reshape([-1, 1]) >> shift, 0b00001111).astype(int4).astype(np.int8).reshape(dst_shape)
    return sa4

