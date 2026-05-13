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
import sys

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclTensorlistStruct, AclFormat

@register("aclnn_group_quant")
class AclnnGroupQuant(BaseApi):
    def init_by_input_data(self, input_data):
        torch.manual_seed(self.task_result.case_config.id)
        x = input_data.kwargs["x"]
        group_index = input_data.kwargs["groupIndex"]

        max_group_index = 5
        group_index = torch.randint(0, max_group_index,group_index.shape, dtype = group_index.dtype)
        group_index = torch.sort(group_index)[0]
        group_index[-1] = x.shape[0]

        input_data.kwargs["groupIndex"] = group_index

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        scale = input_data.kwargs["Scale"]
        offset = input_data.kwargs["offsetOptional"]
        group_index = input_data.kwargs["groupIndex"]
        attr_dst_type = input_data.kwargs["dstDtype"]

        dim_s = x.shape[0]
        dim_h = x.shape[1]
        dim_e = scale.shape[0]
        if attr_dst_type == 3:
            assert dim_h % 8 == 0, "For output y, if datatype is int32, dim of last axis should be multiples of 8"

        x_fp32 = x.to(torch.float32).numpy()
        scale_fp32 = scale.to(torch.float32).numpy()
        offset_fp32 = offset.to(torch.float32).numpy() if offset is not None else None
        y_fp32 = np.empty(shape=(0, dim_h), dtype='float32')

        # check for group_index
        group_index_np = group_index.numpy()
        # print(f"group_index_np------{group_index_np}")
        min_index = np.min(group_index_np)

        assert min_index >= 0, "group_index value should be greater than or equal 0"
        max_index = np.max(group_index_np)

        assert max_index <= dim_s, "group_index value should be less than or equal S"
        if group_index_np.size > 1:
            diff_index = group_index_np[1:] - group_index_np[:-1]
            min_diff = np.min(diff_index)
            assert min_diff >= 0, "group_index value must be monotonically non decreasing"

        assert group_index_np[-1] == dim_s, "group_index last value must be S"

        for row_scale in range(dim_e):
            x_start_row = 0 if row_scale == 0 else group_index_np[row_scale - 1]
            x_end_row = group_index_np[row_scale]
            if x_start_row < x_end_row:
                y_rows = x_fp32[x_start_row: x_end_row] * scale_fp32[row_scale]
                if offset is not None:
                    y_rows = y_rows + offset_fp32
                y_fp32 = np.concatenate([y_fp32, y_rows], axis=0)
        y_round = np.round(y_fp32, 0)
        # print(f"attr_dst_type------{attr_dst_type}")
        # y_dtype = params["dtype_output"][0]
        y_dtype = attr_dst_type
        # print(f"y_dtype1111------{y_dtype}")
        res = None
        if y_dtype == 2:
            res = np.clip(y_round, -128, 127).astype('int8')
        elif y_dtype == 29:
            res = np.clip(y_round, -8, 7).astype('int8')
        # elif y_dtype == 3:
        #     y_int4 = np.clip(y_round, -8, 7).astype('int4')
        #     res = pack_int4_as_int32(y_int4).reshape(-1, (dim_h + 7) // 8)
        else:
            raise Exception("attr dst_type only support 2(int8), 3(int32), 29(int4)")
        output_data = torch.from_numpy(res)
        # print(f"output--------{output_data}")

        # print(f"input_data-------{input_data.kwargs}")

        return output_data

def pack_int4_as_int32(src: np.ndarray):
    """
    Pack int4 numpy array (each int4 number stored in one byte actually) to be continuous bytes stored in uint8
    """
    pack_size = 2
    shift = np.array([0, 4], dtype=np.uint8)
    array = src
    if src.size % pack_size != 0:
        array = np.pad(src.flatten(), (0, pack_size - src.size % pack_size), mode='constant')
    reshaped = array.reshape([-1, 2])

    # bitwise_and is for arm
    u8arr = np.sum(np.bitwise_and(reshaped.view(np.uint8), 0b00001111) << shift, axis=1, dtype=np.uint8)

    val_bytes = u8arr.tobytes()
    s32arr = np.frombuffer(val_bytes, dtype=np.int32)

    return s32arr

@register("pyaclnn_group_quant")
class aclnnTopkExecutor(AclnnBaseApi):
    
    def init_by_input_data(self, input_data: InputDataset):
        import ctypes
        input_data.kwargs["groupIndex"] = input_data.kwargs["groupIndex"].npu()
    
        input_args, output_packages = super().init_by_input_data(input_data)
        
        # === 手动构造固定参数 ===
        input_args[4] = ctypes.c_int(input_data.kwargs["dstDtype"])
        # output_packages[0].dtype = input_data.kwargs["dstDtype"]

        return input_args, output_packages
