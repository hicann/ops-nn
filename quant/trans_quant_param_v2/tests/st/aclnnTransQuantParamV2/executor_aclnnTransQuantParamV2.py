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


from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("function_transquantparamV2")
class MethodTorchNnSoftmaxApi(BaseApi):

    def float32_to_int9(self , fp32_offset):
        fp32_offset = np.clip(fp32_offset, -256, 255)
        int_value = (np.round(fp32_offset)).astype(int)
        int9_value = np.clip(int_value, -256, 255)
        return int9_value

    def get_broadcast_shape(self , ipts):
        def _get_idx(ipt, idx):
            try:
                tmp = ipt[idx]
            except:
                tmp = 1
            return tmp

        max_len = max([len(ipt) for ipt in ipts])
        out_shape = [max([_get_idx(ipt, i) for ipt in ipts]) for i in range(-max_len, 0)]

        return out_shape

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        scale = input_data.kwargs["scale"]
        scale_shape = list(scale.shape)
        fp32_deq_scale = scale.numpy()
        uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(scale_shape)

        # 与高19位运算，模拟硬件
        uint32_deq_scale &= 0XFFFFE000

        uint64_deq_scale = np.zeros(scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)

        uint64_deq_scale |= (1 << 46)


        offset = input_data.kwargs["offset"]
        offset_shape = list(offset.shape)

        fp32_offset = offset.numpy()
        int9_value = self.float32_to_int9(fp32_offset)
        out_shape = self.get_broadcast_shape([scale_shape, offset_shape])

        uint64_int9_value = np.zeros(offset_shape, np.uint64)
        uint64_int9_value = (int9_value & 0X1FF) << 37
        uint64_deq_scale &= 0x4000FFFFFFFF

        if scale_shape != out_shape:
            uint64_deq_scale = np.tile(uint64_deq_scale, out_shape)
        if offset_shape != out_shape:
            uint64_int9_value = np.tile(uint64_int9_value, out_shape)
        uint64_deq_scale |= np.uint64(uint64_int9_value)

        int64_deq_scale = uint64_deq_scale.astype("int64")
        result = torch.from_numpy(int64_deq_scale)
        return result