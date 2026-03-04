# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
gen_data.py
"""
import sys
from functools import reduce

import numpy as np
import torch


class IfmrAlgNpu():
    def __init__(self, bins_num=512,
                 min_percentile=0.999999,
                 max_percentile=0.999999,
                 search_range=[0.7, 1.3],
                 search_step=0.01,
                 with_offset=True):
        self.ifmr_param = {
            "bins_num": bins_num,
            "min_percentile": min_percentile,
            "max_percentile": max_percentile,
            "search_range": search_range,
            "search_step": search_step,
            "with_offset": with_offset
        }

    def forward(self, data):
        bins_num = self.ifmr_param["bins_num"]
        min_percentile = self.ifmr_param["min_percentile"]
        max_percentile = self.ifmr_param["max_percentile"]
        search_range = self.ifmr_param["search_range"]
        search_step = self.ifmr_param["search_step"]
        with_offset = self.ifmr_param["with_offset"]

        # process
        data = data.astype(np.float32)
        data_shape = data.shape
        data_type = data.dtype
        # 数据预处理
        data_max = np.max(data)
        data_min = np.min(data)
        if data_min > 0:
            data_min = 0
        if data_max < 0:
            data_max = 0
        data_num = reduce(lambda x, y: x * y, data_shape)
        data_max = np.array([data_max], dtype=data_type)
        data_min = np.array([data_min], dtype=data_type)
        # 计算累加和
        bins, threshold = np.histogram(data, bins_num)
        cumsum = np.cumsum(bins).astype(np.int32)
        cdf = cumsum / data_num

        print('---------ifmr debug------------')
        print('min', data_min)
        print('max', data_max)
        print('cdf', cumsum)
        print('-------------------------------')
        # 生成 scale&offset
        max_index = np.where(cdf > max_percentile, 0, 1).sum()
        min_index = np.where(cdf > 1 - min_percentile, 0, 1).sum()
        max_init = max_index / bins_num * (data_max - data_min) + data_min
        min_init = min_index / bins_num * (data_max - data_min) + data_min
        step = np.arange(search_range[0], search_range[1], search_step)
        max_list = max_init * step
        min_list = min_init * np.ones(step.shape)
        scale = (max_list - min_list) / 255

        offset = np.round(min_list / scale)
        offset = -(offset + 128)

        # 找到最优
        data_list = data.flatten()
        loss_list = np.zeros(len(step))
        for i in range(len(step)):
            quant_data_list = np.round(data_list / scale[i]) + offset[i]
            np.clip(quant_data_list, -128, 127, out=quant_data_list)
            quant_data_list = (quant_data_list - offset[i]) * scale[i]
            loss = np.sum(np.square(quant_data_list - data_list))
            loss_list[i] = loss

        index = np.unravel_index(np.argmin(loss_list), loss_list.shape)
        best_scale = scale[index]
        best_offset = offset[index]

        return best_scale, best_offset


def gen_golden_data_simple(shape, min, max,
    bins_num=512, min_percentile=0.999999, max_percentile=0.999999,
    search_range=[0.7, 1.3], search_step=0.01, with_offset=True):

    bins_num = int(bins_num)
    min = float(min)
    max = float(max)
    shape = int(shape)

    if (min == max):
        input_np = np.random.uniform(0, 1024, size=shape).astype(float)
    else:
        input_np = np.random.uniform(min, max, size=shape).astype(float)

    input_x = torch.from_numpy(input_np).reshape(shape)
    input_x_fp32 = input_x.to(torch.float32)
    hist = torch.histc(input_x_fp32, bins_num)
    cumsum = torch.cumsum(hist, 0)
    ifmr_mod = IfmrAlgNpu(bins_num, min_percentile,
                 max_percentile, search_range, search_step, with_offset)
    golen_scale, golden_offset = ifmr_mod.forward(input_np)
    # save inputs
    input_x_fp32.numpy().tofile("./inputs.bin")
    torch.max(input_x_fp32).numpy().tofile("./inputs_max.bin")
    torch.min(input_x_fp32).numpy().tofile("./inputs_min.bin")
    cumsum.detach().numpy().tofile("./cumsum.bin")
    # save outputs
    golen_scale.tofile("./golen_scale.bin")
    golden_offset.tofile("./golden_offset.bin")


if __name__ == "__main__":
    gen_golden_data_simple(*sys.argv[1:])