#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import math
import torch

CUBE_N = 16

K0_MAP = {
    torch.float32: 8,
    torch.float16: 16,
    torch.bfloat16: 16,
}


def lcm(m, n):
    return (m * n) // math.gcd(m, n)


def ceil(m, n):
    return (m + n - 1) // n


def ncdhw_to_ndc1hwc0(input_tensor):
    cube_k = K0_MAP[input_tensor.dtype]
    input_shape = input_tensor.shape
    dim_c = input_shape[1]
    pad_value = (cube_k - dim_c % cube_k) % cube_k
    res_tensor = torch.nn.functional.pad(input_tensor, (0, 0, 0, 0, 0, 0, 0, pad_value), mode="constant", value=0)
    dim_c1 = (dim_c + pad_value) // cube_k
    res_tensor = res_tensor.reshape((input_shape[0], dim_c1, cube_k, input_shape[2], input_shape[3], input_shape[4]))
    res_tensor = res_tensor.permute(0, 3, 1, 4, 5, 2)
    return res_tensor.contiguous()


def ndc1hwc0_to_ncdhw(output_tensor, real_cout):
    def ndc1hwc0_to_ndchw():
        output_shape = output_tensor.shape
        res_shape = [output_shape[0], output_shape[1], real_cout, output_shape[3], output_shape[4]]
        res_tensor = torch.zeros(res_shape, dtype=output_tensor.dtype)
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                for k in range(output_shape[2]):
                    for m in range(output_shape[5]):
                        if k * output_shape[5] + m < real_cout:
                            res_tensor[i, j, k * output_shape[5] + m, :, :] = output_tensor[i, j, k, :, :, m]
        return res_tensor

    res_tensor = ndc1hwc0_to_ndchw()
    res_tensor = res_tensor.permute(0, 2, 1, 3, 4)
    return res_tensor.contiguous()


def ncdhw_to_fz3d(filter_tensor, groups=1):
    cube_k = K0_MAP[filter_tensor.dtype]

    cout, cin_ori, dk, kh, kw = filter_tensor.shape
    cout_ori = cout // groups
    enlarge = min(lcm(lcm(cin_ori, cube_k) // cin_ori, lcm(cout_ori, CUBE_N) // cout_ori), groups)
    cin_opt = ceil(enlarge * cin_ori, cube_k) * cube_k
    cout_opt = ceil(enlarge * cout_ori, CUBE_N) * CUBE_N
    group_opt = ceil(groups, enlarge)

    filter_shape_gdc1hwnc0 = [group_opt, dk, cin_opt // cube_k, kh, kw, cout_opt, cube_k]
    res_tensor = torch.zeros(filter_shape_gdc1hwnc0, dtype=filter_tensor.dtype)
    for g in range(groups):
        for ci in range(cin_ori):
            for co in range(cout_ori):
                for kernel_d in range(dk):
                    e = g % enlarge
                    dst_ci = e * cin_ori + ci
                    dst_co = e * cout_ori + co
                    src_co = g * cout_ori + co
                    res_tensor[g // enlarge, kernel_d, dst_ci // cube_k, :, :, dst_co,
                    dst_ci % cube_k] = filter_tensor[
                                       src_co, ci,
                                       kernel_d, :,
                                       :]
    res_tensor = res_tensor.reshape(group_opt * dk * (cin_opt // cube_k) * kh * kw, cout_opt // CUBE_N, CUBE_N,
                                    cube_k)
    return res_tensor.contiguous()
