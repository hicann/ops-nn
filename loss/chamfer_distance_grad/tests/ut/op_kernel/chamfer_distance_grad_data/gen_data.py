#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(current_path)

input_path = os.path.join(project_path, "build/input")
output_path = os.path.join(project_path, "build/output")


def write_file(inputs, outputs):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for input_name in inputs:
        inputs[input_name].tofile(os.path.join(input_path, input_name+".bin"))
    for output_name in outputs:
        outputs[output_name].tofile(os.path.join(output_path, output_name+".bin"))


def chamfer_distance_grad_cpu(grad_dist1, grad_dist2, idx1, idx2, xyz1, xyz2):
    batch, num = grad_dist1.shape
    dtype = grad_dist1.dtype
    grad_xyz1_cpu = np.zeros((batch, num, 2), dtype=dtype)
    grad_xyz2_cpu = np.zeros((batch, num, 2), dtype=dtype)
    for b in range(batch):
        for n in range(num):
            cur_ind = idx1[b][n]
            x1, y1 = xyz1[b][n]
            x2, y2 = xyz2[b][cur_ind]
            g1 = grad_dist1[b][n] * 2
            grad_xyz1_cpu[b][n][0] += (x1 - x2) * g1
            grad_xyz1_cpu[b][n][1] += (y1 - y2) * g1
            grad_xyz2_cpu[b][cur_ind][0] -= (x1 - x2) * g1
            grad_xyz2_cpu[b][cur_ind][1] -= (y1 - y2) * g1
    for b in range(batch):
        for n in range(num):
            cur_ind = idx2[b][n]
            x1, y1 = xyz2[b][n]
            x2, y2 = xyz1[b][cur_ind]
            g2 = grad_dist2[b][n] * 2
            grad_xyz2_cpu[b][n][0] += (x1 - x2) * g2
            grad_xyz2_cpu[b][n][1] += (y1 - y2) * g2
            grad_xyz1_cpu[b][cur_ind][0] -= (x1 - x2) * g2
            grad_xyz1_cpu[b][cur_ind][1] -= (y1 - y2) * g2
    return grad_xyz1_cpu, grad_xyz2_cpu


if __name__ == "__main__":
    batch_size = int(sys.argv[1])
    num = int(sys.argv[2])
    np_dtype = sys.argv[3]

    xyz1 = np.random.uniform(0, 10, (batch_size, num, 2)).astype(np_dtype)
    xyz2 = np.random.uniform(0, 10, (batch_size, num, 2)).astype(np_dtype)
    grad_dist1 = np.random.uniform(0, 3, (batch_size, num)).astype(np_dtype)
    grad_dist2 = np.random.uniform(0, 3, (batch_size, num)).astype(np_dtype)
    idx1 = np.random.randint(0, num, (batch_size, num)).astype(np.int32)
    idx2 = np.random.randint(0, num, (batch_size, num)).astype(np.int32)
    grad_xyz1_cpu, grad_xyz2_cpu = chamfer_distance_grad_cpu(grad_dist1, grad_dist2, idx1, idx2, xyz1, xyz2)


    inputs_npu = {"xyz1":xyz1 ,"xyz2": xyz2, "grad_dist1":grad_dist1, "grad_dist2": grad_dist2,
                  "idx1":idx1, "idx2": idx2}
    outputs_npu = {"grad_xyz1":np.zeros_like(xyz1), "grad_xyz2":np.zeros_like(xyz2)}
    write_file(inputs_npu, outputs_npu)
