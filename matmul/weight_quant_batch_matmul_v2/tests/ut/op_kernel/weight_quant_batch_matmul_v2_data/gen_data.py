#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np


def weight_quant_batch_matmul_v2(m, k, n, group):
    np.random.random([m, k]).astype(np.float16).tofile("x.bin")
    np.random.uniform(-5, 5, [k, n]).astype(np.int8).tofile("weight.bin")
    np.random.random([n, group]).astype(np.float16).tofile("antiquant_scale.bin")
    np.random.random([n, group]).astype(np.float16).tofile("antiquant_offset.bin")
    np.random.random([n, group]).astype(np.float32).tofile("quant_scale.bin")
    np.random.random([n, group]).astype(np.float32).tofile("quant_offset.bin")

if __name__ == '__main__':
    m, k, n, group = [int(p) for p in sys.argv[1:]]
    weight_quant_batch_matmul_v2(m, k, n, group)
