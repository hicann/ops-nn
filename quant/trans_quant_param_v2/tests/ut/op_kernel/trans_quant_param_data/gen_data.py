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
import numpy as np


def quant_batch_matmul_v3(shape, has_offset):

    scale_shape = [shape,]
    scale = np.random.uniform(-1, 1, scale_shape).astype(np.float32)
    scale.tofile("scale.bin")

    if has_offset:
        offset_shape = [shape,]
        offset = np.random.uniform(-5, 5, offset_shape).astype(np.float32)
        offset.tofile("offset.bin")

if __name__ == '__main__':
    shape, has_offset = [int(p) for p in sys.argv[1:3]]
    quant_batch_matmul_v3(shape, has_offset)