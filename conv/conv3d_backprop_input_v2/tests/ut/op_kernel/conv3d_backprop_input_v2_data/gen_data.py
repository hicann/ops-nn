#!/usr/bin/python3
# coding=utf-8
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

def conv3d_backprop_input_v2(n, cin, cout, dout, ho, wo, di, hi, wi, dk, hk, wk):
    input_size = np.array([n, cin, di, hi, wi]).astype(np.int32)
    dedy_shape = [n, cout, dout, ho, wo]
    filter_shape = [cout, cin, dk, hk, wk]
    filter_ = np.random.uniform(-5, 5, filter_shape).astype(np.float16)
    dedy = np.random.uniform(-5, 5, dedy_shape).astype(np.float16)

    input_size.tofile("input_size.bin")
    filter_.tofile("filter.bin")
    dedy.tofile("dedy.bin")

if __name__ == '__main__':
    n, cin, cout, dout, ho, wo, di, hi, wi, dk, hk, wk = [int(p) for p in sys.argv[1:13]]

    conv3d_backprop_input_v2(n, cin, cout, dout, ho, wo, di, hi, wi, dk, hk, wk)
