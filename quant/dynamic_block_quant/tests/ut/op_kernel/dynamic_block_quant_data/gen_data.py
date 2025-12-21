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

case_list = {
    "case1" : {"x":[60, 14, 16, 128], "y":[60, 14, 16, 128]},
    "case2" : {"x":[4, 10, 128], "y":[4, 10, 128]},
    "case3" : {"x":[10, 512], "y":[10, 512]},
    "case4" : {"x":[1024], "y":[1024]},
    "case5" : {"x":[10, 2, 4, 128, 1], "y":[10, 2, 4, 128, 1]},
    "case6" : {"x":[5, 2, 8, 64, 2, 2], "y":[5, 2, 8, 64, 2, 2]},
    "case7" : {"x":[20, 16, 8, 64, 2, 2, 2], "y":[20, 16, 8, 64, 2, 2, 2]},
    "case8" : {"x":[40, 15, 11, 32, 2, 2, 2, 2], "y":[40, 15, 11, 32, 2, 2, 2, 2]},
    "case9" : {"x":[256, 10, 32, 1], "y":[256, 10, 32, 1]},
    "case10" : {"x":[64, 12, 128, 4], "y":[64, 12, 128, 4]},
    "case11" : {"x":[96, 48, 160, 2], "y":[96, 48, 160, 2]},
    "case12" : {"x":[2, 512, 64], "y":[2, 512, 64]},
}

def gen_golden_data_simple(param, dtype):
    x = np.random.random(param["x"]).astype(dtype)
    x.tofile("x.bin")
    x.tofile("y_golden.bin")



if __name__ == "__main__":
    case_name = sys.argv[1]
    dtype = sys.argv[2]
    param = case_list.get(case_name)