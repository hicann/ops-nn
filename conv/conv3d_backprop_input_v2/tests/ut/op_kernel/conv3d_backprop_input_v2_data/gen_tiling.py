#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import sys

params_case_1 = [
    1, 1, 1, 1, 1, 1, # batchDim, groupDim, mDim, kDim, nDim, dDim
    1, 0, # coreNum, unknown
    # conv tiling as belows
    1, 512, 512, 32, 32, 1, 1, 16, 4, # batch, cin, cout, cout1, cin1, cout1G, cin1G, c0, c0Bits
    5, 32, 32, # dout, ho, wo
    5, 32, 32, # di, hi, wi
    1, 1, 1, # dk, hk, wk
    1, 1, 1, 1, # group strideD, strideH, strideW
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # pads & bp_pads
    1, 1, 1, # dilations
    2, 2, 1, 1, 1, # al0pb, bl0pb, cl0pb, al1pb, bl1pb
    1, 512, 32, 16, 5, 1, # sGroup, sCout, sCout1, sCin1, sDin, sHo
    96, 16, 256, 1, 1, 1, # baseM, baseK, baseN, baseD, baseBatch, baseGroup
    # stepM, stepN, stepKa, stepKb, stepBatch, stepGroup, iterateOrder, hf32Flag, sBatch, sM, sCin
    1, 1, 8, 8, 1, 1, 1, 0, 1, 0, 96, 0, 256, 0,
]

params_case_2 = [
    1, 1, 1, 1, 1, 1, # batchDim, groupDim, mDim, kDim, nDim, dDim
    1, 0, # coreNum, unknown
    # conv tiling as belows
    1, 512, 512, 32, 32, 1, 1, 16, 4, # batch, cin, cout, cout1, cin1, cout1G, cin1G, c0, c0Bits
    5, 32, 32, # dout, ho, wo
    5, 32, 32, # di, hi, wi
    1, 1, 1, # dk, hk, wk
    1, 1, 1, 1, # group strideD, strideH, strideW
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # pads & bp_pads
    1, 1, 1, # dilations
    2, 2, 1, 1, 1, # al0pb, bl0pb, cl0pb, al1pb, bl1pb
    1, 512, 32, 16, 5, 1, # sGroup, sCout, sCout1, sCin1, sDin, sHo
    96, 16, 256, 1, 1, 1, # baseM, baseK, baseN, baseD, baseBatch, baseGroup
    # stepM, stepN, stepKa, stepKb, stepBatch, stepGroup, iterateOrder, hf32Flag, sBatch, sM, sCin
    1, 1, 8, 8, 1, 1, 1, 1, 1, 0, 96, 0, 256, 0,
]

params_case_group_depthwise_1 = [
    2, 1, 4, 1, 1, 1, # batchDim, groupDim, mDim, kDim, nDim, dDim
    8, 0, # coreNum, unknown
    # conv tiling as belows
    2, 256, 256, 16, 16, 1, 1, 16, 4, # batch, cin, cout, cout1, cin1, cout1G, cin1G, c0, c0Bits
    1, 4, 4, # dout, ho, wo
    1, 8, 8, # di, hi, wi
    1, 2, 2, # dk, hk, wk
    16, 1, 2, 2, # group strideD, strideH, strideW
    0, 0, 0, 0, 0, 0, 1, 0, 1, 1, # pads & bp_pads
    1, 1, 1, # dilations
    2, 2, 1, 1, 1, # al0pb, bl0pb, cl0pb, al1pb, bl1pb
    16, 256, 1, 1, 1, 1, # sGroup, sCout, sCout1, sCin1, sDin, sHo
    16, 64, 16, 1, 1, 1, # baseM, baseK, baseN, baseD, baseBatch, baseGroup
    # stepM, stepN, stepKa, stepKb, stepBatch, stepGroup,
    1, 1, 2, 2, 1, 1,
    # iterateOrder, hf32Flag, initOutputFlag, outputPadD, outputPadH, reserved, sBatch, sM, sCin
    1, 0, 0, 0, 0, 0, 1, 0, 16, 0, 256, 0,
]

params_info = {
    "test_case_1": params_case_1,
    "test_case_2": params_case_2,
    "test_case_3": params_case_group_depthwise_1,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list[:], dtype=np.int32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)
    # attention_score_core_params.tofile(tiling_file)
    # transposeTilingDataTailCore.tofile(tiling_file)

if __name__ == '__main__':
    main()
