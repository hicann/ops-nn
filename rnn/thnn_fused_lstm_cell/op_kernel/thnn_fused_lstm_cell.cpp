/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file thnn_fused_lstm_cell.cpp
 * \brief
 */
#include "thnn_fused_lstm_cell.h"
#include "kernel_operator.h"
using namespace AscendC;
using namespace ThnnFusedLstmCellNS;

extern "C" __global__ __aicore__ void thnn_fused_lstm_cell(
    GM_ADDR inputGates, GM_ADDR hiddenGates, GM_ADDR cx, GM_ADDR inputBias, GM_ADDR hiddenBias,
    GM_ADDR hy, GM_ADDR cy, GM_ADDR storage, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    ThnnFusedLstmCell<DTYPE_HY> op(tilingData);
    op.Init(inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage);
    op.Process();
}