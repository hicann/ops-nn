/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_single_layer_lstm_grad_tiling_def.h
 * \brief
 */
#ifndef TEST_SINGLE_LAYER_LSTM_GRAD_TILING_DEF_H_
#define TEST_SINGLE_LAYER_LSTM_GRAD_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

struct SingleLayerLstmGradTilingDataTest {
    int64_t ubSize = 0;
    // lstm input tiling
    int64_t timeStep = 0;
    int64_t batch = 0;
    int64_t inputSize = 0;
    int64_t hiddenSize = 0;
    int64_t isBias = 0;
    int64_t isSeqLength = 0;

    // vector block params
    int64_t singleCoreM = 0;
    int64_t singleCoreMTail = 0;
    int64_t singleCoreN = 0;
    int64_t singleCoreNTail = 0;
    int64_t baseN = 0;
    int64_t baseM = 0;
    int64_t mCnt = 0;
    int64_t nCnt = 0;

    // reduce block params
    int64_t singleCoreReduceN = 0;
    int64_t singleCoreReduceNTail = 0;
    int64_t baseReduceN = 0;
    int64_t nReduceCnt = 0;
    int64_t maxReduceNumOnce = 0;
    int64_t reduceBlockSize = 0;

    // lstm attr
    int64_t gateOrder = 0;
    int64_t direction = 0;
    float cellClip = 0;
    float forgetBias = 0;

    // split and concat params
    int64_t inputSizeAligned = 0;
    int64_t hiddenSizeAligned = 0;
    int64_t oneLineAligned = 0;

    CutBatchTiling dxhInputTiling;
    CutBatchTiling dxhHiddenTiling;
    CutBatchTiling xhInputTiling;
    CutBatchTiling xhHiddenTiling;

    // matmul params
    TCubeTiling dwMMParam;
    TCubeTiling dgateMMParam;
};

inline void InitSingleLayerLstmGradTilingDataTest(uint8_t* tiling, SingleLayerLstmGradTilingDataTest* data)
{
    memcpy(data, tiling, sizeof(SingleLayerLstmGradTilingDataTest));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                    \
    SingleLayerLstmGradTilingDataTest tiling_data;                  \
    InitSingleLayerLstmGradTilingDataTest(tiling_arg, &tiling_data)
#endif // FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H_TEST_KERNEL