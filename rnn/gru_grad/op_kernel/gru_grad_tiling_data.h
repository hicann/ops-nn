/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_tiling_data.h
 * \brief
 */
#ifndef _GRU_GRAD_TILING_DATA_H_
#define _GRU_GRAD_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

struct CutBatchTiling {
    int64_t taskNum = 0;
    int64_t copyMLines = 0;
    int64_t copyMLinesTail = 0;
    int64_t nLoop = 0;
    int64_t copyNLength = 0;
    int64_t copyNLengthTail = 0;
    int64_t splitTaskPerCore = 0;
    int64_t splitPreCore = 0;
};

struct GruGradTilingData {
    int64_t ubSize = 0;
    int64_t timeStep = 0;
    int64_t batch = 0;
    int64_t inputSize = 0;
    int64_t hiddenSize = 0;
    int64_t isBias = 0;
    int64_t isSeqLength = 0;
    int64_t totalSteps = 0; // sum(batch_sizes) 当不定长; = timeStep * batch 当定长

    int64_t singleCoreM = 0;
    int64_t singleCoreMTail = 0;
    int64_t singleCoreN = 0;
    int64_t singleCoreNTail = 0;
    int64_t baseN = 0;
    int64_t baseM = 0;
    int64_t mCnt = 0;
    int64_t nCnt = 0;

    int64_t singleCoreReduceN = 0;
    int64_t singleCoreReduceNTail = 0;
    int64_t baseReduceN = 0;
    int64_t nReduceCnt = 0;
    int64_t maxReduceNumOnce = 0;
    int64_t reduceBlockSize = 0;

    int64_t direction = 0;

    int64_t inputSizeAligned = 0;
    int64_t hiddenSizeAligned = 0;
    int64_t oneLineAligned = 0;
    CutBatchTiling dxhInputTiling;
    CutBatchTiling dxhHiddenTiling;
    CutBatchTiling xhInputTiling;
    CutBatchTiling xhHiddenTiling;

    TCubeTiling dwIhMMParam;
    TCubeTiling dwHhMMParam;
    TCubeTiling dgateMMParam;
    TCubeTiling dxMMParam;
};

#endif // _GRU_GRAD_TILING_DATA_H_
