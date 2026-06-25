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
 * \file gru_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _GRU_TILING_DATA_H_
#define _GRU_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

struct GruTilingData {
    int64_t tilingKey = 0;
    int64_t usedCoreNum = 0;

    //  gru input tiling
    int64_t timeStep = 0;
    int64_t batch = 0;
    int64_t inputSize = 0;
    int64_t hiddenSize = 0;
    int64_t isBias = 0;
    int64_t isInith = 0;
    int64_t isSeqLength = 0;
    int64_t totalSteps = 0;

    // gru attr
    int64_t direction = 0;
    int64_t isTraining = 0;

    // vector block params
    int64_t singleCoreM = 0;
    int64_t singleCoreMTail = 0;
    int64_t singleCoreN = 0;
    int64_t singleCoreNTail = 0;
    int64_t baseN = 0;
    int64_t baseM = 0;
    int64_t mCnt = 0;
    int64_t nCnt = 0;

    TCubeTiling inputMMParam;
    TCubeTiling hiddenMMParam;
};

#endif