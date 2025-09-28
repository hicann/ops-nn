/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file cross_entropy_loss_tiling.cpp
 * \brief
 */

#ifndef _TEST_CROSS_ENTROPY_LOSS_H_
#define _TEST_CROSS_ENTROPY_LOSS_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct CrossEntropyLossTilingData {
    uint64_t targetNum;
    uint64_t frontCoreNum;
    uint64_t tailCoreNum;
    uint64_t frontBatchNum;
    uint64_t tailBatchNum;
    uint64_t inputUbSize;
    uint64_t castTmpBufByte;
    uint64_t lnTmpBufSize;
    uint64_t weightTmpBufSize;
    uint64_t weight4SmoothingBufSize;
    uint64_t totalTmpBufByte;
    uint64_t ubLoopNum;
    uint64_t ubTailNum;
    uint64_t vecLoopNum;
    uint64_t vecTailNum;
    uint64_t tailVecLoopNum;
    uint64_t tailVecTailNum;
    uint64_t reduction;
    int64_t ignoreIndex;
    float labelSmoothing;
    uint32_t defaultWeight = 0;
};

inline void InitCrossEntropyLossTilingData(uint8_t* tiling, CrossEntropyLossTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(CrossEntropyLossTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    CrossEntropyLossTilingData tilingData;         \
    InitCrossEntropyLossTilingData(tilingPointer, &tilingData)
#endif
