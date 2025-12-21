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
 * \file cross_entropy_loss_grad_tiling_data.h
 * \brief cross_entropy_loss_grad_tiling_data
 */

 #ifndef CROSS_ENTROPY_LOSS_GRAD_TILING_DATA_H
 #define CROSS_ENTROPY_LOSS_GRAD_TILING_DATA_H

 #include <cstdint>

struct CrossEntropyLossGradRegbaseTilingData {
    int64_t rowVal;
    int64_t colVal;
    int64_t reduction;
    int64_t ignoreIndex;
    float labelSmoothing;
    int64_t usedCoreNum;
    int64_t frontCoreNum;
    int64_t tailCoreNum;
    int64_t frontRowNum;
    int64_t tailRowNum; // c循环的主块次数，
    int64_t alignColLoopNum; // c循环主块的一次可以有多少个数
    int64_t colLoop; // c循环尾块
    int64_t colLoopNumTail;
    int64_t alignNCNum;
    int64_t targetSize;
    int64_t targetCastSize;
    int64_t smoothSize;
    int64_t gradLossSize;
    int64_t ignoreSize;
    int64_t targetWeightSize;
    int64_t tBuf2Size;
    int64_t tBuf3Size;
    int64_t kTimes;
    int64_t cOnceNum;
    int64_t cOnceNumTail;
    int64_t kTimesTail;
    int64_t cOnceNumTailAlign;
    int64_t cacheStart;
};
 #endif
