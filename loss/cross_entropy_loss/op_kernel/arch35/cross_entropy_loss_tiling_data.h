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
 * \file cross_entropy_loss_tiling_data.h
 * \brief cross_entropy_loss_tiling_data.h
 */

#ifndef _CROSS_ENTROPY_LOSS_REGBASE_TILING_DATA_H_
#define _CROSS_ENTROPY_LOSS_REGBASE_TILING_DATA_H_

struct CrossEntropyLossRegBaseTilingData {
    int64_t realCoreNum;
    int64_t frontCoreNum;
    int64_t frontCoreNSize;
    int64_t onceNSize;
    int64_t onceNSizeT;
    int64_t nLoopTimesB;
    int64_t nLoopTimesT;
    int64_t nTailNumB;
    int64_t nTailNumT;
    int64_t ignoreIndex;
    int64_t cLoopTimes;   // c循环的主块次数，
    int64_t cOnceNum;     // c循环主块的一次可以有多少个数
    int64_t cOnceNumTail; // c循环尾块
    int64_t C;
    int64_t kTimesTail; // 二分累加时的尾块的循环次数
    int64_t kTimes;     // 二分累加2的k次方内循环次数
    int64_t cacheStart; // 缓存更新时最大的位置
    int64_t db;
    float labelSmooth1; // 1.0f -labelsmoothing
    float labelSmoothC; // labelsmoothing / C
};
#endif
