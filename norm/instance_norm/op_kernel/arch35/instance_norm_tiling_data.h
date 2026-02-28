/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file instance_norm_tiling_data.h
 * \brief
 */
#ifndef _INSTANCE_NORM_TILING_DATA_H_
#define _INSTANCE_NORM_TILING_DATA_H_

struct InstanceNormARFullReduceTilingData {
    uint64_t numN;
    uint64_t numC;
    uint64_t numR;
    uint64_t rAlign;
    uint64_t cInner;
    uint64_t cOuter;
    uint64_t cTail;             // c轴上尾部长度
    uint64_t binaryAddQuotient; // 小于numR的最大二次幂
    uint64_t perCoreCnt;        // 平均每个核处理的块数
    float epsilon;
    float avgFactor;
};

struct InstanceNormARWelfordTilingData {
    uint64_t a1;            // 在AR Pattern下输入tensor的N轴
    uint64_t a0;            // 在AR Pattern下输入tensor的C轴
    uint64_t r;             // 输入tensor的列，即reduce的轴
    uint64_t blockNum;      // 实际使用的core数量
    uint64_t totalTiles;
    uint64_t tilesPerCore;
    uint64_t a0Outer;
    uint64_t a0Inner;
    uint64_t a0Tail;
    uint64_t welfordTileLength;         // tile块的元素个数
    uint64_t welfordTempSize;    // welford临时buffer的大小
    uint64_t welfordUpdateTimes; // welford update的次数
    uint64_t welfordUpdateTail;  // welford update的尾数
    uint64_t apiTempBufferSize;
    float epsilon;
};

struct InstanceNormARAFullReduceTilingData {
    int64_t usedCoreNum;
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t totalA1Len;
    int64_t totalRLen;
    int64_t totalA0Len;
    int64_t a0Outer;
    uint64_t tileA0Len;
    uint64_t tileA0Tail;
    uint64_t powerOfTwoForR; // 小于r的最大的2的幂次
    uint64_t binaryAddQuotient;
    uint64_t binaryAddK;
    uint64_t binaryAddLast;
    float epsilon;
};

struct InstanceNormARAWelfordTilingData {
    int64_t a1;
    int64_t r;
    int64_t a0;
    int64_t usedCoreNum;
    int64_t totalTiles;
    int64_t tilesPerCore;
    int64_t a0Outer;
    uint64_t tileA0Len;
    uint64_t tileA0Tail; 
    uint64_t welfordrFactor;         // welford的迭代段的长度
    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLast;
    int64_t powerOfTwoForR;
    float epsilon;
};

struct InstanceNormReduceEmptyTilingData {
    uint64_t perCoreElements;          // 每个核要处理的元素个数（头核）
    uint64_t lastCoreElements;         // 尾核要处理的元素个数
    uint64_t perCoreLoops;             // 每个核的循环数（头核）
    uint64_t perCorePerLoopElements;   // 每个核每次循环处理的元素数（头核头部循环）
    uint64_t perCoreLastLoopElements;  // 每个核末次循环处理的元素数（头核末次循环）
    uint64_t lastCoreLoops;            // 尾核的循环数
    uint64_t lastCorePerLoopElements;  // 尾核每次循环处理的元素数（头部循环）
    uint64_t lastCoreLastLoopElements; // 尾核末次循环处理的元素数（末次循环）
};

#endif
