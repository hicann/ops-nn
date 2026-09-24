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
 * \file unique_sort_tiling_data.h
 * \brief unique_sort_tiling_data.h
 */

#ifndef UNIQUE_SORT_TILING_DATA_H
#define UNIQUE_SORT_TILING_DATA_H

// Match Sort's resident/batched merge policy constants.
constexpr uint32_t SORT_BATCH_MERGE_MIN_ROWS = 4;
constexpr uint32_t SORT_RESIDENT_MERGE_MIN_BLOCKS = 2U;
constexpr uint32_t SORT_RESIDENT_MERGE_MAX_BLOCKS = 3U;

struct UniqueSortRegBaseTilingData {
    uint32_t numTileDataSize;     // elements per UB tile
    uint32_t unsortedDimParallel; // parallel b-axis slices/cores
    uint32_t lastDimTileNum;      // radix: h-axis tile count; multi-core merge: output axis length (elements)
    uint32_t sortLoopTimes;       // b-axis loop count
    uint32_t lastDimNeedCore;     // h-axis core count
    // Schedule-specific fields; each schedule interprets its own units.
    // radix: globalHistGmWk_ core count; radix-one-core: input queue bytes;
    // single-core merge: elements per window; multi-core merge: elements per window;
    // AxisOneCopy: elements per copy loop.
    uint32_t keyParams0;
    // radix: 清零 excusiveBinsGmWk_ 的核
    // radix_one_core: y2OutQue 需要的 ub 大小
    // merge: xQue ub 大小
    uint32_t keyParams1;
    // radix: 清零的一次 ub 数据量
    // radix_one_core: 输出 int64 时，一半的 ub 偏移
    // merge: y2OutQue 的 ub 大小
    // small-axis two-stage: rank-inverse 标志
    uint32_t keyParams2;
    // radix: 清零 globalHistGmWk_ ub 循环次数
    // radix_one_core: 队列 buffer 数
    // merge: 32 个数对齐的 alginH
    // small-axis insertion/two-stage: non-last-axis flag
    uint32_t keyParams3;
    // radix: 清零 chunk 大小
    // merge_sort(sch0): 队列 buffer 数
    // intra_core(sch4): extract chunk 大小
    // non-last small-axis radix(sch10): phase-shared UB layout flag
    // non-last small-axis two-stage(sch11): outer slices per batch (0 means ordinary mapping)
    uint32_t keyParams4;
    uint32_t keyParams5;    // radix: clear chunk elements; intra-core merge: maximum merge iterations
    uint32_t tmpUbSize;     // sort高级api需要的临时ub大小
    int64_t lastAxisNum;    // h轴大小
    int64_t unsortedDimNum; // b轴大小
    // Non-last small-axis fields describe the original [outer, axis, inner] GM layout
    // and the aligned UB row strides used by tile-local transpose schedules.
    int64_t outerSize;
    int64_t innerSize;
    uint32_t innerLoopNum;
    uint32_t innerChunk;
    uint32_t inputRowBytes;
    uint32_t valueAxisBytes;
    uint32_t indexAxisBytes;
    uint32_t outputIndexRowBytes;
};
#endif // UNIQUE_SORT_TILING_DATA_H
