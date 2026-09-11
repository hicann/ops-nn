/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_TILING_DATA_H
#define KTH_VALUE_TILING_DATA_H

#include <cstdint>

constexpr uint32_t MEDIAN_MODE_STATIC = 0U;
constexpr uint32_t MEDIAN_MODE_PROPAGATE_NAN = 1U;
constexpr uint32_t MEDIAN_MODE_IGNORE_NAN = 2U;

struct KthValueTilingData {
    uint32_t numTileDataSize;     // h轴ub一次处理个数
    uint32_t unsortedDimParallel; // b轴使用的核数
    uint32_t lastDimTileNum;      // h轴循环次数
    uint32_t sortLoopTimes;       // b轴循环次数
    uint32_t lastDimNeedCore;     // h轴需要的核数
    // keyParamsxxx 预留参数
    // radix: globalHistGmWk_ 使用核数
    // radix_one_core: inqueX 的 ub 大小
    // merge_sort_one_core: 一次处理几个 h
    // merge_more_core: 最大处理元素数
    // intra_core: batchPerCore_
    // small_axis_insertion/two_stage: batchSize_
    // non_last_small_axis: sortCount_
    // axis_one_copy: copyElemsPerLoop_
    uint32_t keyParams0;
    // radix: 清零 excusiveBinsGmWk_ 的核
    // radix_one_core: idxUbSize_ (y2OutQue ub 大小)
    // merge_sort_one_core: xQue ub 大小
    // small_axis_insertion/two_stage: batchNum_
    // non_last_small_axis: inputValueAxisBytes_ (bf16 merge)
    // axis_one_copy: loopTimes_
    uint32_t keyParams1;
    // radix: 清零的一次 ub 数据量
    // merge_sort_one_core: y2OutQue 的 ub 大小
    // small_axis_insertion/two_stage: rank-inverse 标志
    uint32_t keyParams2;
    // radix: 清零 globalHistGmWk_ ub 循环次数
    // radix_one_core: 队列 buffer 数
    // merge_sort_one_core: 32 个数对齐的 alignSize_
    // intra_core: alignNum_
    uint32_t keyParams3;
    // radix: 清零 excusiveBinsGmWk_ chunk 大小
    // radix_one_core: outputRowsPerLoop_
    // merge_sort_one_core: 队列 buffer 数
    // intra_core: extractChunkSize_
    uint32_t keyParams4;
    // radix: 清零 globalHistGmWk_ chunk 大小
    // intra_core: 最大归并迭代次数
    uint32_t keyParams5;
    uint32_t tmpUbSize;      // 高级api需要的临时ub大小
    int64_t kthIndex;        // 零基kth偏移，避免按k展开binary
    int64_t lastAxisNum;     // h轴大小
    int64_t unsortedDimNum;  // b轴大小
    int64_t outerSize;       // non-last-axis: 外层维度大小
    int64_t innerSize;       // non-last-axis: 内层维度大小
    uint32_t innerLoopNum;   // non-last-axis: inner分块循环次数
    uint32_t innerChunk;     // non-last-axis: inner分块大小
    uint32_t inputRowBytes;  // non-last-axis: 输入一行字节数
    uint32_t valueAxisBytes; // non-last-axis: value轴字节数
    uint32_t indexAxisBytes; // non-last-axis: index轴字节数
    // 0: KthValue, 1: Median (propagate NaN), 2: NanMedian (ignore NaN).
    uint32_t medianMode;
};

#endif
