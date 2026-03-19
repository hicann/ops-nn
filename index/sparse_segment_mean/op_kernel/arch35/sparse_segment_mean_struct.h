/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_segment_mean_struct.h
 * \brief tiling base data
 */

#ifndef SPARSE_SEGMENT_MEAN_STRUCT_H
#define SPARSE_SEGMENT_MEAN_STRUCT_H

class SparseSegmentMeanSimtTilingData
{
public:
    int64_t needCoreNum{0};
    int64_t innerSize{0};
    int64_t gatherSize{0};
    int64_t segmentNum{0};
    int64_t outterSize{0};
    int64_t threadNumX{0};
    int64_t threadNumY{0};
    int64_t perCoreSegmentNum{0};
    int64_t resSegmentNum{0};
    int64_t normalCoreSegmentNum{0}; // 正常核处理的SegmentNum数
    int64_t secondToLastCoreSegmentNum{0};     // 倒数第二个核处理的SegmentNum数
    int64_t lastCoreSegmentNum{0}; // 尾核处理的SegmentNum数
    int64_t specialBlockTiling{0}; // 特殊分核
};

class SparseSegmentMeanSimdTilingData
{
public:
    int64_t tilingkey{0};
    int64_t usedCoreNum{0};
    int64_t innerSize{0};
    int64_t gatherSize{0};
    int64_t xBufferSize{0};
    int64_t yBufferSize{0};
    int64_t sharedTmpBufferSize{0};
    int64_t normalCoreInnerNum{0};
    int64_t tailCoreInnerNum{0};
    int64_t innerOuter{0};
    int64_t normalCoreIndicesNum{0};
    int64_t tailCoreIndicesNum{0};
    int64_t indicesOuter{0};
    int64_t perCoreInnerElements{0};
    int64_t tailCoreInnerElements{0};
    int64_t normalCoreProcessNumForClear{0};
    int64_t tailCoreProcessNumForClear{0};
    int64_t usedCoreNumForClear{0};
    int64_t inBufferSize{0};
    int64_t outBufferSize{0};
    int64_t usedCoreNumForMulCore{0};
    int64_t workspaceBufferSize{0};
};

class SparseSegmentMeanFullLoadTilingData
{
public:
    int64_t innerSize{0};         // 输入非索引轴的元素个数
    int64_t outterSize{0};        // 输入indices的元素个数
    int64_t segmentNum{0};        // 输入segment的元素个数
    int64_t gatherSize{0};        // 索引轴的个数
    int64_t xBufferSize{0};       // x的数据总量
    int64_t indicesBufferSize{0}; // indices每核的总数据量
    int64_t threadNumX{0};        // innerSize使用的线程
    int64_t threadNumY{0};        // 计算indices使用的线程
    int64_t perCoreSegmentNum{0}; // segment每核的元素个数
    int64_t resSegmentNum{0};     // segment剩余的元素个数
    int64_t needCoreNum{0};          // 开的核数
    int64_t normalCoreSegmentNum{0}; // 正常核处理的SegmentNum数
    int64_t secondToLastCoreSegmentNum{0};     // 倒数第二个核处理的SegmentNum数
    int64_t lastCoreSegmentNum{0}; // 尾核处理的SegmentNum数
    int64_t specialBlockTiling{0}; // 特殊分核
};

#endif