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
 * \file in_training_update_grad_tiling_data.h
 * \brief
 */
#ifndef IN_TRAINING_UPDATE_GRAD_TILING_DATA_H_
#define IN_TRAINING_UPDATE_GRAD_TILING_DATA_H_

#include <cstdint>

// Tiling keys: the single source of truth shared by the host tiling (GetTilingKey) and the
// kernel entry dispatch. Both sides include this header, so the values cannot drift apart;
// a mismatched binary would previously dispatch on independently maintained literals.
// The AscendC precompilation phase only accepts numeric constants or macros inside
// TILING_KEY_IS(), so the values are exposed as macros; the namespace constants below are
// derived from the same macros for host-side (non-TILING_KEY_IS) use.
#define IN_UG_TILING_KEY_REDUCE_EMPTY 50000
#define IN_UG_TILING_KEY_FULL_LOAD 100000
#define IN_UG_TILING_KEY_STREAM 200000

namespace InTrainingUpdateGradKeys {
constexpr uint64_t REDUCE_EMPTY = IN_UG_TILING_KEY_REDUCE_EMPTY;
constexpr uint64_t FULL_LOAD = IN_UG_TILING_KEY_FULL_LOAD;
constexpr uint64_t STREAM = IN_UG_TILING_KEY_STREAM;
static_assert(REDUCE_EMPTY != FULL_LOAD && REDUCE_EMPTY != STREAM && FULL_LOAD != STREAM,
              "in_training_update_grad tiling keys must be distinct");
} // namespace InTrainingUpdateGradKeys

// FullLoad (TilingKey 100000): the whole spatial block (R*C0) of one (n,c1) group fits UB.
struct InTrainingUpdateGradFullLoadTilingData {
    uint32_t numC1;
    uint32_t numD;
    uint32_t numHW;         // H * W
    uint32_t numC0;         // 16
    uint32_t reduceR;       // D * H * W
    uint32_t groupNum;      // N * C1
    uint32_t usedCoreNum;   // actually used AIV cores
    uint32_t perCoreGroups; // groups handled by a head core
    uint32_t blockLenElem;  // H * W * C0 (one D-slice, contiguous)
    float epsilon;
};

// Stream (TilingKey 200000): R*C0 exceeds UB; accumulate per D-slice in row chunks.
// 结构体必须 <= 框架分配的最大 tiling 容量(register default 的 FullLoad 结构 + 8B tiling key = 48B):
// 因此 dPerSplit / splitNum / perCoreTasks / c0Aligned 均由 kernel 从 numD/groupNum/usedCoreNum/numC0
// 推导, 只下发 dSplits 与 dBatch 两个新增量。
struct InTrainingUpdateGradStreamTilingData {
    uint32_t numC1;
    uint32_t numD;
    uint32_t numHW;    // H * W
    uint32_t numC0;    // 16
    uint32_t groupNum; // N * C1
    uint32_t usedCoreNum;
    uint32_t blockLenElem;   // H * W * C0
    uint32_t streamTileRows; // rows that fit one dy/x UB chunk (>= numHW when the row path is used)
    uint32_t dSplits;        // D 方向切分数（每个 group 由 dSplits 个核分摊）
    uint32_t dBatch;         // 单条 strided DMA 聚合的 D 切片数（0 表示走逐 d 行块路径）
    float epsilon;
};

// ReduceEmpty (TilingKey 50000): R == 0 (empty spatial); write 0.0 to both outputs (SUM semantics).
// Splits the N*C1*C0 output elements across cores / loops (same pattern as instance_norm).
struct InTrainingUpdateGradReduceEmptyTilingData {
    uint32_t perCoreElements;
    uint32_t lastCoreElements;
    uint32_t perCoreLoops;
    uint32_t perCorePerLoopElements;
    uint32_t perCoreLastLoopElements;
    uint32_t lastCoreLoops;
    uint32_t lastCorePerLoopElements;
    uint32_t lastCoreLastLoopElements;
};

#endif // IN_TRAINING_UPDATE_GRAD_TILING_DATA_H_
