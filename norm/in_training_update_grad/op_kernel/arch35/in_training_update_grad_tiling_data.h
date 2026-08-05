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
struct InTrainingUpdateGradStreamTilingData {
    uint32_t numC1;
    uint32_t numD;
    uint32_t numHW;    // H * W
    uint32_t numC0;    // 16
    uint32_t reduceR;  // D * H * W
    uint32_t groupNum; // N * C1
    uint32_t usedCoreNum;
    uint32_t perCoreGroups;
    uint32_t blockLenElem;   // H * W * C0
    uint32_t streamTileRows; // rows (in H*W units) processed per chunk inside a D-slice
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
