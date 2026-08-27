/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

constexpr int64_t MAX_INPUT_SLOTS = 2;
constexpr int64_t MAX_OUTPUT_SLOTS = 1;
constexpr int64_t PHYS_NODES = 3;

struct SplitResult {
    int64_t axis;
    int64_t aI;
    int64_t aO;
    int64_t aITail;
};

struct MultiCoreResult {
    int64_t numCores;
    int64_t totalTiles;
    int64_t tilesMain;
    int64_t coresTail;
};

template <int64_t RANK>
struct AscendRequantTilingData {
    SplitResult split;
    MultiCoreResult multicore;
    int64_t rank;
    int64_t perBufBytes;
    int64_t maxBroShape[RANK];
    int64_t numInputs;
    int64_t numOutputs;
    int64_t reluFlag;
    int64_t inputShapes[MAX_INPUT_SLOTS][RANK];
    int64_t inputStrides[MAX_INPUT_SLOTS][RANK];
    int64_t outputShapes[MAX_OUTPUT_SLOTS][RANK];
    int64_t outputStrides[MAX_OUTPUT_SLOTS][RANK];
};
