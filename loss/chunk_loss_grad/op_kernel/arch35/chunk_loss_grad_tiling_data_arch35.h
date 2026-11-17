/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct ChunkLossGradArch35TilingData {
    uint64_t numTokens;
    uint64_t hiddenDim;
    uint64_t vocabSize;
    uint32_t coreNum;
    uint32_t tokenTile;
    uint32_t vocabTile;
    uint32_t vectorChunk;
    uint64_t logitsBytes;
    uint64_t logitsGradBytes;
    uint64_t partialBytes;
    uint64_t coreWorkspaceBytes;
    AscendC::tiling::TCubeTiling logitsTiling;
    AscendC::tiling::TCubeTiling weightTiling;
    uint32_t scheduleMode;
    uint32_t tokenPitch;
    uint32_t cacheElements;
    uint32_t residentAccum;
    uint32_t accumElements;
    uint32_t slotCount;
    int64_t ignoreIndex;
    float alpha;
    uint32_t reserved;
};
