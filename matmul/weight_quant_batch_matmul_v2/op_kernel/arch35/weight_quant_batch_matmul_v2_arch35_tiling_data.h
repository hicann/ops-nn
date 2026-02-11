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
 * \file weight_quant_batch_matmul_v2_arch35_tiling_data.h
 * \brief
 */

#ifndef ARCH35_WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H
#define ARCH35_WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H
#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace wqbmmv2_tiling {
#pragma pack(push, 8)
struct WeightQuantBatchMatmulV2RegBaseTilingDataParams {
    uint8_t cubeNumBlocksN = 0;
    uint8_t cubeNumBlocksM = 0;
    uint8_t vecCoreParallel = 0;
    uint8_t reserve1 = 0;

    uint16_t AL1Pingpong = 0;
    uint16_t BL1Pingpong = 0;

    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t groupSize = 0;
    uint64_t mSize = 0;
    uint64_t nBubSize = 0;
    uint64_t kBubSize = 0;
    AscendC::tiling::TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct WeightQuantBatchMatmulV2ASTilingDataParams {
    uint32_t cubeNumBlocksM = 0;
    uint32_t cubeNumBlocksN = 0;
    uint32_t hasBias = 0;
    uint32_t firstTailBlockCount = 0;
    uint32_t secondTailBlockCount = 0;
    uint32_t weightL2Cacheable = 0;
    uint32_t mainBlockL1Size = 0;
    uint32_t firstTailBlockL1Size = 0;
    uint32_t secondTailBlockL1Size = 0;
    uint32_t aPreloadSize = 0;
    uint64_t groupSize = 0;
    uint64_t mainBlockCount = 0;
    uint64_t mSize = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    AscendC::tiling::TCubeTiling matmulTiling;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct WeightQuantBatchMatmulV2BatchParams {
    uint64_t batchA = 0;
    uint64_t batchB = 0;
    uint64_t batchC = 0;
    uint64_t batchA1 = 0;
    uint64_t batchA2 = 0;
    uint64_t batchA3 = 0;
    uint64_t batchA4 = 0;
    uint64_t batchB1 = 0;
    uint64_t batchB2 = 0;
    uint64_t batchB3 = 0;
    uint64_t batchB4 = 0;
    uint64_t batchC1 = 0;
    uint64_t batchC2 = 0;
    uint64_t batchC3 = 0;
    uint64_t batchC4 = 0;
    uint64_t biasWithBatch = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct WeightQuantBatchMatmulV2ASWTilingDataParams {
    uint32_t hasBias = 0; // 预留参数，当前不支持有Bias的场景
    uint32_t mTailTile = 0;
    uint32_t nTailTile = 0;
    uint32_t reserved1 = 0;
    AscendC::tiling::TCubeTiling matmulTiling;
    WeightQuantBatchMatmulV2BatchParams params;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) WeightQuantBatchMatmulV2TilingData
{
    uint8_t vecBlockDimN;
    uint8_t vecBlockDimK;
    uint8_t cubeBlockDimN;
    uint8_t cubeBlockDimM;
    uint8_t cubeBlockDimK;
    uint8_t kPadSize;
    uint8_t nPadSize;
    uint8_t haveBatchA;
    uint8_t haveBatchB;
    uint8_t reserve1;
    uint8_t reserve2;
    uint8_t reserve3;
    uint16_t vecSingleKGroupNum;
    uint16_t vecSingleKTailGroupNum;
    uint16_t AL1Pingpong;
    uint16_t BL1Pingpong;
    uint32_t vecSingleK;
    uint32_t vecSingleN;
    uint32_t vec2SingleM;
    uint32_t vecSingleKTail;
    uint32_t vecSingleNTail;
    uint32_t wInQueueSize;
    uint32_t offsetInQueueSize;
    uint32_t scaleInQueueSize;
    uint32_t wOutQueueSize;
    uint32_t antiQuantTmpBufferSize;
    uint32_t vecCubeNRatio;
    uint32_t vecCubeTailNRatio;
    uint32_t vecCubeKRatio;
    uint32_t vecCubeTailKRatio;
    uint32_t cubeTailM;
    uint32_t cubeTailN;
    uint32_t cubeSingleNLoop;
    uint32_t cubeSingleNTailLoop;
    uint32_t repeatAxisMax;
    uint64_t vecSingleKLoop;
    uint64_t vecSingleNLoop;
    uint64_t vecSingleKTailLoop;
    uint64_t vecSingleNTailLoop;
    uint64_t vec2SingleMLoop;
    uint64_t kAlign;
    uint64_t nAlign;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t groupSize;
    uint64_t mSize;
    uint64_t blockBatch;
    uint64_t shapeBatch;
    uint64_t mAubSize;
    uint64_t kAubSize;
    uint64_t nBubSize;
    uint64_t kBubSize;
    uint64_t mCubSize;
    uint64_t nCubSize;
    uint64_t mAL1Size;
    uint64_t kAL1Size;
    uint64_t nBL1Size;
    uint64_t kBL1Size;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)
} // namespace wqbmmv2_tiling
#endif // ARCH35_WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H
