/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _TEST_MATMUL_V3_TILING_TILING_H_
#define _TEST_MATMUL_V3_TILING_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

struct L2cacheUseInfo {
  uint32_t l2CacheFlag;
};

struct L2cacheTilePara {
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
};

struct MatMulRunInfo {
    uint32_t transA;
    uint32_t transB;
    uint32_t nd2nzA;
    uint32_t nd2nzB;
    uint32_t isNzA;
    uint32_t isNzB;
    uint32_t isHf32;
};

struct MatmulTilingData {
    int32_t usedCoreNum;
    int32_t M;
    int32_t N;
    int32_t Ka;
    int32_t Kb;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t singleCoreK;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t depthA1;
    int32_t depthB1;
    int32_t stepM;
    int32_t stepN;
    int32_t stepka;
    int32_t stepkb;
    int32_t depthAL1CacheUB;
    int32_t depthBL1CacheUB;
    int32_t dbL0A;
    int32_t dbL0B;
    int32_t dbL0C;
    int32_t isBias;
    int32_t transLength;
    int32_t iterateOrder;
    int32_t usedL1Size;
    int32_t usedL0CSize;
    int32_t usedUBSize;
    int32_t batchM;
    int32_t batchN;
    int32_t singleBatchM;
    int32_t singleBatchN;
    TCubeTiling matmulTiling;
    L2cacheTilePara tileL2cacheTiling;
    MatMulRunInfo matmulRunInfo;
    L2cacheUseInfo l2cacheUseInfo;
    uint32_t baseAN;
    uint32_t baseAD;
    uint32_t baseBN;
    uint32_t baseBD;
};

inline void InitMatmulTilingData(uint8_t* tiling, MatmulTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MatmulTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    MatmulTilingData tiling_data;                                                 \
    InitMatmulTilingData(tiling_arg, &tiling_data)
#endif  // FOREACH_MINIMUM_SCALAR_TILING_DEF_H