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
 * \file gather_elements_v2_tiling_def.h
 * \brief
 */

#ifndef GATHER_ELEMENTS_V2_TILING_DEF_H
#define GATHER_ELEMENTS_V2_TILING_DEF_H

#include "kernel_tiling/kernel_tiling.h"

#define __aicore__
#define DTYPE_X float

struct GatherElementsV2TilingParam {
    uint64_t xPreDim = 1;
    uint64_t xGatherDim = 1;
    uint64_t xPostDim = 1;
    uint64_t idxPreDim = 1;
    uint64_t idxGatherDim = 1;
    uint64_t idxPostDim = 1;

    uint64_t coreGroupNum = 0;
    uint64_t formerGroupNum = 0;
    uint64_t tailGroupNum = 0;
    uint64_t formerGroupPreDim = 0;
    uint64_t tailGroupPreDim = 0;
    uint64_t formerGroupCoreNum = 0;
    uint64_t tailGroupCoreNum = 0;

    uint64_t formerGroupFormerNum = 0;
    uint64_t formerGroupTailNum = 0;
    uint64_t formerGroupFormerPostDim = 0;
    uint64_t formerGroupTailPostDim = 0;
    uint64_t tailGroupFormerNum = 0;
    uint64_t tailGroupTailNum = 0;
    uint64_t tailGroupFormerPostDim = 0;
    uint64_t tailGroupTailPostDim = 0;
};

struct GatherElementsV2TransTiling {
    uint64_t carryNumAlign = 0;
    uint64_t xCarryNumAlign = 0;
    uint64_t idxCarryNumAlign = 0;
    uint64_t inBufferSize = 0;
    uint64_t outBufferSize = 0;
    uint64_t transGatherDimSlice = 0;
    uint64_t idxGatherDimSlice = 0;
    uint64_t workspacePerBlock = 0;
};

struct GatherElementsV2ScalarTiling {
    uint64_t formerGroupFormerData = 0;
    uint64_t formerGroupTailData = 0;
    uint64_t tailGroupFormerData = 0;
    uint64_t tailGroupTailData = 0;
    uint64_t maxIdxDataAlign = 0;
};

struct GatherElementsV2LastDimTilingParam {
    int64_t xShape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexShape[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t xStrideArray[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexStrideArray[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t dimNum = 0;
    int64_t specialDataMove = 0;
    int64_t xSliceNum = 0;
    int64_t indexSliceNum = 0;
    int64_t reservedXSize = 0;
    int64_t reservedIndexSize = 0;
    int64_t indexAxisSizeEqualOne = 0;
    int64_t scalarMode = 0;
    int64_t formerCoreRowNum = 0;
    int64_t formerCoreNum = 0;
    int64_t eachCalculationLines = 0;
    int64_t xBufferSize = 0;
    int64_t indexBufferSize = 0;
    int64_t yBufferSize = 0;
    int64_t maskBufferSize = 0;
    int64_t scalarModeLength = 0;
    int64_t dataMoveUBStride = 0;
};

struct GatherElementsV2TilingData {
    GatherElementsV2TilingParam params;
    GatherElementsV2TransTiling transTiling;
    GatherElementsV2ScalarTiling scalarTiling;
    GatherElementsV2LastDimTilingParam lastDimTiling;
};

inline void InitGatherElementsV2TilingData(uint8_t* tiling, GatherElementsV2TilingData* data)
{
    memcpy(data, tiling, sizeof(GatherElementsV2TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    GatherElementsV2TilingData tiling_data;      \
    InitGatherElementsV2TilingData(tiling_arg, &tiling_data)
#endif // GATHER_ELEMENTS_V2_TILING_DEF_H