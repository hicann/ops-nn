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
 * \file gather_elements_v2_tiling_data.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_V2_TILING_DATA_H
#define GATHER_ELEMENTS_V2_TILING_DATA_H

#include <cstdint>

struct GatherElementsV2TilingParam {
    uint64_t xPreDim;
    uint64_t xGatherDim;
    uint64_t xPostDim;
    uint64_t idxPreDim;
    uint64_t idxGatherDim;
    uint64_t idxPostDim;

    uint64_t coreGroupNum;
    uint64_t formerGroupNum;
    uint64_t tailGroupNum;

    uint64_t formerGroupPreDim;
    uint64_t tailGroupPreDim;
    uint64_t formerGroupCoreNum;
    uint64_t tailGroupCoreNum;

    uint64_t formerGroupFormerNum;
    uint64_t formerGroupTailNum;
    uint64_t formerGroupFormerPostDim;
    uint64_t formerGroupTailPostDim;

    uint64_t tailGroupFormerNum;
    uint64_t tailGroupTailNum;
    uint64_t tailGroupFormerPostDim;
    uint64_t tailGroupTailPostDim;
};

struct GatherElementsV2TransTiling {
    uint64_t carryNumAlign;
    uint64_t xCarryNumAlign;
    uint64_t idxCarryNumAlign;

    uint64_t inBufferSize;
    uint64_t outBufferSize;
    uint64_t transGatherDimSlice;
    uint64_t idxGatherDimSlice;

    uint64_t workspacePerBlock;
};

struct GatherElementsV2ScalarTiling {
    uint64_t formerGroupFormerData;
    uint64_t formerGroupTailData;
    uint64_t tailGroupFormerData;
    uint64_t tailGroupTailData;
    uint64_t maxIdxDataAlign;
};

struct GatherElementsV2LastDimTilingParam {
    int64_t xShape[8];
    int64_t indexShape[8];
    int64_t xStrideArray[8];
    int64_t indexStrideArray[8];

    int64_t dimNum;
    int64_t specialDataMove;
    int64_t xSliceNum;
    int64_t indexSliceNum;
    int64_t reservedXSize;
    int64_t reservedIndexSize;

    int64_t indexAxisSizeEqualOne;
    int64_t scalarMode;
    int64_t formerCoreRowNum;
    int64_t formerCoreNum;
    int64_t eachCalculationLines;

    int64_t xBufferSize;
    int64_t indexBufferSize;
    int64_t yBufferSize;
    int64_t maskBufferSize;
    int64_t scalarModeLength;

    int64_t dataMoveUBStride;
};

struct GatherElementsV2TilingData {
    GatherElementsV2TilingParam params;
    GatherElementsV2TransTiling transTiling;
    GatherElementsV2ScalarTiling scalarTiling;
    GatherElementsV2LastDimTilingParam lastDimTiling;
};

#endif // GATHER_ELEMENTS_V2_TILING_DATA_H
