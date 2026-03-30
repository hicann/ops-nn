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
 * \file scatter_update_common.h
 * \brief scatter_update
 */
#ifndef ASCENDC_SCATTER_STRUCT_H_
#define ASCENDC_SCATTER_STRUCT_H_

class ScatterUpdateTilingData {
public:
    uint64_t varShape[2];
    uint64_t indicesSize;
    uint64_t normBlockIndices;
    uint64_t tailBlockIndices;
    uint64_t indicesFactor;
    uint64_t normBlockLoop;
    uint64_t tailBlockLoop;
    uint64_t normBlockTail;
    uint64_t tailBlockTail;
    uint64_t rowTotal;
    uint64_t colTotal;
    uint64_t rowBase;
    uint64_t colBase;
    uint64_t rowTail;
    uint64_t colTail;
    uint64_t rowTileNum;
    uint64_t colTileNum;
    uint64_t usedCoreNum;
    uint64_t normBlockColNum;
    uint64_t normBlockRowNum;
    uint64_t tailBlockColNum;
    uint64_t tailBlockRowNum;
    uint64_t normNeedSplitRow;
    uint64_t tailNeedSplitRow;
    uint64_t processRowPerUb;
    uint64_t processColNum;
    uint64_t rowLoopByUb;
    uint64_t processRowTail;
    uint64_t indicesUbFactor;
    uint64_t updateUbSize;
    uint64_t processColPerUb;
    uint64_t colLoopByUb;
    uint64_t processColTail;
    uint64_t indicesBatchCopySizeAlign;
    uint64_t varStride;
    uint64_t updateColUbFactor;
    uint64_t indicesLoopSize;
    uint64_t indicesTailLoopNum;
    uint64_t updatesNormBlockLoop;
    uint64_t updatesTailBlockLoop;
    uint64_t updatesNormBlockTailLoopSize;
    uint64_t updatesTailBlockTailLoopSize;
    uint64_t maskNormBlockLen;
    uint64_t maskTailBlockLen;
    bool isIndicesSizeInt64;
    uint64_t indicesCastMode;
};

#endif