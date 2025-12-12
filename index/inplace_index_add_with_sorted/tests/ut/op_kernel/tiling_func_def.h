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
 * \file tiling_func_def.h
 * \brief
 */

#ifndef __TILING_FUNCTION_DEF_H__
#define __TILING_FUNCTION_DEF_H__

#include <iostream>
#include <string>
#include <cstdint>
#include <map>
#include <cmath>
#include "inplace_index_add_with_sorted_tiling_def.h"
#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "graph/ge_error_codes.h"

namespace optiling {
const int32_t FLOAT32_TILING_KEY = 1;
const int32_t FLOAT16_TILING_KEY = 2;
const int32_t BF16_TILING_KEY = 3;
const int32_t INT16_TILING_KEY = 4;
const int32_t INT32_TILING_KEY = 5;
const int32_t FLOAT32_FIX_TILING_KEY = 6;
const int32_t OTHER_DIM_TILING_KEY = 100;
const int32_t FLOAT32_OTHER_DIM_TILING_KEY = 101;
const int32_t FLOAT16_OTHER_DIM_TILING_KEY = 102;
const int32_t BF16_OTHER_DIM_TILING_KEY = 103;
const int32_t INT16_OTHER_DIM_TILING_KEY = 104;
const int32_t INT32_OTHER_DIM_TILING_KEY = 105;

const int32_t SIZE_OF_FP16 = 2;
const int32_t SIZE_OF_FP32 = 4;
const int32_t SIZE_OF_INT32 = 4;
const int32_t SIZE_OF_BF16 = 2;

const int32_t INPUT_0 = 0;
const int32_t INPUT_1 = 1;
const int32_t INPUT_2 = 2;
const int32_t INPUT_3 = 3;
const int32_t INPUT_4 = 4;
const int32_t BUF_CNT_2 = 2;
const int32_t BUF_CNT_3 = 3;
const int32_t BUF_CNT_6 = 6;
const int32_t BUF_CNT_7 = 7;
const int32_t BLOCK_SIZE = 32;
const uint32_t UB_INDEX_NUM = 2048;
const uint32_t INDEX_BUFFER_SIZE = UB_INDEX_NUM * 2 * SIZE_OF_INT32;
const uint32_t RESERVED_BUFFER_SIZE = 1024;
static const std::map<int32_t, int32_t> DTYPE_BUF_CNT_MAP = {
    {FLOAT32_TILING_KEY, BUF_CNT_2},
    {FLOAT16_TILING_KEY, BUF_CNT_7},
    {BF16_TILING_KEY, BUF_CNT_7},
    {INT16_TILING_KEY, BUF_CNT_2},
    {INT32_TILING_KEY, BUF_CNT_2},
    {FLOAT32_FIX_TILING_KEY, BUF_CNT_3},
    {FLOAT32_OTHER_DIM_TILING_KEY, BUF_CNT_2},
    {FLOAT16_OTHER_DIM_TILING_KEY, BUF_CNT_6},
    {BF16_OTHER_DIM_TILING_KEY, BUF_CNT_6},
    {INT16_OTHER_DIM_TILING_KEY, BUF_CNT_2},
    {INT32_OTHER_DIM_TILING_KEY, BUF_CNT_2}};

struct InplaceIndexAddWithSortedCompileInfo {
    uint32_t totalCoreNum = 40;
    int64_t ubSizePlatForm = 0;
};

class InplaceIndexAddWithSortedTilingDef
{
public:
    InplaceIndexAddWithSortedTilingDef(
        const std::vector<std::vector<int64_t>> shapesInfo, const std::vector<int64_t> attsInfo,
        ge::DataType varDtypeIn = ge::DT_FLOAT)
        : dimAttr(attsInfo[0])
    {
        varDtype = varDtypeIn;
        inputSize = 4;
        if (varDtype == ge::DT_FLOAT16 || varDtype == ge::DT_BF16 || varDtype == ge::DT_INT16) {
            inputSize = 2;
        }

        for (const auto& dim : shapesInfo[0]) {
            varShape.AppendDim(dim);
        }
        for (const auto& dim : shapesInfo[1]) {
            valueShape.AppendDim(dim);
        }
        for (const auto& dim : shapesInfo[2]) {
            idxShape.AppendDim(dim);
        }
        for (const auto& dim : shapesInfo[2]) {
            posShape.AppendDim(dim);
        }
    };

    ge::graphStatus RunKernelTiling();
    void TilingDataSet(uint8_t* tilingPtr);
    int32_t GetTilingKey()
    {
        return static_cast<int32_t>(tilingKey);
    };
    int32_t GetNeedCoreNum()
    {
        return static_cast<int32_t>(usedCoreNum);
    };

private:
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b)
    {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a + b - 1) / b);
    };

    template <typename T1, typename T2>
    inline T1 AlignA2B(T1 a, T2 b)
    {
        a = int64_t(a);
        b = int64_t(b);
        return T1(b == 0 ? a : (a / b) * b);
    };
    void Init();
    void processFirstDimTilingData();

private:
    InplaceIndexAddWithSortedTilingData tilingData;
    const InplaceIndexAddWithSortedCompileInfo ptrCompileInfo = {48, 196608};

    gert::Shape varShape;
    gert::Shape valueShape;
    gert::Shape idxShape;
    gert::Shape posShape;
    int64_t dimAttr = -1;
    ge::DataType varDtype = ge::DT_UNDEFINED;
    int32_t dtypeSize = 0;

    uint32_t tilingKey = 0;
    int64_t ubSize = 1;
    uint32_t inputCount = 1;
    uint32_t updatesCount = 1;
    uint32_t indicesCount = 1;
    uint32_t updatesOneTime = 1;
    uint32_t inputSize = 1;
    uint32_t coreNum = 1;

    uint32_t usedCoreNum = 1;
    uint32_t enableAlpha = 1;
    uint32_t eachIndexCount = 1;
    uint32_t lastIndexCount = 1;
    uint32_t maxSize = 0;
    uint32_t eachNum = 1;
    uint32_t eachLoop = 1;
    uint32_t eachTail = 0;
    uint32_t batchNum = 1;
    uint32_t eachBatchNum = 1;
    uint32_t lastBatchNum = 1;
    uint32_t varDimNum = 1;
    uint32_t eachUBIndexRound = 1;
    uint32_t eachUBIndexCount = 1;
    uint32_t eachUBIndexTail = 1;
    uint32_t lastUBIndexRound = 1;
    uint32_t lastUBIndexCount = 1;
    uint32_t lastUBIndexTail = 1;
    uint32_t workspaceSize = 1024 * 1024 * 16;
};

void InplaceIndexAddWithSortedTilingDef::Init()
{
    coreNum = ptrCompileInfo.totalCoreNum;
    ubSize = ptrCompileInfo.ubSizePlatForm - INDEX_BUFFER_SIZE - RESERVED_BUFFER_SIZE;

    size_t xDimNum = varShape.GetDimNum();
    dimAttr = dimAttr < 0 ? xDimNum + dimAttr : dimAttr;
}

ge::graphStatus InplaceIndexAddWithSortedTilingDef::RunKernelTiling()
{
    Init();
    auto inputDimNum = varShape.GetDimNum();
    for (uint32_t i = 0; i < inputDimNum; ++i) {
        auto dimInput = varShape.GetDim(i);
        auto dimUpdates = valueShape.GetDim(i);
        if (i < dimAttr) {
            batchNum *= dimUpdates;
        }
        if (i == dimAttr) {
            varDimNum = dimInput;
        }
        if (i >= dimAttr + 1) {
            updatesOneTime *= dimUpdates;
        }
        inputCount *= dimInput;
        updatesCount *= dimUpdates;
    }
    indicesCount = idxShape.GetDim(INPUT_0);
    if (ge::DT_FLOAT == varDtype) {
        tilingKey = FLOAT32_TILING_KEY;
    } else if (ge::DT_FLOAT16 == varDtype) {
        tilingKey = FLOAT16_TILING_KEY;
    } else if (ge::DT_BF16 == varDtype) {
        tilingKey = BF16_TILING_KEY;
    } else if (ge::DT_INT16 == varDtype) {
        tilingKey = INT16_TILING_KEY;
    } else if (ge::DT_INT32 == varDtype) {
        tilingKey = INT32_TILING_KEY;
    }
    processFirstDimTilingData();
    return ge::GRAPH_SUCCESS;
}

void InplaceIndexAddWithSortedTilingDef::processFirstDimTilingData()
{
    const auto iter = DTYPE_BUF_CNT_MAP.find(tilingKey);
    maxSize = AlignA2B((ubSize / iter->second), BLOCK_SIZE) / inputSize;
    usedCoreNum = indicesCount < coreNum ? indicesCount : coreNum;
    eachIndexCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (indicesCount + eachIndexCount - 1) / eachIndexCount;
    lastIndexCount = indicesCount - eachIndexCount * (usedCoreNum - 1);
    eachNum = updatesOneTime;
    eachTail = eachNum;
    if (eachNum > maxSize) {
        eachLoop = (eachNum + maxSize - 1) / maxSize;
        eachNum = maxSize;
        eachTail = updatesOneTime - (eachLoop - 1) * eachNum;
    }
    if (eachIndexCount > UB_INDEX_NUM) {
        eachUBIndexRound = (eachIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        eachUBIndexCount = UB_INDEX_NUM;
        eachUBIndexTail = eachIndexCount - (eachUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        eachUBIndexRound = 1;
        eachUBIndexCount = eachIndexCount;
        eachUBIndexTail = eachIndexCount;
    }
    if (lastIndexCount > UB_INDEX_NUM) {
        lastUBIndexRound = (lastIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        lastUBIndexCount = UB_INDEX_NUM;
        lastUBIndexTail = lastUBIndexCount - (lastUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        lastUBIndexRound = 1;
        lastUBIndexCount = lastIndexCount;
        lastUBIndexTail = lastIndexCount;
    }
}

void InplaceIndexAddWithSortedTilingDef::TilingDataSet(uint8_t* tilingPtr)
{
    tilingData.usedCoreNum = usedCoreNum;
    tilingData.enableAlpha = enableAlpha;
    tilingData.eachIndexCount = eachIndexCount;
    tilingData.lastIndexCount = lastIndexCount;
    tilingData.batchCount = batchNum;
    tilingData.eachBatchNum = eachBatchNum;
    tilingData.lastBatchNum = lastBatchNum;
    tilingData.inputCount = inputCount;
    tilingData.indicesCount = indicesCount;
    tilingData.updatesCount = updatesCount;
    tilingData.updatesOneTime = updatesOneTime;
    tilingData.maxSize = maxSize;
    tilingData.eachNum = eachNum;
    tilingData.eachLoop = eachLoop;
    tilingData.eachTail = eachTail;
    tilingData.varDimNum = varDimNum;
    tilingData.eachUBIndexRound = eachUBIndexRound;
    tilingData.eachUBIndexCount = eachUBIndexCount;
    tilingData.eachUBIndexTail = eachUBIndexTail;
    tilingData.lastUBIndexRound = lastUBIndexRound;
    tilingData.lastUBIndexCount = lastUBIndexCount;
    tilingData.lastUBIndexTail = lastUBIndexTail;
    memcpy(tilingPtr, &tilingData, sizeof(InplaceIndexAddWithSortedTilingData));
}

} // namespace optiling

#endif // __TILING_FUNCTION_DEF_H__