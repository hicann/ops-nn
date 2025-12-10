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
 * \file dynamic_quant_update_scatter_v2_base.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_BASE_H
#define DYNAMIC_QUANT_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DynamicQuantUpdateScatterV2NDOpt {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t SIXTEEN = 16;
constexpr uint32_t SEVEN = 7;
constexpr uint32_t EIGHT = 8;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t THIRTY_TWO = 32;
constexpr uint32_t SIXTY_THREE = 63;
constexpr uint32_t SIXTY_FOUR = 64;
constexpr uint32_t MAX_VALUE_NUM = 8;
constexpr int32_t ELEM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float)
constexpr int32_t ELEM_PER_REP_HALF = 128; // ONE_REPEAT_BYTE_SIZE / sizeof(half)
constexpr float DYNAMIC_QUANT_INT8_SCALE = 255.0;
constexpr float DYNAMIC_QUANT_INT8_SYM_SCALE = 127.0;
constexpr float DYNAMIC_QUANT_INT8_OFFSET = 127.0;
constexpr float DYNAMIC_QUANT_INT4_SCALE = 15.0;
constexpr float DYNAMIC_QUANT_INT4_SYM_SCALE = 7.0;
constexpr float DYNAMIC_QUANT_INT4_OFFSET = 7.0;
constexpr float DYNAMIC_QUANT_EPSINON = 1e-12;
constexpr float MIN_FLOAT_VALUE = -3.4028235e+38f;
constexpr float MAX_FLOAT_VALUE = 3.402823466e+38f;

class DynamicQuantUpdateScatterV2Base {
public:
    __aicore__ inline DynamicQuantUpdateScatterV2Base()
    {}

    __aicore__ inline void ParseTilingData(const DynamicQuantUpdateScatterV2TilingData* tilingData)
    {
        tilingData_.rowLen = tilingData->rowLen;
        tilingData_.coreNum = tilingData->coreNum;
        tilingData_.headCoreNum = tilingData->headCoreNum;
        tilingData_.rowPerHeadCore = tilingData->rowPerHeadCore;
        tilingData_.rowPerTailCore = tilingData->rowPerTailCore;
        tilingData_.multiRowNumTailCore = tilingData->multiRowNumTailCore;
        tilingData_.multiRowNumHeadCore = tilingData->multiRowNumHeadCore;
        tilingData_.dstSeqLen = tilingData->dstSeqLen; // sequence len
        tilingData_.batchSize = tilingData->batchSize; // batch size
    }

    __aicore__ inline void InitBaseBuffer()
    {
        pPipe->InitBuffer(tempCastUb, sizeHalfLen * sizeof(float));
        pPipe->InitBuffer(fp32_buf_, sizeHalfLen * sizeof(float));
    }

    __aicore__ inline void InitCommonParams()
    {
        lenHead = rowPerHeadCore * tilingData_.rowLen;
        lenTail = rowPerTailCore * tilingData_.rowLen;
        outLenQuant = tilingData_.batchSize * tilingData_.dstSeqLen;
        outLen = (outLenQuant * tilingData_.rowLen) >> 1;
    }

    __aicore__ inline void InitParams()
    {
        blockIdx = GetBlockIdx();
        rowPerHeadCore = tilingData_.rowPerHeadCore;
        rowPerTailCore = tilingData_.rowPerTailCore;

        if (blockIdx < tilingData_.headCoreNum) {
            rowPerCore = rowPerHeadCore;
            multiRowNum = tilingData_.multiRowNumHeadCore;
            loopCnt = rowPerHeadCore / multiRowNum;
            remainRow = rowPerHeadCore % multiRowNum;
        } else if (blockIdx >= tilingData_.headCoreNum && blockIdx < tilingData_.coreNum) {
            rowPerCore = rowPerTailCore;
            multiRowNum = tilingData_.multiRowNumTailCore;
            loopCnt = rowPerTailCore / multiRowNum;
            remainRow = rowPerTailCore % multiRowNum;
        } else {
        }
        sizeHalfLen = (tilingData_.rowLen + FIFTEEN) / SIXTEEN * SIXTEEN;
        rightPadding = sizeHalfLen - tilingData_.rowLen;
        if (rightPadding > 0) {
            isPad = true;
        }

        outAlignLen = (tilingData_.rowLen + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;
        sizeFloatLen = (multiRowNum + SEVEN) / EIGHT * EIGHT;
        sizeIntLen = (tilingData_.batchSize + SEVEN) / EIGHT * EIGHT;
        lenMultiRow = multiRowNum * sizeHalfLen;
        lenGMMultiRow = multiRowNum * tilingData_.rowLen;
        InitCommonParams();
    }

    __aicore__ inline void CopyIndicesIn()
    {
        LocalTensor<int32_t> indicesLocal = indicesQueue.AllocTensor<int32_t>();

        DataCopy(indicesLocal, indicesGm, sizeIntLen);

        indicesQueue.EnQue(indicesLocal);
    }

    __aicore__ inline float SafeDiv(float a, float b)
    {
        if (b < DYNAMIC_QUANT_EPSINON && b > -DYNAMIC_QUANT_EPSINON) {
            return a;
        }
        return a / b;
    }

    __aicore__ inline void GetScaleAndOffset(float max_value, float min_value, float& scale, float& offset)
    {
        scale = GetMax((max_value - min_value) / DYNAMIC_QUANT_INT4_SCALE, DYNAMIC_QUANT_EPSINON);
        offset = DYNAMIC_QUANT_INT4_OFFSET - SafeDiv(max_value, scale);
    }

    __aicore__ inline float GetMax(float a, float b)
    {
        return a > b ? a : b;
    }

    __aicore__ inline uint64_t GetDstOffset(
        uint64_t rowIndex, uint64_t coreBatchIndex, const LocalTensor<int32_t>& indicesLocal)
    {
        uint64_t indexIdx = blockIdx * rowPerHeadCore + coreBatchIndex * multiRowNum + rowIndex;
        uint64_t validIdx = indicesLocal.GetValue(indexIdx);
        uint64_t dstOffset = (indexIdx * tilingData_.dstSeqLen + validIdx) * tilingData_.rowLen;
        return dstOffset;
    }

protected:
    TPipe* pPipe = nullptr;
    /* tiling data */
    DynamicQuantUpdateScatterV2TilingData tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> indicesQueue;

    /* variable */
    uint64_t blockIdx;
    uint64_t sizeFloatLen;
    uint64_t sizeHalfLen;
    uint64_t outAlignLen;
    uint64_t multiRowNum;
    uint64_t sizeIntLen;
    uint64_t rowPerHeadCore;
    uint64_t rowPerTailCore;
    uint64_t rowPerCore;
    uint64_t lenHead;
    uint64_t lenTail;
    uint64_t lenMultiRow;
    uint64_t lenGMMultiRow;
    uint64_t outLen;
    uint64_t outLenQuant;
    uint64_t outLenHead;
    uint64_t outLenTail;
    uint64_t loopCnt = 0;
    uint64_t remainRow = 0;
    uint8_t rightPadding = 0;
    bool isPad = false;

    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempCastUb;

    GlobalTensor<int32_t> indicesGm;
};
} // namespace DynamicQuantUpdateScatterV2NDOpt
#endif // DynamicQuantUpdateScatterV2Base