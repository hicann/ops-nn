/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef DATA_FORMAT_DIM_MAP_H
#define DATA_FORMAT_DIM_MAP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "data_format_dim_map_tiling_data.h"
#include "data_format_dim_map_tiling_key.h"

namespace NsDataFormatDimMap {

using namespace AscendC;

template <typename T>
class DataFormatDimMap {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline DataFormatDimMap() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const DataFormatDimMapTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> modBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> cmpBuf;
    TBuf<QuePosition::VECCALC> resultBuf;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int32_t formatLen_ = 0;
    int32_t indValues_[10] = {0};
};

template <typename T>
__aicore__ inline void DataFormatDimMap<T>::Init(
    GM_ADDR x, GM_ADDR y, const DataFormatDimMapTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ <= 0) {
        return;
    }
    ubLength_ = tilingData->ubFactor;
    formatLen_ = tilingData->formatLen;

    for (int32_t i = 0; i < 2 * formatLen_ && i < 10; i++) {
        indValues_[i] = tilingData->expandedTable[i];
    }

    inputGM.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    uint32_t floatBytes = ubLength_ * sizeof(float);
    pipe.InitBuffer(castBuf, floatBytes);
    pipe.InitBuffer(modBuf, floatBytes);
    pipe.InitBuffer(tmpBuf, floatBytes);
    pipe.InitBuffer(resultBuf, floatBytes);

    uint32_t cmpBytes = ((ubLength_ / 8 + 31) / 32) * 32;
    pipe.InitBuffer(cmpBuf, cmpBytes);
}

template <typename T>
__aicore__ inline void DataFormatDimMap<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(inputLocal, inputGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void DataFormatDimMap<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], outputLocal, copyParams);
    outputQueue.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void DataFormatDimMap<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
    AscendC::LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

    AscendC::LocalTensor<float> castLocal = castBuf.template Get<float>();
    AscendC::LocalTensor<float> modLocal = modBuf.template Get<float>();
    AscendC::LocalTensor<float> tmpLocal = tmpBuf.template Get<float>();
    AscendC::LocalTensor<uint8_t> cmpLocal = cmpBuf.template Get<uint8_t>();
    AscendC::LocalTensor<float> resultLocal = resultBuf.template Get<float>();

    int32_t count = static_cast<int32_t>(currentNum);

    if constexpr (sizeof(T) == 8) {
        AscendC::LocalTensor<int32_t> tmpI32 = castBuf.template Get<int32_t>();
        AscendC::Cast(tmpI32, inputLocal, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(castLocal, tmpI32, AscendC::RoundMode::CAST_RINT, count);
    } else {
        AscendC::Cast(castLocal, inputLocal, AscendC::RoundMode::CAST_RINT, count);
    }

    float fN = static_cast<float>(formatLen_);
    float invN = 1.0f / fN;
    AscendC::Adds(modLocal, castLocal, fN, count);
    AscendC::Muls(tmpLocal, modLocal, invN, count);
    AscendC::Floor(tmpLocal, tmpLocal, count);
    AscendC::Muls(tmpLocal, tmpLocal, fN, count);
    AscendC::Sub(modLocal, modLocal, tmpLocal, count);

    AscendC::Duplicate(resultLocal, static_cast<float>(indValues_[formatLen_ - 1]), count);
    for (int32_t i = formatLen_ - 2; i >= 0; i--) {
        AscendC::Compares(cmpLocal, modLocal, static_cast<float>(i),
                          AscendC::CMPMODE::EQ, static_cast<uint32_t>(count));
        AscendC::Duplicate(tmpLocal, static_cast<float>(indValues_[i]), count);
        AscendC::Select(resultLocal, cmpLocal, tmpLocal, resultLocal,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
    }

    if constexpr (sizeof(T) == 8) {
        AscendC::LocalTensor<int32_t> tmpI32 = castBuf.template Get<int32_t>();
        AscendC::Cast(tmpI32, resultLocal, AscendC::RoundMode::CAST_RINT, count);
        AscendC::Cast(outputLocal, tmpI32, AscendC::RoundMode::CAST_NONE, count);
    } else {
        AscendC::Cast(outputLocal, resultLocal, AscendC::RoundMode::CAST_RINT, count);
    }

    outputQueue.EnQue(outputLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void DataFormatDimMap<T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsDataFormatDimMap
#endif // DATA_FORMAT_DIM_MAP_H
