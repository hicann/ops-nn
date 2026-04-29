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

#ifndef CELU_H
#define CELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "celu_tiling_data.h"
#include "celu_tiling_key.h"

namespace NsCelu {

using namespace AscendC;

template <typename T>
__aicore__ inline T GetExpUpperBound();

template <>
__aicore__ inline half GetExpUpperBound<half>()
{
    return static_cast<half>(11.0f);
}

template <>
__aicore__ inline float GetExpUpperBound<float>()
{
    return 87.0f;
}

template <typename T>
class Celu {
public:
    __aicore__ inline Celu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> calcBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;

    float alpha1_ = 0.0f;
    float alpha2_ = 0.0f;
    float alpha3_ = 0.0f;
};

template <typename T>
__aicore__ inline void Celu<T>::Init(GM_ADDR x, GM_ADDR y, const CeluTilingData* tilingData)
{
    if (tilingData->totalNum == 0 || tilingData->blockFactor == 0 || tilingData->ubFactor == 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }

    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    if (remainderLength <= 0) {
        blockLength_ = 0;
        ubLength_ = 0;
        return;
    }
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    alpha1_ = tilingData->alpha1;
    alpha2_ = tilingData->alpha2;
    alpha3_ = tilingData->alpha3;

    inputGM.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueue, 1, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, 1, ubLength_ * sizeof(T));
    int64_t maskAlignedSize = ((ubLength_ + 7) / 8 + 31) / 32 * 32;
    int64_t maskAreaInT = (maskAlignedSize + sizeof(T) - 1) / sizeof(T);
    pipe.InitBuffer(calcBuf, (2 * ubLength_ + maskAreaInT) * sizeof(T));
}

template <typename T>
__aicore__ inline void Celu<T>::CopyIn(int64_t progress, int64_t currentNum)
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
__aicore__ inline void Celu<T>::CopyOut(int64_t progress, int64_t currentNum)
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
__aicore__ inline void Celu<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
    AscendC::LocalTensor<T> fullBuf = calcBuf.Get<T>();
    AscendC::LocalTensor<T> negResult = fullBuf[0];
    AscendC::LocalTensor<T> zerosBuf = fullBuf[ubLength_];
    AscendC::LocalTensor<uint8_t> maskBuf = fullBuf[2 * ubLength_].template ReinterpretCast<uint8_t>();
    AscendC::LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

    AscendC::Divs(negResult, inputLocal, static_cast<T>(alpha2_), currentNum);
    AscendC::Mins(negResult, negResult, GetExpUpperBound<T>(), currentNum);
    AscendC::Exp(outputLocal, negResult, currentNum);
    AscendC::Adds(outputLocal, outputLocal, static_cast<T>(-1), currentNum);
    AscendC::Muls(negResult, outputLocal, static_cast<T>(alpha1_), currentNum);
    AscendC::Duplicate(outputLocal, static_cast<T>(alpha3_ * 3.0f), currentNum);
    AscendC::Duplicate(zerosBuf, static_cast<T>(0), currentNum);
    AscendC::Compare(maskBuf, inputLocal, zerosBuf, AscendC::CMPMODE::GE,
        static_cast<uint32_t>(currentNum));
    AscendC::Select(outputLocal, maskBuf, outputLocal, negResult,
        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(currentNum));

    inputQueue.FreeTensor(inputLocal);
    outputQueue.EnQue(outputLocal);
}

template <typename T>
__aicore__ inline void Celu<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
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

} // namespace NsCelu

#endif // CELU_H
