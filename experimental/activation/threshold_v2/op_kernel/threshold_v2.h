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
/**
 * \file threshold_v2.h
 * \brief ThresholdV2 Kernel class
 *
 * Formula: y = (x > threshold) ? x : value
 */
#ifndef THRESHOLD_V2_H
#define THRESHOLD_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "threshold_v2_tiling_data.h"
#include "threshold_v2_tiling_key.h"

namespace NsThresholdV2 {

using namespace AscendC;

template <typename T>
__aicore__ inline T FloatToT(float v) {
    return static_cast<T>(v);
}

template <>
__aicore__ inline bfloat16_t FloatToT<bfloat16_t>(float v) {
    bfloat16_t result;
    uint32_t bits;
    *(reinterpret_cast<float*>(&bits)) = v;
    *(reinterpret_cast<uint16_t*>(&result)) = static_cast<uint16_t>(bits >> 16);
    return result;
}

template <typename T, int BUFFER_MODE>
class ThresholdV2 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr int64_t ELEMS_PER_REPEAT = 256 / sizeof(T);

public:
    __aicore__ inline ThresholdV2() {};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const ThresholdV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> maskBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    T thresholdT_;
    T valueT_;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void ThresholdV2<T, BUFFER_MODE>::Init(
    GM_ADDR self, GM_ADDR out, const ThresholdV2TilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    thresholdT_ = FloatToT<T>(tilingData->thresholdVal);
    valueT_ = FloatToT<T>(tilingData->valueVal);

    inputGM.SetGlobalBuffer((__gm__ T*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)out + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    int64_t maskBytes = ((ubLength_ + 255) / 256) * 32;
    pipe.InitBuffer(maskBuf, maskBytes);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ThresholdV2<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
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

template <typename T, int BUFFER_MODE>
__aicore__ inline void ThresholdV2<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal = inputQueue.template DeQue<T>();
    AscendC::LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

    uint32_t alignNum = static_cast<uint32_t>(
        ((currentNum + ELEMS_PER_REPEAT - 1) / ELEMS_PER_REPEAT) * ELEMS_PER_REPEAT);

    AscendC::LocalTensor<uint8_t> maskLocal = maskBuf.template Get<uint8_t>();

    AscendC::Compares(maskLocal,
                      inputLocal, thresholdT_, AscendC::CMPMODE::GT, alignNum);

    AscendC::Select(outputLocal,
                    maskLocal,
                    inputLocal, valueT_,
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignNum);

    outputQueue.template EnQue<T>(outputLocal);
    inputQueue.FreeTensor(inputLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ThresholdV2<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
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

template <typename T, int BUFFER_MODE>
__aicore__ inline void ThresholdV2<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsThresholdV2
#endif // THRESHOLD_V2_H
