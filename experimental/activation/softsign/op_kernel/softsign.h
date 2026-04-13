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

/*!
 * \file softsign.h
 * \brief Softsign kernel class definition (arch35)
 *
 * Softsign(x) = x / (1 + |x|)
 *
 * Two kernel classes:
 *   - SoftsignKernel<float, BUFFER_MODE>: for float (direct compute, no Cast)
 *   - SoftsignCastKernel<TIn, BUFFER_MODE>: for half/bfloat16 (Cast->float->compute->Cast->TIn)
 */
#ifndef SOFTSIGN_H
#define SOFTSIGN_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softsign_tiling_data.h"
#include "softsign_tiling_key.h"

namespace NsSoftsign {

using namespace AscendC;

// ============================================================================
// SoftsignKernel: float direct compute path (no Cast needed)
// ============================================================================
template <typename T, int BUFFER_MODE>
class SoftsignKernel {
    static_assert(std::is_same_v<T, float>, "SoftsignKernel only supports float; use SoftsignCastKernel for half/bf16");
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline SoftsignKernel() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftsignTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignKernel<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const SoftsignTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGM.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignKernel<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignKernel<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueue.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();
    AscendC::LocalTensor<T> tmpLocal = tmpBuf.Get<T>();

    AscendC::Abs(tmpLocal, xLocal, currentNum);
    AscendC::Adds(tmpLocal, tmpLocal, static_cast<T>(1), currentNum);
    AscendC::Div(yLocal, xLocal, tmpLocal, currentNum);

    outputQueue.template EnQue<T>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignKernel<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void SoftsignKernel<T, BUFFER_MODE>::Process()
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

// ============================================================================
// SoftsignCastKernel: half / bfloat16 Cast compute path
// Cast TIn->float -> Abs -> Adds -> Div -> Cast float->TIn
// ============================================================================
template <typename TIn, int BUFFER_MODE>
class SoftsignCastKernel {
    static_assert(std::is_same_v<TIn, half> || std::is_same_v<TIn, bfloat16_t>,
                  "SoftsignCastKernel only supports half and bfloat16_t");
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline SoftsignCastKernel() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SoftsignTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> castInBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> calcOutBuf;

    GlobalTensor<TIn> inputGM;
    GlobalTensor<TIn> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename TIn, int BUFFER_MODE>
__aicore__ inline void SoftsignCastKernel<TIn, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR y, const SoftsignTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGM.SetGlobalBuffer((__gm__ TIn*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ TIn*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(TIn));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(TIn));
    pipe.InitBuffer(castInBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(calcOutBuf, ubLength_ * sizeof(float));
}

template <typename TIn, int BUFFER_MODE>
__aicore__ inline void SoftsignCastKernel<TIn, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<TIn> xLocal = inputQueue.template AllocTensor<TIn>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(TIn));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(xLocal);
}

template <typename TIn, int BUFFER_MODE>
__aicore__ inline void SoftsignCastKernel<TIn, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<TIn> xIn = inputQueue.template DeQue<TIn>();
    AscendC::LocalTensor<TIn> yOut = outputQueue.template AllocTensor<TIn>();
    AscendC::LocalTensor<float> xFloat = castInBuf.Get<float>();
    AscendC::LocalTensor<float> tmpFloat = tmpBuf.Get<float>();
    AscendC::LocalTensor<float> yFloat = calcOutBuf.Get<float>();

    AscendC::Cast(xFloat, xIn, RoundMode::CAST_NONE, currentNum);
    AscendC::Abs(tmpFloat, xFloat, currentNum);
    AscendC::Adds(tmpFloat, tmpFloat, 1.0f, currentNum);
    AscendC::Div(yFloat, xFloat, tmpFloat, currentNum);
    AscendC::Cast(yOut, yFloat, RoundMode::CAST_ROUND, currentNum);

    outputQueue.template EnQue<TIn>(yOut);
    inputQueue.FreeTensor(xIn);
}

template <typename TIn, int BUFFER_MODE>
__aicore__ inline void SoftsignCastKernel<TIn, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<TIn> yLocal = outputQueue.template DeQue<TIn>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(TIn));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

template <typename TIn, int BUFFER_MODE>
__aicore__ inline void SoftsignCastKernel<TIn, BUFFER_MODE>::Process()
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

} // namespace NsSoftsign

#endif // SOFTSIGN_H
