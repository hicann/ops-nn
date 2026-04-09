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
 * \file hard_sigmoid_grad.h
 * \brief HardSigmoidGrad kernel class definition (arch35)
 *
 * Computes: grad_input = grad_output * alpha  if 0 < alpha*x + beta < 1
 *                        0                    otherwise
 *
 * Implementation uses two-pass CompareScalar+Select approach:
 * 1. Compute tmp = alpha * x + beta
 * 2. CompareScalar(mask, tmp, 0, GT) => bitmask where tmp > 0
 * 3. result = grad_output * alpha
 * 4. Select(result, mask, result, 0.0) => zero where tmp <= 0
 * 5. CompareScalar(mask, tmp, 1, LT) => bitmask where tmp < 1
 * 6. Select(result, mask, result, 0.0) => zero where tmp >= 1
 */
#ifndef HARD_SIGMOID_GRAD_H
#define HARD_SIGMOID_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_sigmoid_grad_tiling_data.h"
#include "hard_sigmoid_grad_tiling_key.h"

namespace NsHardSigmoidGrad {

using namespace AscendC;

template <typename T>
class HardSigmoidGrad {
    static constexpr int32_t BUFFER_NUM = 2; // double buffer

public:
    __aicore__ inline HardSigmoidGrad() {};

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                 const HardSigmoidGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> gradOutputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> selfQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gradInputQueue;
    // Temp buffers for computation
    TBuf<QuePosition::VECCALC> tmpBuf;       // for alpha*x+beta intermediate
    TBuf<QuePosition::VECCALC> maskBuf;      // for compare bitmask results

    GlobalTensor<T> gradOutputGM;
    GlobalTensor<T> selfGM;
    GlobalTensor<T> gradInputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float alpha_ = 0.16666666f;
    float beta_ = 0.5f;
};

template <typename T>
__aicore__ inline void HardSigmoidGrad<T>::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
    const HardSigmoidGradTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalLength - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ <= 0) {
        return;
    }
    ubLength_ = tilingData->ubFactor;
    alpha_ = tilingData->alpha;
    beta_ = tilingData->beta;

    gradOutputGM.SetGlobalBuffer((__gm__ T*)gradOutput + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    selfGM.SetGlobalBuffer((__gm__ T*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    gradInputGM.SetGlobalBuffer((__gm__ T*)gradInput + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(gradOutputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(selfQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(gradInputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    // tmp buffer for intermediate computation (alpha*x+beta)
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(T));
    // mask buffer for compare bitmask: N float32 elements => N/8 uint8 bytes, 32-byte aligned
    int64_t maskBufSize = ((ubLength_ + 7) / 8 + 31) / 32 * 32;
    if (maskBufSize < 32) maskBufSize = 32;
    pipe.InitBuffer(maskBuf, maskBufSize);
}

template <typename T>
__aicore__ inline void HardSigmoidGrad<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> gradLocal = gradOutputQueue.template AllocTensor<T>();
    LocalTensor<T> selfLocal = selfQueue.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(gradLocal, gradOutputGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(selfLocal, selfGM[progress * ubLength_], copyParams, {false, 0, 0, 0});

    gradOutputQueue.EnQue(gradLocal);
    selfQueue.EnQue(selfLocal);
}

template <typename T>
__aicore__ inline void HardSigmoidGrad<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> gradLocal = gradOutputQueue.template DeQue<T>();
    LocalTensor<T> selfLocal = selfQueue.template DeQue<T>();
    LocalTensor<T> resultLocal = gradInputQueue.template AllocTensor<T>();
    LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();

    // Align count to 256-byte boundary for CompareScalar
    // For float32: 256 bytes / 4 = 64 elements
    // For float16: 256 bytes / 2 = 128 elements
    constexpr int64_t COMPARE_ALIGN = static_cast<int64_t>(256 / sizeof(T));
    int64_t alignedCount = (currentNum + COMPARE_ALIGN - 1) / COMPARE_ALIGN * COMPARE_ALIGN;

    // Step 1: tmp = alpha * self + beta
    Muls(tmpLocal, selfLocal, static_cast<T>(alpha_), currentNum);
    Adds(tmpLocal, tmpLocal, static_cast<T>(beta_), currentNum);

    // Step 2: result = grad_output * alpha
    Muls(resultLocal, gradLocal, static_cast<T>(alpha_), currentNum);

    // Step 3: mask = (tmp > 0), select result where mask=1, else 0
    CompareScalar(maskLocal, tmpLocal, static_cast<T>(0), CMPMODE::GT, alignedCount);
    Select(resultLocal, maskLocal, resultLocal, static_cast<T>(0),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, currentNum);

    // Step 4: mask = (tmp < 1), select result where mask=1, else 0
    CompareScalar(maskLocal, tmpLocal, static_cast<T>(1), CMPMODE::LT, alignedCount);
    Select(resultLocal, maskLocal, resultLocal, static_cast<T>(0),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, currentNum);

    gradInputQueue.template EnQue<T>(resultLocal);
    gradOutputQueue.FreeTensor(gradLocal);
    selfQueue.FreeTensor(selfLocal);
}

template <typename T>
__aicore__ inline void HardSigmoidGrad<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> resultLocal = gradInputQueue.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(gradInputGM[progress * ubLength_], resultLocal, copyParams);
    gradInputQueue.FreeTensor(resultLocal);
}

template <typename T>
__aicore__ inline void HardSigmoidGrad<T>::Process()
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

} // namespace NsHardSigmoidGrad
#endif // HARD_SIGMOID_GRAD_H
