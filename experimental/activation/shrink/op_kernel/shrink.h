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
 * @file shrink.h
 * @brief Shrink Kernel class (arch35)
 *
 * Template parameters:
 *   - T: data type (half / float)
 *   - BUFFER_MODE: buffer mode (0=single, 1=double)
 *
 * Computation:
 *   out = self - bias,  if self > lambd
 *   out = self + bias,  if self < -lambd
 *   out = 0,            if -lambd <= self <= lambd
 *
 * Strategy (2 queues + 1 compute buffer: input, output, midBuf):
 *   1. Compares(self < -lambd) -> maskNeg
 *   2. Compares(self > lambd) -> maskPos
 *   3. outLocal = self + (-bias) using Adds
 *   4. Select using maskPos: where maskPos=1 keep self-bias, else pick 0
 *   5. midLocal = self + bias using Adds
 *   6. Select using maskNeg: where maskNeg=1 pick self+bias, else keep current
 *
 * Uses Compares for scalar comparison (outputs bitmask),
 * Select with VSEL_TENSOR_TENSOR_MODE, And for mask combination.
 */
#ifndef SHRINK_H
#define SHRINK_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "shrink_tiling_data.h"
#include "shrink_tiling_key.h"

namespace NsShrink {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class Shrink {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline Shrink() {};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out,
                                const ShrinkTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe_;
    // 2 queues (input, output) + 1 compute buffer (mid)
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    // Mid buffer: pure compute, no MTE pipeline needed, saves double-buffer overhead
    TBuf<TPosition::VECCALC> midBuf_;

    // Mask buffers for Compares output (bitmask: 1 bit per element)
    TBuf<TPosition::VECCALC> maskBuf1_;
    TBuf<TPosition::VECCALC> maskBuf2_;

    GlobalTensor<T> gmIn_;
    GlobalTensor<T> gmOut_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lambd_ = 1.0f;
    float bias_ = 1.0f;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void Shrink<T, BUFFER_MODE>::Init(
    GM_ADDR self, GM_ADDR out,
    const ShrinkTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;
    lambd_ = tilingData->lambd;
    bias_ = tilingData->bias;

    int64_t offset = tilingData->blockFactor * AscendC::GetBlockIdx();
    gmIn_.SetGlobalBuffer((__gm__ T*)self + offset, blockLength_);
    gmOut_.SetGlobalBuffer((__gm__ T*)out + offset, blockLength_);

    // 2 data queues + 1 compute buffer
    pipe_.InitBuffer(inputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe_.InitBuffer(midBuf_, ubLength_ * sizeof(T));

    // Mask buffers: ubLength/8 bytes, 32-byte aligned, minimum 32 bytes
    int64_t maskBytes = (ubLength_ / 8 + 31) & ~31;
    if (maskBytes < 32) maskBytes = 32;
    pipe_.InitBuffer(maskBuf1_, maskBytes);
    pipe_.InitBuffer(maskBuf2_, maskBytes);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Shrink<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inLocal = inputQueue_.template AllocTensor<T>();

    // byteLen is uint16_t (max 65535); Tiling ensures currentNum * sizeof(T) <= 65535
    DataCopyPad(inLocal, gmIn_[progress * ubLength_],
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0},
        {false, 0, 0, 0});

    inputQueue_.EnQue(inLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Shrink<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outLocal = outputQueue_.template DeQue<T>();

    // byteLen is uint16_t (max 65535); Tiling ensures currentNum * sizeof(T) <= 65535
    DataCopyPad(gmOut_[progress * ubLength_], outLocal,
        {1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0});

    outputQueue_.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Shrink<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> inLocal = inputQueue_.template DeQue<T>();
    LocalTensor<T> midLocal = midBuf_.template Get<T>();
    LocalTensor<T> outLocal = outputQueue_.template AllocTensor<T>();
    LocalTensor<uint8_t> maskNeg = maskBuf1_.template Get<uint8_t>();
    LocalTensor<uint8_t> maskPos = maskBuf2_.template Get<uint8_t>();

    // computeNum: 256B-aligned count for Compares (Level 2 API requires count*sizeof(T) % 256 == 0)
    // Adds/Duplicate/Select use currentNum (actual valid element count) to avoid processing
    // invalid padding data in the last chunk.
    int64_t typeSize = sizeof(T);
    int64_t computeAlignment = 256 / typeSize;
    int64_t computeNum = ((currentNum + computeAlignment - 1) / computeAlignment) * computeAlignment;

    // Phase 1: Compute masks from original self
    Compares(maskNeg, inLocal, static_cast<T>(-lambd_), CMPMODE::LT, computeNum);
    Compares(maskPos, inLocal, static_cast<T>(lambd_), CMPMODE::GT, computeNum);

    // Phase 2: Compute result incrementally
    // Step 1: outLocal = self - bias (using Adds with negated bias)
    T negBias = static_cast<T>(-bias_);
    Adds(outLocal, inLocal, negBias, currentNum);

    // Step 2: Duplicate midLocal as zero buffer for dead zone
    Duplicate(midLocal, static_cast<T>(0.0f), currentNum);

    // Step 3: Where maskPos=0 (not positive region), pick 0; maskPos=1 keep self-bias
    Select(outLocal, maskPos, outLocal, midLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, currentNum);

    // Step 4: midLocal = self + bias (negative region value, inLocal still has original self)
    T posBias = static_cast<T>(bias_);
    Adds(midLocal, inLocal, posBias, currentNum);

    // Step 5: Where maskNeg=1 (negative region), pick self+bias; else keep current
    Select(outLocal, maskNeg, midLocal, outLocal,
           SELMODE::VSEL_TENSOR_TENSOR_MODE, currentNum);

    outputQueue_.template EnQue<T>(outLocal);
    inputQueue_.FreeTensor(inLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void Shrink<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0) return;

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsShrink

#endif // SHRINK_H
