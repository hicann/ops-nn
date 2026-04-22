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
 * \file fused_bias_leaky_relu.h
 * \brief FusedBiasLeakyRelu kernel class definition (arch35)
 *
 * Computes: y = LeakyRelu(x + bias, negative_slope) * scale
 *
 * Template parameters:
 *   - T: Data type (half/float)
 *   - BUFFER_MODE: Buffer mode (0=single buffer, 1=double buffer)
 */
#ifndef FUSED_BIAS_LEAKY_RELU_H
#define FUSED_BIAS_LEAKY_RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fused_bias_leaky_relu_tiling_data.h"
#include "fused_bias_leaky_relu_tiling_key.h"

namespace NsFusedBiasLeakyRelu {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class FusedBiasLeakyRelu {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

public:
    __aicore__ inline FusedBiasLeakyRelu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 const FusedBiasLeakyReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueBias;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMBias;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;   // Number of elements this core processes
    int64_t ubLength_ = 0;      // Number of elements per UB loop iteration (= ubFactor)
    int64_t loopCount_ = 0;     // Number of full UB loop iterations
    int64_t tailNum_ = 0;       // Remaining elements in the tail iteration
    T negativeSlope_;            // Type-converted negative_slope
    T scale_;                    // Type-converted scale
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void FusedBiasLeakyRelu<T, BUFFER_MODE>::Init(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y,
    const FusedBiasLeakyReluTilingData* tilingData)
{
    // 1. Compute GM offset and processing length for this core
    int64_t blockOffset = GetBlockIdx() * tilingData->blockFactor;
    int64_t remainderLength = tilingData->totalNum - blockOffset;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;

    // 2. UB loop parameters
    ubLength_ = tilingData->ubFactor;
    loopCount_ = blockLength_ / ubLength_;
    tailNum_ = blockLength_ % ubLength_;

    // 3. Scalar type conversion (float32 -> T)
    negativeSlope_ = static_cast<T>(tilingData->negativeSlope);
    scale_ = static_cast<T>(tilingData->scale);

    // 4. GM Tensor initialization (based on this core's offset)
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength_);
    inputGMBias.SetGlobalBuffer((__gm__ T*)bias + blockOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength_);

    // 5. Queue initialization
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueBias, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FusedBiasLeakyRelu<T, BUFFER_MODE>::Process()
{
    // Safety: if this core has no elements to process (e.g. totalNum < coreNum), skip
    if (blockLength_ <= 0) {
        return;
    }
    // Full loop iterations: each processes ubLength_ elements
    for (int64_t i = 0; i < loopCount_; i++) {
        CopyIn(i, ubLength_);
        Compute(ubLength_);
        CopyOut(i, ubLength_);
    }
    // Tail iteration: process remaining tailNum_ elements
    if (tailNum_ > 0) {
        CopyIn(loopCount_, tailNum_);
        Compute(tailNum_);
        CopyOut(loopCount_, tailNum_);
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FusedBiasLeakyRelu<T, BUFFER_MODE>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    LocalTensor<T> biasLocal = inputQueueBias.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(biasLocal, inputGMBias[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
    inputQueueBias.EnQue(biasLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FusedBiasLeakyRelu<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    LocalTensor<T> biasLocal = inputQueueBias.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    // t = x + bias (in-place on xLocal)
    Add(xLocal, xLocal, biasLocal, currentNum);
    // LeakyRelu in-place on xLocal
    LeakyRelu(xLocal, xLocal, negativeSlope_, currentNum);
    // y = t * scale
    Muls(yLocal, xLocal, scale_, currentNum);

    outputQueueY.EnQue(yLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueBias.FreeTensor(biasLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FusedBiasLeakyRelu<T, BUFFER_MODE>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

} // namespace NsFusedBiasLeakyRelu
#endif // FUSED_BIAS_LEAKY_RELU_H
