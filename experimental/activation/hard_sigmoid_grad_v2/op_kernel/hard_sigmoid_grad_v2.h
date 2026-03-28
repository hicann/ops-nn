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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file hard_sigmoid_grad_v2.h
 * \brief HardSigmoidGradV2 kernel class definition (arch32)
 *
 * Template parameters:
 *   - T: Data type (half/float/bfloat16_t)
 *   - BUFFER_MODE: Buffer mode (0=single buffer, 1=double buffer)
 *
 * Computation:
 *   grad_input = grad_output * ((self > -3) & (self < 3)) * (1/6)
 *
 * For half/float: direct Compares + And + Muls + Select
 * For bfloat16_t: Cast to float, compute, Cast back (arch32 does not support bf16 in Compare/Muls/Select)
 */
#ifndef HARD_SIGMOID_GRAD_V2_H
#define HARD_SIGMOID_GRAD_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "hard_sigmoid_grad_v2_tiling_data.h"
#include "hard_sigmoid_grad_v2_tiling_key.h"

namespace NsHardSigmoidGradV2 {

template <typename T, int BUFFER_MODE>
class HardSigmoidGradV2 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool IS_BF16 = std::is_same<T, bfloat16_t>::value;
    // Compute type: float for bf16, T itself for half/float
    using ComputeT = typename std::conditional<IS_BF16, float, T>::type;

public:
    __aicore__ inline HardSigmoidGradV2() {};

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
                                const HardSigmoidGradV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gradQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> selfQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    // Temporary buffers for mask computation
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maskBuf2;
    // Extra float buffers for bf16 path (Cast bf16 -> float -> compute -> Cast back)
    AscendC::TBuf<AscendC::QuePosition::VECCALC> floatBuf1;  // for selfLocal cast to float
    AscendC::TBuf<AscendC::QuePosition::VECCALC> floatBuf2;  // for gradLocal cast to float / result

    AscendC::GlobalTensor<T> gradGM;
    AscendC::GlobalTensor<T> selfGM;
    AscendC::GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV2<T, BUFFER_MODE>::Init(
    GM_ADDR gradOutput, GM_ADDR self, GM_ADDR gradInput,
    const HardSigmoidGradV2TilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    gradGM.SetGlobalBuffer((__gm__ T*)gradOutput + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    selfGM.SetGlobalBuffer((__gm__ T*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)gradInput + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(gradQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(selfQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));

    // Mask buffers: Compares outputs bit mask, 1 bit per element -> ceil(ubLength_/8) bytes
    // Align to 32 bytes
    int64_t maskBytes = (ubLength_ / 8 + 31) / 32 * 32;
    if (maskBytes < 32) maskBytes = 32;
    pipe.InitBuffer(maskBuf1, maskBytes);
    pipe.InitBuffer(maskBuf2, maskBytes);

    if constexpr (IS_BF16) {
        // bf16 path needs float temp buffers for computation
        pipe.InitBuffer(floatBuf1, ubLength_ * sizeof(float));
        pipe.InitBuffer(floatBuf2, ubLength_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV2<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> gradLocal = gradQueue.template AllocTensor<T>();
    AscendC::LocalTensor<T> selfLocal = selfQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(gradLocal, gradGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    AscendC::DataCopyPad(selfLocal, selfGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    gradQueue.EnQue(gradLocal);
    selfQueue.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV2<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> resultLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], resultLocal, copyParams);
    outputQueue.FreeTensor(resultLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV2<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> gradLocal = gradQueue.template DeQue<T>();
    AscendC::LocalTensor<T> selfLocal = selfQueue.template DeQue<T>();
    AscendC::LocalTensor<T> resultLocal = outputQueue.template AllocTensor<T>();

    // Get mask buffers
    AscendC::LocalTensor<uint8_t> mask1 = maskBuf1.Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> mask2 = maskBuf2.Get<uint8_t>();

    // Use ubLength_ (guaranteed aligned to 256/sizeof(ComputeT)) for vector APIs.
    // Compares Level 2 requires count*sizeof(T) % 256 == 0.
    // CopyIn/CopyOut use currentNum for actual data; extra buffer elements are don't-care.
    uint32_t count = static_cast<uint32_t>(ubLength_);
    int32_t maskCount = static_cast<int32_t>((ubLength_ + 7) / 8);

    if constexpr (IS_BF16) {
        // bf16 path: Cast to float, compute in float, Cast back
        AscendC::LocalTensor<float> selfFloat = floatBuf1.Get<float>();
        AscendC::LocalTensor<float> gradFloat = floatBuf2.Get<float>();

        // Cast bf16 -> float
        AscendC::Cast(selfFloat, selfLocal, AscendC::RoundMode::CAST_NONE, count);
        AscendC::Cast(gradFloat, gradLocal, AscendC::RoundMode::CAST_NONE, count);

        // Step 1: Compares self > -3.0 -> mask1
        AscendC::Compares(mask1, selfFloat, static_cast<float>(-3.0f), AscendC::CMPMODE::GT, count);
        // Step 2: Compares self < 3.0 -> mask2
        AscendC::Compares(mask2, selfFloat, static_cast<float>(3.0f), AscendC::CMPMODE::LT, count);
        // Step 3: And mask1 & mask2 -> mask1
        AscendC::And(mask1, mask1, mask2, maskCount);
        // Step 4: Muls gradFloat * (1/6) -> gradFloat (in-place)
        AscendC::Muls(gradFloat, gradFloat, static_cast<float>(1.0f / 6.0f), count);
        // Step 5: Select: mask1 ? gradFloat : 0 -> selfFloat (reuse as result)
        AscendC::Select(selfFloat, mask1, gradFloat, static_cast<float>(0.0f),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        // Cast float -> bf16
        AscendC::Cast(resultLocal, selfFloat, AscendC::RoundMode::CAST_RINT, count);
    } else {
        // half/float path: direct computation
        // Step 1: Compares self > -3.0 -> mask1
        ComputeT negThree = static_cast<ComputeT>(-3.0);
        AscendC::Compares(mask1, selfLocal, negThree, AscendC::CMPMODE::GT, count);
        // Step 2: Compares self < 3.0 -> mask2
        ComputeT posThree = static_cast<ComputeT>(3.0);
        AscendC::Compares(mask2, selfLocal, posThree, AscendC::CMPMODE::LT, count);
        // Step 3: And mask1 & mask2 -> mask1
        AscendC::And(mask1, mask1, mask2, maskCount);
        // Step 4: Muls gradLocal * (1/6) -> gradLocal (in-place)
        ComputeT oneOverSix = static_cast<ComputeT>(1.0 / 6.0);
        AscendC::Muls(gradLocal, gradLocal, oneOverSix, count);
        // Step 5: Select: mask1 ? gradLocal : 0 -> resultLocal
        ComputeT zero = static_cast<ComputeT>(0.0);
        AscendC::Select(resultLocal, mask1, gradLocal, zero,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }

    outputQueue.template EnQue<T>(resultLocal);
    gradQueue.FreeTensor(gradLocal);
    selfQueue.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void HardSigmoidGradV2<T, BUFFER_MODE>::Process()
{
    if (blockLength_ <= 0) return;  // Empty tensor protection
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsHardSigmoidGradV2
#endif // HARD_SIGMOID_GRAD_V2_H
