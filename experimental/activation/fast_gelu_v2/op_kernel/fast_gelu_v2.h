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
 * \file fast_gelu_v2.h
 * \brief FastGeluV2 kernel class definition (arch35)
 *
 * float32/float16 path (original formula):
 *   y = x * (sgn(x) * [-0.1444 * (clip(|0.7071*x|, max=1.769) - 1.769)^2 + 0.5] + 0.5)
 *   where sgn(x) = (x + 1e-12) / |x + 1e-12|
 *
 * bfloat16 path (equivalent formula, avoids Abs/Div):
 *   y = |x| * inner + 0.5 * x
 *   where inner = -0.1444 * (clip(0.7071 * |x|, max=1.769) - 1.769)^2 + 0.5
 *   |x| via Muls(-1) + Max
 *
 * Template parameters:
 *   - T: data type (half, float, bfloat16_t)
 *   - BUFFER_MODE: 0=single buffer, 1=double buffer
 */
#ifndef FAST_GELU_V2_H
#define FAST_GELU_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fast_gelu_v2_tiling_data.h"
#include "fast_gelu_v2_tiling_key.h"

namespace NsFastGeluV2 {

using namespace AscendC;

// Helper trait: compute type is float for half (FP16->FP32 promotion), T otherwise
template <typename T>
struct ComputeType { using type = T; };
template <>
struct ComputeType<half> { using type = float; };

template <typename T, int BUFFER_MODE>
class FastGeluV2 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    using CT = typename ComputeType<T>::type; // Compute type: float for half, T otherwise
    static constexpr bool NEED_CAST = !std::is_same<T, CT>::value;

public:
    __aicore__ inline FastGeluV2() {};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tmpBuf1;
    TBuf<TPosition::VECCALC> tmpBuf2;
    TBuf<TPosition::VECCALC> tmpBuf3;
    // Additional buffers for FP16->FP32 cast (input cast and output cast)
    TBuf<TPosition::VECCALC> castInBuf;
    TBuf<TPosition::VECCALC> castOutBuf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void FastGeluV2<T, BUFFER_MODE>::Init(GM_ADDR x, GM_ADDR y,
                                                         const FastGeluV2TilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    // Temporary buffers use CT (compute type) size: float for FP16, T for others
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(CT));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(CT));
    pipe.InitBuffer(tmpBuf3, ubLength_ * sizeof(CT));
    if constexpr (NEED_CAST) {
        // Cast buffers: castInBuf holds FP32 input, castOutBuf holds FP32 output before cast back
        pipe.InitBuffer(castInBuf, ubLength_ * sizeof(CT));
        pipe.InitBuffer(castOutBuf, ubLength_ * sizeof(CT));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FastGeluV2<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FastGeluV2<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FastGeluV2<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();

    // Get compute-type tensors (CT = float for FP16, T for FP32)
    AscendC::LocalTensor<CT> tmp1 = tmpBuf1.Get<CT>();
    AscendC::LocalTensor<CT> tmp2 = tmpBuf2.Get<CT>();
    AscendC::LocalTensor<CT> tmp3 = tmpBuf3.Get<CT>();

    // For FP16: Cast input to FP32 for computation, use FP32 throughout, cast back at end
    AscendC::LocalTensor<CT> xCompute;
    if constexpr (NEED_CAST) {
        xCompute = castInBuf.Get<CT>();
        AscendC::Cast(xCompute, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
    } else {
        // T == CT, xLocal is already the compute type
        xCompute = xLocal;
    }

    if constexpr (std::is_same<T, bfloat16_t>::value) {
        // bfloat16 equivalent formula: y = |x| * inner + 0.5 * x
        // Avoids Abs and Div which do not support bfloat16

        // Step 1: neg_x = -x
        AscendC::Muls(tmp1, xCompute, (CT)(-1.0f), currentNum);

        // Step 2: abs_x = max(x, -x)
        AscendC::Max(tmp2, xCompute, tmp1, currentNum);

        // Step 3: scaled_x = 0.7071 * abs_x
        AscendC::Muls(tmp1, tmp2, (CT)(0.7071f), currentNum);

        // Step 4: clipped_x = min(scaled_x, 1.769)
        AscendC::Mins(tmp1, tmp1, (CT)(1.769f), currentNum);

        // Step 5: diff = clipped_x - 1.769
        AscendC::Adds(tmp1, tmp1, (CT)(-1.769f), currentNum);

        // Step 6: sq = diff^2
        AscendC::Mul(tmp1, tmp1, tmp1, currentNum);

        // Step 7a: inner = -0.1444 * sq
        AscendC::Muls(tmp1, tmp1, (CT)(-0.1444f), currentNum);

        // Step 7b: inner = inner + 0.5
        AscendC::Adds(tmp1, tmp1, (CT)(0.5f), currentNum);

        // Step 8: term1 = abs_x * inner
        AscendC::Mul(tmp3, tmp2, tmp1, currentNum);

        // Step 9: term2 = 0.5 * x
        AscendC::Muls(tmp1, xCompute, (CT)(0.5f), currentNum);

        // Step 10: y = term1 + term2
        AscendC::Add(yLocal, tmp3, tmp1, currentNum);
    } else {
        // float32 / float16 path: original formula
        // FastGeluV2(x) = x * (sgn(x) * inner + 0.5)
        // where inner = -0.1444 * (clip(0.7071 * |x|, max=1.769) - 1.769)^2 + 0.5
        // and sgn(x) = (x + eps) / |x + eps|

        // Step 1: x_eps = x + eps
        AscendC::Adds(tmp1, xCompute, (CT)1e-12, currentNum);

        // Step 2a: abs_xeps = |x_eps|
        AscendC::Abs(tmp2, tmp1, currentNum);

        // Step 2b: sgn_x = x_eps / abs_xeps
        AscendC::Div(tmp1, tmp1, tmp2, currentNum);

        // Step 3: abs_x = |x|
        AscendC::Abs(tmp2, xCompute, currentNum);

        // Step 4: scaled_x = 0.7071 * abs_x
        AscendC::Muls(tmp2, tmp2, (CT)0.7071, currentNum);

        // Step 5: clipped_x = min(scaled_x, 1.769)
        AscendC::Mins(tmp2, tmp2, (CT)1.769, currentNum);

        // Step 6: diff = clipped_x - 1.769
        AscendC::Adds(tmp2, tmp2, (CT)(-1.769), currentNum);

        // Step 7: sq = diff^2
        AscendC::Mul(tmp2, tmp2, tmp2, currentNum);

        // Step 8a: inner = -0.1444 * sq
        AscendC::Muls(tmp2, tmp2, (CT)(-0.1444), currentNum);

        // Step 8b: inner = inner + 0.5
        AscendC::Adds(tmp2, tmp2, (CT)0.5, currentNum);

        // Step 9a: bracket = sgn_x * inner
        AscendC::Mul(tmp3, tmp1, tmp2, currentNum);

        // Step 9b: bracket = bracket + 0.5
        AscendC::Adds(tmp3, tmp3, (CT)0.5, currentNum);

        // Step 10: y = x * bracket
        if constexpr (NEED_CAST) {
            // Compute result in FP32, then cast back to FP16
            AscendC::LocalTensor<CT> resultFloat = castOutBuf.Get<CT>();
            AscendC::Mul(resultFloat, xCompute, tmp3, currentNum);
            AscendC::Cast(yLocal, resultFloat, AscendC::RoundMode::CAST_ROUND, currentNum);
        } else {
            AscendC::Mul(yLocal, xCompute, tmp3, currentNum);
        }
    }

    outputQueueY.template EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void FastGeluV2<T, BUFFER_MODE>::Process()
{
    // Guard: skip computation if this core has no work assigned
    if (blockLength_ <= 0) {
        return;
    }
    // Loop over UB-sized tiles within this core's block.
    // The last tile may process fewer than ubLength_ elements (tail handling).
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsFastGeluV2
#endif // FAST_GELU_V2_H
