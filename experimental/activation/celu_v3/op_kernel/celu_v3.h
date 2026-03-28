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
 * \file celu_v3.h
 * \brief CeluV3 kernel class definition (arch32)
 *
 * CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
 *
 * Template parameter T:
 *   - float:    direct computation in float32
 *   - half:     cast to fp32 -> compute -> cast back to fp16
 *   - bfloat16_t: cast to fp32 -> compute -> cast back to bf16
 *
 * Buffer layout (single buffer):
 *   inputQueue(1 buf):  ubFactor * sizeof(T)
 *   outputQueue(1 buf): ubFactor * sizeof(T)
 *   tmpBuf1_:           ubFactor * sizeof(float) - for exp intermediate
 *   tmpBuf2_:           ubFactor * sizeof(float) - for max(0,x) intermediate
 */
#ifndef CELU_V3_H
#define CELU_V3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "celu_v3_tiling_data.h"
#include "celu_v3_tiling_key.h"

namespace NsCeluV3 {

using AscendC::TPipe;
using AscendC::TQue;
using AscendC::TBuf;
using AscendC::QuePosition;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::DataCopyParams;
using AscendC::DataCopyPad;
using AscendC::RoundMode;
using AscendC::GetBlockIdx;
using AscendC::Muls;
using AscendC::Exp;
using AscendC::Adds;
using AscendC::Mins;
using AscendC::Maxs;
using AscendC::Add;
using AscendC::Cast;

template <typename T>
class CeluV3 {
public:
    __aicore__ inline CeluV3() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const CeluV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);

    // float32 direct computation
    __aicore__ inline void ComputeFloat32(LocalTensor<float>& xFloat, LocalTensor<float>& yFloat,
                                          int64_t alignedNum);
    // float16/bf16 path: cast to fp32, compute, cast back
    template <typename SrcT>
    __aicore__ inline void ComputeWithCast(LocalTensor<SrcT>& xLocal, LocalTensor<SrcT>& yLocal,
                                           int64_t currentNum, int64_t alignedNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1_;  // exp intermediate (float32)
    TBuf<QuePosition::VECCALC> tmpBuf2_;  // max(0,x) intermediate (float32)

    GlobalTensor<T> selfGM_;
    GlobalTensor<T> outGM_;

    float alphaVal_;
    float invAlpha_;
    int64_t blockOffset_;
    int64_t blockLen_;
    int64_t ubFactor_;
};

template <typename T>
__aicore__ inline void CeluV3<T>::Init(GM_ADDR self, GM_ADDR out, const CeluV3TilingData* tilingData)
{
    alphaVal_ = tilingData->alphaVal;
    invAlpha_ = tilingData->invAlpha;
    ubFactor_ = tilingData->ubFactor;

    // Handle empty tensor or excess cores: early return with blockLen_=0
    if (tilingData->totalElements == 0 || tilingData->blockFactor == 0) {
        blockOffset_ = 0;
        blockLen_ = 0;
        return;
    }

    // Compute per-core offset and length
    blockOffset_ = tilingData->blockFactor * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t remaining = tilingData->totalElements - blockOffset_;
    if (remaining <= 0) {
        blockLen_ = 0;
        return;
    }
    blockLen_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;

    selfGM_.SetGlobalBuffer((__gm__ T*)self + blockOffset_, blockLen_);
    outGM_.SetGlobalBuffer((__gm__ T*)out + blockOffset_, blockLen_);

    pipe.InitBuffer(inputQueue, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outputQueue, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1_, ubFactor_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2_, ubFactor_ * sizeof(float));
}

template <typename T>
__aicore__ inline void CeluV3<T>::CopyIn(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, selfGM_[gmOffset], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void CeluV3<T>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outGM_[gmOffset], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void CeluV3<T>::ComputeFloat32(LocalTensor<float>& xFloat,
                                                   LocalTensor<float>& yFloat,
                                                   int64_t alignedNum)
{
    // Get temporary buffers
    LocalTensor<float> tmp1 = tmpBuf1_.template Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2_.template Get<float>();

    // Step 1: compute exp(x/alpha) - 1 part -> tmp1
    // tmp1 = x * (1/alpha)
    Muls(tmp1, xFloat, invAlpha_, alignedNum);
    // Clamp exp argument to [-87.3, 87.3] for numerical stability
    // exp(87.3) ~ 7.5e37 (within fp32 range), exp(88.8) overflows to Inf
    Mins(tmp1, tmp1, 87.3f, alignedNum);
    Maxs(tmp1, tmp1, -87.3f, alignedNum);
    // tmp1 = exp(x/alpha)
    Exp(tmp1, tmp1, alignedNum);
    // tmp1 = exp(x/alpha) - 1
    Adds(tmp1, tmp1, -1.0f, alignedNum);
    // tmp1 = alpha * (exp(x/alpha) - 1)
    Muls(tmp1, tmp1, alphaVal_, alignedNum);
    // tmp1 = min(0, alpha * (exp(x/alpha) - 1))
    Mins(tmp1, tmp1, 0.0f, alignedNum);

    // Step 2: compute max(0, x) -> tmp2
    Maxs(tmp2, xFloat, 0.0f, alignedNum);

    // Step 3: result = max(0, x) + min(0, ...)
    Add(yFloat, tmp2, tmp1, alignedNum);
}

template <typename T>
template <typename SrcT>
__aicore__ inline void CeluV3<T>::ComputeWithCast(LocalTensor<SrcT>& xLocal,
                                                    LocalTensor<SrcT>& yLocal,
                                                    int64_t currentNum,
                                                    int64_t alignedNum)
{
    // Get temporary buffers (used as fp32 workspace)
    LocalTensor<float> tmp1 = tmpBuf1_.template Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2_.template Get<float>();

    // Cast input to fp32
    // tmp1 = float32(x)
    Cast(tmp1, xLocal, RoundMode::CAST_NONE, alignedNum);

    // Compute CELU in fp32
    // tmp2 = x * (1/alpha)
    Muls(tmp2, tmp1, invAlpha_, alignedNum);
    // Clamp exp argument to [-87.3, 87.3] for numerical stability
    // exp(87.3) ~ 7.5e37 (within fp32 range), exp(88.8) overflows to Inf
    Mins(tmp2, tmp2, 87.3f, alignedNum);
    Maxs(tmp2, tmp2, -87.3f, alignedNum);
    // tmp2 = exp(x/alpha)
    Exp(tmp2, tmp2, alignedNum);
    // tmp2 = exp(x/alpha) - 1
    Adds(tmp2, tmp2, -1.0f, alignedNum);
    // tmp2 = alpha * (exp(x/alpha) - 1)
    Muls(tmp2, tmp2, alphaVal_, alignedNum);
    // tmp2 = min(0, alpha * (exp(x/alpha) - 1))
    Mins(tmp2, tmp2, 0.0f, alignedNum);

    // tmp1 still holds fp32(x), compute max(0, x)
    Maxs(tmp1, tmp1, 0.0f, alignedNum);

    // tmp1 = max(0,x) + min(0, ...)
    Add(tmp1, tmp1, tmp2, alignedNum);

    // Cast back to original type
    Cast(yLocal, tmp1, RoundMode::CAST_ROUND, alignedNum);
}

template <typename T>
__aicore__ inline void CeluV3<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();

    // Align to 32 bytes for vector operations.
    // For float32: align to 8 elements (32B / 4B).
    // For fp16/bf16: Cast between T and float requires alignment to both
    //   32/sizeof(T) = 16 elements and 32/sizeof(float) = 8 elements,
    //   so use 16-element alignment (LCM of 8 and 16).
    constexpr int64_t floatBlock = 32 / sizeof(float);  // 8
    constexpr int64_t typeBlock = 32 / sizeof(T);        // 8 for float, 16 for half/bf16
    constexpr int64_t alignBlock = (floatBlock > typeBlock) ? floatBlock : typeBlock;
    int64_t alignedNum = ((currentNum + alignBlock - 1) / alignBlock) * alignBlock;

    if constexpr (std::is_same_v<T, float>) {
        // float32 direct computation
        ComputeFloat32(xLocal, yLocal, alignedNum);
    } else {
        // float16/bfloat16 path: cast to fp32, compute, cast back
        ComputeWithCast(xLocal, yLocal, currentNum, alignedNum);
    }

    outputQueue.template EnQue<T>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void CeluV3<T>::Process()
{
    // Early return for empty tensor or excess cores
    if (blockLen_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLen_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t gmOffset = i * ubFactor_;
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLen_ - gmOffset) : ubFactor_;
        CopyIn(gmOffset, currentNum);
        Compute(currentNum);
        CopyOut(gmOffset, currentNum);
    }
}

} // namespace NsCeluV3
#endif // CELU_V3_H
