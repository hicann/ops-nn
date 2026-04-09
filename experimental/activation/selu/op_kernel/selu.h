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
 * \file selu.h
 * \brief Selu kernel class definition (arch32)
 *
 * SELU(x) = scale * [max(0, x) + min(0, alpha * (exp(x) - 1))]
 *
 * where alpha = 1.6732632423543772848170429916717
 *       scale = 1.0507009873554804934193349852946
 *
 * Template parameter T (mapped by TilingKey):
 *   - float:       TilingKey 0: direct computation in float32
 *   - half:        TilingKey 1: direct computation in float16
 *   - bfloat16_t:  TilingKey 2: cast to fp32 -> compute -> cast back to bf16
 *   - int32_t:     TilingKey 3: cast to fp32 -> compute -> cast back to int32
 *   - int8_t:      TilingKey 4: cast to fp32 -> compute -> cast back to int8
 *
 * Buffer layout (single buffer):
 *   inputQueue(1 buf):  ubFactor * sizeof(T)
 *   outputQueue(1 buf): ubFactor * sizeof(T)
 *   tmpBuf1_:           ubFactor * sizeof(computeT)
 *   tmpBuf2_:           ubFactor * sizeof(computeT)
 */
#ifndef SELU_H
#define SELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "selu_tiling_data.h"
#include "selu_tiling_key.h"

namespace NsSelu {

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

// SELU fixed constants
constexpr float ALPHA_F32 = 1.6732632423543772848170429916717f;
constexpr float SCALE_F32 = 1.0507009873554804934193349852946f;
constexpr half  ALPHA_F16 = half(1.6732632423543772848170429916717);
constexpr half  SCALE_F16 = half(1.0507009873554804934193349852946);

// Compute type trait: maps T -> compute type size
// float -> float(4), half -> half(2), bfloat16_t -> float(4),
// int32_t -> float(4), int8_t -> float(4)
template <typename T>
struct ComputeTypeTraits {
    static constexpr int64_t size = sizeof(float); // default
};
template <> struct ComputeTypeTraits<float>      { static constexpr int64_t size = sizeof(float); };
template <> struct ComputeTypeTraits<half>       { static constexpr int64_t size = sizeof(half); };
template <> struct ComputeTypeTraits<bfloat16_t> { static constexpr int64_t size = sizeof(float); };
template <> struct ComputeTypeTraits<int32_t>    { static constexpr int64_t size = sizeof(float); };
template <> struct ComputeTypeTraits<int8_t>     { static constexpr int64_t size = sizeof(float); };

template <typename T>
class Selu {
public:
    __aicore__ inline Selu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SeluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);

    // float32 direct computation
    __aicore__ inline void ComputeFloat32(LocalTensor<float>& xFloat,
                                           LocalTensor<float>& yFloat,
                                           int64_t alignedNum);
    // float16 direct computation
    __aicore__ inline void ComputeFloat16(LocalTensor<half>& xHalf,
                                           LocalTensor<half>& yHalf,
                                           int64_t alignedNum);
    // bf16/int32/int8 path: cast to fp32, compute, cast back
    template <typename SrcT>
    __aicore__ inline void ComputeCastFp32(LocalTensor<SrcT>& xLocal,
                                            LocalTensor<SrcT>& yLocal,
                                            int64_t currentNum,
                                            int64_t alignedNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1_;  // exp/cast intermediate
    TBuf<QuePosition::VECCALC> tmpBuf2_;  // max(0,x)/calc workspace

    GlobalTensor<T> xGM_;
    GlobalTensor<T> yGM_;

    int64_t blockOffset_;
    int64_t blockLen_;
    int64_t ubFactor_;
};

// =============================================================================
// Init
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::Init(GM_ADDR x, GM_ADDR y, const SeluTilingData* tilingData)
{
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

    xGM_.SetGlobalBuffer((__gm__ T*)x + blockOffset_, blockLen_);
    yGM_.SetGlobalBuffer((__gm__ T*)y + blockOffset_, blockLen_);

    constexpr int64_t computeTSize = ComputeTypeTraits<T>::size;
    pipe.InitBuffer(inputQueue, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outputQueue, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1_, ubFactor_ * computeTSize);
    pipe.InitBuffer(tmpBuf2_, ubFactor_ * computeTSize);
}

// =============================================================================
// CopyIn
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::CopyIn(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, xGM_[gmOffset], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(xLocal);
}

// =============================================================================
// CopyOut
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(yGM_[gmOffset], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

// =============================================================================
// ComputeFloat32 - FP32 direct computation
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::ComputeFloat32(LocalTensor<float>& xFloat,
                                                   LocalTensor<float>& yFloat,
                                                   int64_t alignedNum)
{
    LocalTensor<float> tmp1 = tmpBuf1_.template Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2_.template Get<float>();

    // Step 1: compute alpha * (exp(min(x, 0)) - 1) -> tmp1
    // Clamp exp input to <= 0 to prevent overflow (SELU only uses exp in x<=0 region)
    Mins(tmp1, xFloat, 0.0f, alignedNum);     // tmp1 = min(x, 0)
    Exp(tmp1, tmp1, alignedNum);               // tmp1 = exp(min(x, 0)), value in (0, 1]
    Adds(tmp1, tmp1, -1.0f, alignedNum);       // tmp1 = exp(min(x, 0)) - 1
    Muls(tmp1, tmp1, ALPHA_F32, alignedNum);   // tmp1 = alpha * (exp(min(x, 0)) - 1)
    Mins(tmp1, tmp1, 0.0f, alignedNum);        // tmp1 = min(0, alpha*(exp(...)-1))

    // Step 2: compute max(0, x) -> tmp2
    Maxs(tmp2, xFloat, 0.0f, alignedNum);      // tmp2 = max(0, x)

    // Step 3: merge and multiply by scale
    Add(yFloat, tmp2, tmp1, alignedNum);             // y = max(0,x) + min(0,...)
    Muls(yFloat, yFloat, SCALE_F32, alignedNum);     // y = scale * y
}

// =============================================================================
// ComputeFloat16 - FP16 direct computation
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::ComputeFloat16(LocalTensor<half>& xHalf,
                                                   LocalTensor<half>& yHalf,
                                                   int64_t alignedNum)
{
    LocalTensor<half> tmp1 = tmpBuf1_.template Get<half>();
    LocalTensor<half> tmp2 = tmpBuf2_.template Get<half>();

    // Step 1: compute alpha * (exp(min(x, 0)) - 1) -> tmp1
    // Clamp exp input to <= 0 to prevent overflow
    Mins(tmp1, xHalf, half(0.0), alignedNum);       // tmp1 = min(x, 0)
    Exp(tmp1, tmp1, alignedNum);                     // tmp1 = exp(min(x, 0))
    Adds(tmp1, tmp1, half(-1.0), alignedNum);        // tmp1 = exp(min(x, 0)) - 1
    Muls(tmp1, tmp1, ALPHA_F16, alignedNum);         // tmp1 = alpha * (exp(...) - 1)
    Mins(tmp1, tmp1, half(0.0), alignedNum);         // tmp1 = min(0, ...)

    // Step 2: compute max(0, x) -> tmp2
    Maxs(tmp2, xHalf, half(0.0), alignedNum);        // tmp2 = max(0, x)

    // Step 3: merge and multiply by scale
    Add(yHalf, tmp2, tmp1, alignedNum);              // y = max(0,x) + min(0,...)
    Muls(yHalf, yHalf, SCALE_F16, alignedNum);       // y = scale * y
}

// =============================================================================
// ComputeCastFp32 - BF16/INT32 path: cast to FP32, compute, cast back
// =============================================================================
template <typename T>
template <typename SrcT>
__aicore__ inline void Selu<T>::ComputeCastFp32(LocalTensor<SrcT>& xLocal,
                                                    LocalTensor<SrcT>& yLocal,
                                                    int64_t currentNum,
                                                    int64_t alignedNum)
{
    LocalTensor<float> tmp1 = tmpBuf1_.template Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2_.template Get<float>();

    // Cast input to fp32
    // Hardware-supported Cast paths on arch32:
    //   int8 -> half -> float (2 steps, no direct int8->float)
    //   bfloat16 -> float (direct)
    //   int32 -> float (direct)
    if constexpr (std::is_same_v<SrcT, int8_t>) {
        // Two-step: int8 -> half -> float
        // Reuse tmp2 as half intermediate (same buffer, reinterpreted)
        LocalTensor<half> tmpHalf = tmp2.template ReinterpretCast<half>();
        Cast(tmpHalf, xLocal, RoundMode::CAST_NONE, alignedNum);
        Cast(tmp1, tmpHalf, RoundMode::CAST_NONE, alignedNum);
    } else {
        Cast(tmp1, xLocal, RoundMode::CAST_NONE, alignedNum);
    }

    // Compute SELU in fp32
    // Step 1: alpha * (exp(min(x, 0)) - 1)
    Mins(tmp2, tmp1, 0.0f, alignedNum);
    Exp(tmp2, tmp2, alignedNum);
    Adds(tmp2, tmp2, -1.0f, alignedNum);
    Muls(tmp2, tmp2, ALPHA_F32, alignedNum);
    Mins(tmp2, tmp2, 0.0f, alignedNum);

    // Step 2: max(0, x)
    Maxs(tmp1, tmp1, 0.0f, alignedNum);

    // Step 3: merge and scale
    Add(tmp1, tmp1, tmp2, alignedNum);
    Muls(tmp1, tmp1, SCALE_F32, alignedNum);

    // Cast back to original type
    // Hardware-supported output Cast paths on arch32:
    //   float -> int32 (direct, CAST_TRUNC for truncation toward zero)
    //   float -> bfloat16 (direct, CAST_ROUND)
    //   float -> half -> int8 (2 steps, no direct float->int8 or int32->int8)
    // For int32: fp32 -> int32 (CAST_TRUNC) to match C++ truncation.
    // For bf16: fp32 -> bf16 (CAST_ROUND) to round to nearest.
    // For int8: fp32 -> half (CAST_TRUNC) -> int8 (CAST_TRUNC).
    //   Using CAST_TRUNC for float->half to match truncation-toward-zero
    //   semantics (same as PyTorch's int() conversion).
    //   Then CAST_TRUNC for half->int8 truncates toward zero.
    if constexpr (std::is_same_v<SrcT, int8_t>) {
        // Two-step: float -> half -> int8
        LocalTensor<half> tmpHalf = tmp2.template ReinterpretCast<half>();
        Cast(tmpHalf, tmp1, RoundMode::CAST_TRUNC, alignedNum);
        Cast(yLocal, tmpHalf, RoundMode::CAST_TRUNC, alignedNum);
    } else if constexpr (std::is_same_v<SrcT, int32_t>) {
        Cast(yLocal, tmp1, RoundMode::CAST_TRUNC, alignedNum);
    } else {
        Cast(yLocal, tmp1, RoundMode::CAST_ROUND, alignedNum);
    }
}

// =============================================================================
// Compute - dispatch to appropriate method based on T
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();

    constexpr int64_t computeTSize = ComputeTypeTraits<T>::size;
    constexpr int64_t typeBlock = 32 / sizeof(T);
    constexpr int64_t computeBlock = 32 / computeTSize;
    constexpr int64_t alignBlock = (typeBlock > computeBlock) ? typeBlock : computeBlock;
    int64_t alignedNum = ((currentNum + alignBlock - 1) / alignBlock) * alignBlock;

    if constexpr (std::is_same_v<T, float>) {
        ComputeFloat32(xLocal, yLocal, alignedNum);
    } else if constexpr (std::is_same_v<T, half>) {
        ComputeFloat16(xLocal, yLocal, alignedNum);
    } else if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>) {
        ComputeCastFp32(xLocal, yLocal, currentNum, alignedNum);
    }

    outputQueue.template EnQue<T>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

// =============================================================================
// Process
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::Process()
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

} // namespace NsSelu
#endif // SELU_H
