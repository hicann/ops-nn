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
 * \brief Selu kernel class definition (arch35, Ascend950)
 *
 * SELU(x) = scale * [max(0, x) + min(0, alpha * (exp(x) - 1))]
 *
 * where alpha = 1.6732632423543772848170429916717
 *       scale = 1.0507009873554804934193349852946
 *
 * Template parameter T (mapped by TilingKey):
 *   - float:       TilingKey 0: direct computation in float32
 *   - half:        TilingKey 1: cast fp16 -> fp32 -> compute -> cast back to fp16  ← 用户约定：FP16 必须走 FP32 中间
 *   - bfloat16_t:  TilingKey 2: cast bf16 -> fp32 -> compute -> cast back to bf16
 *   - int32_t:     TilingKey 3: cast int32 -> fp32 -> compute -> cast back to int32
 *   - int8_t:      TilingKey 4: int8 -> half -> float -> compute -> half -> int8
 *
 * Buffer layout (single buffer):
 *   inputQueue(1 buf):  ubFactor * sizeof(T)
 *   outputQueue(1 buf): ubFactor * sizeof(T)
 *   tmpBuf1_:           ubFactor * sizeof(float)  // 全 dtype 统一用 fp32 中间
 *   tmpBuf2_:           ubFactor * sizeof(float)
 *
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

// SELU fixed constants（全部以 fp32 表示，FP16/BF16/INT 路径均通过 cast 到 fp32 计算后再回原 dtype）
constexpr float ALPHA_F32 = 1.6732632423543772848170429916717f;
constexpr float SCALE_F32 = 1.0507009873554804934193349852946f;

// Compute type trait: 所有 dtype 统一用 fp32 作为中间计算类型
template <typename T>
struct ComputeTypeTraits {
    static constexpr int64_t size = sizeof(float);
};

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
    // 非 fp32 dtype（half/bf16/int32/int8）统一走 cast-to-fp32 路径
    template <typename SrcT>
    __aicore__ inline void ComputeCastFp32(LocalTensor<SrcT>& xLocal,
                                            LocalTensor<SrcT>& yLocal,
                                            int64_t currentNum,
                                            int64_t alignedNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1_;  // exp/cast intermediate (fp32)
    TBuf<QuePosition::VECCALC> tmpBuf2_;  // max(0,x)/calc workspace (fp32)

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

    blockOffset_ = tilingData->blockFactor * static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t remaining = tilingData->totalElements - blockOffset_;
    if (remaining <= 0) {
        blockLen_ = 0;
        return;
    }
    blockLen_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;

    xGM_.SetGlobalBuffer((__gm__ T*)x + blockOffset_, blockLen_);
    yGM_.SetGlobalBuffer((__gm__ T*)y + blockOffset_, blockLen_);

    // 所有 dtype 统一使用 fp32 作为中间类型，tmpBuf 大小固定 = ubFactor * sizeof(float)
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
// ComputeFloat32 - FP32 direct computation (fast path, no cast)
// =============================================================================
template <typename T>
__aicore__ inline void Selu<T>::ComputeFloat32(LocalTensor<float>& xFloat,
                                                   LocalTensor<float>& yFloat,
                                                   int64_t alignedNum)
{
    LocalTensor<float> tmp1 = tmpBuf1_.template Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2_.template Get<float>();

    // Step 1: alpha * (exp(min(x, 0)) - 1) -> tmp1
    // Clamp exp 输入到 <= 0 防止上溢（SELU 只在 x<=0 区域用 exp）
    Mins(tmp1, xFloat, 0.0f, alignedNum);
    Exp(tmp1, tmp1, alignedNum);
    Adds(tmp1, tmp1, -1.0f, alignedNum);
    Muls(tmp1, tmp1, ALPHA_F32, alignedNum);
    Mins(tmp1, tmp1, 0.0f, alignedNum);

    // Step 2: max(0, x) -> tmp2
    Maxs(tmp2, xFloat, 0.0f, alignedNum);

    // Step 3: merge and multiply by scale
    Add(yFloat, tmp2, tmp1, alignedNum);
    Muls(yFloat, yFloat, SCALE_F32, alignedNum);
}

// =============================================================================
// ComputeCastFp32 - 非 fp32 dtype 统一 cast 路径
// FP16 / BF16 / INT32 / INT8 全部走：源 dtype -> fp32 -> compute -> 源 dtype
//
// Hardware-supported Cast paths on arch35:
//   half     <-> float (direct)
//   bfloat16 <-> float (direct)
//   int32    <-> float (direct, CAST_TRUNC for round-to-zero)
//   int8     <-> float 需经 half 中转（int8 <-> half <-> float）
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

    // ---- Cast input to fp32 ----
    if constexpr (std::is_same_v<SrcT, int8_t>) {
        // int8 -> half -> float (两步)
        LocalTensor<half> tmpHalf = tmp2.template ReinterpretCast<half>();
        Cast(tmpHalf, xLocal, RoundMode::CAST_NONE, alignedNum);
        Cast(tmp1, tmpHalf, RoundMode::CAST_NONE, alignedNum);
    } else {
        // half / bfloat16 / int32 -> float (一步)
        Cast(tmp1, xLocal, RoundMode::CAST_NONE, alignedNum);
    }

    // ---- Compute SELU in fp32 ----
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

    // ---- Cast back to original type ----
    // half:    fp32 -> half (CAST_ROUND, RNE 四舍五入到最近偶数)
    // bf16:    fp32 -> bf16 (CAST_ROUND)
    // int32:   fp32 -> int32 (CAST_TRUNC，向零截断，与 C++ int() 语义一致)
    // int8:    fp32 -> half -> int8 (两步，half->int8 用 CAST_TRUNC 截断到零)
    if constexpr (std::is_same_v<SrcT, int8_t>) {
        LocalTensor<half> tmpHalf = tmp2.template ReinterpretCast<half>();
        Cast(tmpHalf, tmp1, RoundMode::CAST_TRUNC, alignedNum);
        Cast(yLocal, tmpHalf, RoundMode::CAST_TRUNC, alignedNum);
    } else if constexpr (std::is_same_v<SrcT, int32_t>) {
        Cast(yLocal, tmp1, RoundMode::CAST_TRUNC, alignedNum);
    } else {
        // half / bfloat16 都走 CAST_ROUND（最近偶数舍入）
        Cast(yLocal, tmp1, RoundMode::CAST_ROUND, alignedNum);
    }
}

// =============================================================================
// Compute - dispatch by T
// 用户约定：所有非 fp32 的 dtype（含 FP16/BF16/INT*）统一走 ComputeCastFp32。
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
    } else {
        // half / bfloat16_t / int32_t / int8_t 全部走 cast-to-fp32 路径
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
