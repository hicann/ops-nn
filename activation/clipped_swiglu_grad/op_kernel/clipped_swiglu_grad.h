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
 * \file clipped_swiglu_grad.h
 * \brief ClippedSwigluGrad kernel implementation (910B / 910_93, UB pipeline)
 *
 * 反向公式（与 golden 对齐）：
 *   A = clamp(a, max=limit);  B = clamp(b, -limit, limit);  s = sigmoid(alpha * A)
 *   maskA = (a <= limit);     maskB = (-limit <= b <= limit)
 *   da = dy * (B + bias) * s * (1 + alpha * A * (1 - s)) * maskA
 *   db = dy * A * s * maskB
 *   dx 散回：interleaved -> Scatter(da,even) + Scatter(db,odd);
 *           front/back  -> Copy(da,first_half) + Copy(db,second_half)
 *
 * Buffer layout (half = xQueSpace_ / sizeof(float) / SWI_FACTOR):
 *   tmpBuf1_[0..half]    : tmpA (a -> A_clamped)
 *   tmpBuf1_[half..2*half]: tmpB (b -> B_clamped+bias)
 *   xFloatLocal[0..half]  : sBuf (s = sigmoid)
 *   xFloatLocal[half..2*half]: scratch (db -> da intermediate)
 *   tmpBuf2_[0..half]    : Gather offsets (GetAB), da staging (interleaved ScatterResult)
 *   tmpBuf2_[half..2*half]: db storage (interleaved only, float)
 *   tmpBuf1_ (ScatterResult 阶段复用): interleaved 交错索引表 (uint32 字节偏移)
 *   dxFloatLocal (ScatterResult 阶段复用): 索引表临时区, 随后被 Gather 结果覆盖
 *   maskBufA_ / maskBufB_ : CompareScalar bitmasks (small)
 */
#ifndef OPP_CLIPPED_SWIGLU_GRAD_H
#define OPP_CLIPPED_SWIGLU_GRAD_H
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "clipped_swiglu_grad_base.h"

namespace ClippedSwigluGradOps {
using namespace AscendC;
constexpr static int64_t BITS_PER_BYTE = 8;
constexpr static int64_t GATHER_REG_ELEM = 64;

template <typename T, bool isInterleaved, bool isGroup>
class ClippedSwigluGradBase
    : public ClippedSwigluGradSchedBase<T, isInterleaved, isGroup, ClippedSwigluGradBase<T, isInterleaved, isGroup>> {
public:
    using SchedBase = ClippedSwigluGradSchedBase<T, isInterleaved, isGroup,
                                                 ClippedSwigluGradBase<T, isInterleaved, isGroup>>;
    __aicore__ inline ClippedSwigluGradBase(const ClippedSwigluGradTilingData* tilingData, TPipe* pipe)
        : SchedBase(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR groupIndex, GM_ADDR gradXOut);
    __aicore__ inline void ProcessSingleLoop(int64_t xOffset, int64_t dyOffset, int64_t dxOffset);

private:
    __aicore__ inline void Compute(LocalTensor<float>& xFloatLocal, LocalTensor<float>& dyFloatLocal,
                                   LocalTensor<float>& tmpUbF32, LocalTensor<float>& dxFloatLocal);
    __aicore__ inline void ComputeMasks(LocalTensor<float>& tmpA, LocalTensor<float>& tmpB, LocalTensor<uint8_t>& maskA,
                                        LocalTensor<uint8_t>& maskB);
    __aicore__ inline void ComputeSigmoid(LocalTensor<float>& tmpA, LocalTensor<float>& sBuf,
                                          LocalTensor<float>& scratch);
    __aicore__ inline void ComputeDb(LocalTensor<float>& tmpA, LocalTensor<float>& sBuf,
                                     LocalTensor<float>& dyFloatLocal, LocalTensor<float>& scratch,
                                     LocalTensor<float>& tmpB, LocalTensor<uint8_t>& maskB,
                                     LocalTensor<float>& dxFloatLocal);
    __aicore__ inline void ComputeDa(LocalTensor<float>& tmpA, LocalTensor<float>& sBuf, LocalTensor<float>& tmpB,
                                     LocalTensor<float>& dyFloatLocal, LocalTensor<float>& scratch,
                                     LocalTensor<uint8_t>& maskA);
    __aicore__ inline void ScatterResult(LocalTensor<float>& scratch, LocalTensor<float>& dxFloatLocal);
    __aicore__ inline void GetAB(LocalTensor<float>& tmpA, LocalTensor<float>& tmpB, LocalTensor<float>& xFloatLocal);

    using SchedBase::AlignBytes;
    using SchedBase::calPairNum_;
    using SchedBase::CopyIn;
    using SchedBase::CopyOut;
    using SchedBase::dxDbOffset_;
    using SchedBase::dxQueue_;
    using SchedBase::dyLocalOffset_;
    using SchedBase::dyQueue_;
    using SchedBase::half_;
    using SchedBase::InitCommon;
    using SchedBase::pairNum_;
    using SchedBase::pipe_;
    using SchedBase::tiling_;
    using SchedBase::ubMaxPair_;
    using SchedBase::xLocalOffset1_;
    using SchedBase::xLocalOffset2_;
    using SchedBase::xQueSpace_;
    using SchedBase::xQueue_;

    /* ascendc variable */
    TBuf<TPosition::VECCALC> tmpBuf1_;
    TBuf<TPosition::VECCALC> tmpBuf2_;
    TBuf<TPosition::VECCALC> maskBufA_;
    TBuf<TPosition::VECCALC> maskBufB_;
};

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::Init(GM_ADDR gradY, GM_ADDR x,
                                                                              GM_ADDR groupIndex, GM_ADDR gradXOut)
{
    InitCommon(gradY, x, groupIndex, gradXOut);
    pipe_->InitBuffer(tmpBuf1_, xQueSpace_);
    pipe_->InitBuffer(tmpBuf2_, xQueSpace_);
    int64_t maskBytes = (ubMaxPair_ + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
    int64_t maskBufSize = AlignBytes(maskBytes);
    if (maskBufSize < BLOCK_SIZE) {
        maskBufSize = BLOCK_SIZE;
    }
    pipe_->InitBuffer(maskBufA_, maskBufSize);
    pipe_->InitBuffer(maskBufB_, maskBufSize);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ProcessSingleLoop(int64_t xOffset,
                                                                                           int64_t dyOffset,
                                                                                           int64_t dxOffset)
{
    CopyIn(xOffset, dyOffset);
    LocalTensor<T> xDTypeLocal = xQueue_.template DeQue<T>();
    LocalTensor<T> dyDTypeLocal = dyQueue_.template DeQue<T>();
    LocalTensor<float> xFloatLocal = xDTypeLocal.template ReinterpretCast<float>();
    LocalTensor<float> dyFloatLocal = dyDTypeLocal.template ReinterpretCast<float>();

    if constexpr (!std::is_same_v<T, float>) {
        if constexpr (!isInterleaved) {
            Cast(xFloatLocal, xDTypeLocal[xLocalOffset1_], RoundMode::CAST_NONE, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(xFloatLocal[half_], xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], RoundMode::CAST_NONE, calPairNum_);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(xFloatLocal, xDTypeLocal[xLocalOffset1_], RoundMode::CAST_NONE, calPairNum_ * SWI_FACTOR);
            PipeBarrier<PIPE_V>();
        }
        Cast(dyFloatLocal, dyDTypeLocal[dyLocalOffset_], RoundMode::CAST_NONE, calPairNum_);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> tmpUbF32 = tmpBuf1_.Get<float>();
    LocalTensor<float> dxFloatLocal = dxQueue_.template AllocTensor<float>();
    Compute(xFloatLocal, dyFloatLocal, tmpUbF32, dxFloatLocal);

    LocalTensor<T> dxDTypeLocal = dxFloatLocal.template ReinterpretCast<T>();
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        if constexpr (!isInterleaved) {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_RINT, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(dxDTypeLocal[dxDbOffset_], dxFloatLocal[half_], RoundMode::CAST_RINT, calPairNum_);
        } else {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_RINT, calPairNum_ * SWI_FACTOR);
        }
    } else if constexpr (std::is_same_v<T, half>) {
        if constexpr (!isInterleaved) {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_NONE, calPairNum_);
            PipeBarrier<PIPE_V>();
            Cast(dxDTypeLocal[dxDbOffset_], dxFloatLocal[half_], RoundMode::CAST_NONE, calPairNum_);
        } else {
            Cast(dxDTypeLocal, dxFloatLocal, RoundMode::CAST_NONE, calPairNum_ * SWI_FACTOR);
        }
    }
    dxQueue_.template EnQue<T>(dxDTypeLocal);
    CopyOut(dxOffset);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::Compute(LocalTensor<float>& xFloatLocal,
                                                                                 LocalTensor<float>& dyFloatLocal,
                                                                                 LocalTensor<float>& tmpUbF32,
                                                                                 LocalTensor<float>& dxFloatLocal)
{
    LocalTensor<float> tmpA = tmpUbF32;
    LocalTensor<float> tmpB = tmpUbF32[half_];
    LocalTensor<float> sBuf = xFloatLocal;
    LocalTensor<float> scratch = xFloatLocal[half_];
    LocalTensor<uint8_t> maskA = maskBufA_.Get<uint8_t>();
    LocalTensor<uint8_t> maskB = maskBufB_.Get<uint8_t>();

    GetAB(tmpA, tmpB, xFloatLocal);
    ComputeMasks(tmpA, tmpB, maskA, maskB);
    ComputeSigmoid(tmpA, sBuf, scratch);
    ComputeDb(tmpA, sBuf, dyFloatLocal, scratch, tmpB, maskB, dxFloatLocal);
    ComputeDa(tmpA, sBuf, tmpB, dyFloatLocal, scratch, maskA);
    ScatterResult(scratch, dxFloatLocal);

    xQueue_.FreeTensor(xFloatLocal);
    dyQueue_.FreeTensor(dyFloatLocal);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeMasks(LocalTensor<float>& tmpA,
                                                                                      LocalTensor<float>& tmpB,
                                                                                      LocalTensor<uint8_t>& maskA,
                                                                                      LocalTensor<uint8_t>& maskB)
{
    constexpr int64_t CMP_ALIGN = 64;
    int64_t alignedCount = (calPairNum_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
    CompareScalar(maskA, tmpA, tiling_->limit, CMPMODE::LE, alignedCount);
    CompareScalar(maskB, tmpB, tiling_->limit, CMPMODE::LE, alignedCount);
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeSigmoid(LocalTensor<float>& tmpA,
                                                                                        LocalTensor<float>& sBuf,
                                                                                        LocalTensor<float>& scratch)
{
    Mins(tmpA, tmpA, tiling_->limit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Muls(sBuf, tmpA, -1 * tiling_->alpha, calPairNum_);
    PipeBarrier<PIPE_V>();
    Exp(sBuf, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(sBuf, sBuf, (float)1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Duplicate(scratch, (float)1.0, calPairNum_);
    Div(sBuf, scratch, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeDb(
    LocalTensor<float>& tmpA, LocalTensor<float>& sBuf, LocalTensor<float>& dyFloatLocal, LocalTensor<float>& scratch,
    LocalTensor<float>& tmpB, LocalTensor<uint8_t>& maskB, LocalTensor<float>& dxFloatLocal)
{
    Mul(scratch, tmpA, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, dyFloatLocal, calPairNum_);
    PipeBarrier<PIPE_V>();
    Select(scratch, maskB, scratch, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, calPairNum_);
    PipeBarrier<PIPE_V>();
    constexpr int64_t CMP_ALIGN = 64;
    int64_t alignedCount = (calPairNum_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
    CompareScalar(maskB, tmpB, -1 * tiling_->limit, CMPMODE::GE, alignedCount);
    Select(scratch, maskB, scratch, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, calPairNum_);
    PipeBarrier<PIPE_V>();

    if constexpr (!isInterleaved) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dxFloatLocal[half_], scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    } else {
        LocalTensor<float> dbStorage = tmpBuf2_.Get<float>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dbStorage[half_], scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ComputeDa(
    LocalTensor<float>& tmpA, LocalTensor<float>& sBuf, LocalTensor<float>& tmpB, LocalTensor<float>& dyFloatLocal,
    LocalTensor<float>& scratch, LocalTensor<uint8_t>& maskA)
{
    Mins(tmpB, tmpB, tiling_->limit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Maxs(tmpB, tmpB, -1 * tiling_->limit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(tmpB, tmpB, tiling_->bias, calPairNum_);
    PipeBarrier<PIPE_V>();

    Muls(scratch, sBuf, (float)-1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(scratch, scratch, (float)1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, tmpA, calPairNum_);
    PipeBarrier<PIPE_V>();
    Muls(scratch, scratch, tiling_->alpha, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(scratch, scratch, (float)1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, sBuf, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, tmpB, calPairNum_);
    PipeBarrier<PIPE_V>();
    Mul(scratch, scratch, dyFloatLocal, calPairNum_);
    PipeBarrier<PIPE_V>();
    Select(scratch, maskA, scratch, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, calPairNum_);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::ScatterResult(LocalTensor<float>& scratch,
                                                                                       LocalTensor<float>& dxFloatLocal)
{
    if constexpr (!isInterleaved) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dxFloatLocal, scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    } else {
        LocalTensor<float> dbStorage = tmpBuf2_.Get<float>();
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(dbStorage, scratch, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();

        int64_t cnt = SWI_FACTOR * calPairNum_;
        int64_t cntAligned = (cnt + GATHER_REG_ELEM - 1) / GATHER_REG_ELEM * GATHER_REG_ELEM;
        LocalTensor<int32_t> interleaveIdx = tmpBuf1_.Get<int32_t>();
        LocalTensor<int32_t> idxTmp = dxFloatLocal.template ReinterpretCast<int32_t>();
        ArithProgression(interleaveIdx, static_cast<int32_t>(0), static_cast<int32_t>(1),
                         static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        ShiftRight(idxTmp, interleaveIdx, static_cast<int32_t>(1), static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        Muls(idxTmp, idxTmp, static_cast<int32_t>(8 * half_ - 4), static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        Muls(interleaveIdx, interleaveIdx, static_cast<int32_t>(4 * half_), static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        Sub(interleaveIdx, interleaveIdx, idxTmp, static_cast<int32_t>(GATHER_REG_ELEM));
        PipeBarrier<PIPE_V>();
        for (int64_t built = GATHER_REG_ELEM; built < cntAligned; built *= SWI_FACTOR) {
            int64_t len = cntAligned - built < built ? cntAligned - built : built;
            Adds(interleaveIdx[built], interleaveIdx, static_cast<int32_t>(SWI_FACTOR * built),
                 static_cast<int32_t>(len));
            PipeBarrier<PIPE_V>();
        }
        Gather(dxFloatLocal, dbStorage, interleaveIdx.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
               static_cast<uint32_t>(cnt));
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool isInterleaved, bool isGroup>
__aicore__ inline void ClippedSwigluGradBase<T, isInterleaved, isGroup>::GetAB(LocalTensor<float>& tmpA,
                                                                               LocalTensor<float>& tmpB,
                                                                               LocalTensor<float>& xFloatLocal)
{
    if constexpr (!isInterleaved) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(tmpA, xFloatLocal, AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        Copy<float, false>(tmpB, xFloatLocal[half_], AscendC::MASK_PLACEHOLDER, 1, {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    } else {
        LocalTensor<int32_t> xOffsetLocalI32 = tmpBuf2_.Get<int32_t>();
        ArithProgression(xOffsetLocalI32, static_cast<int32_t>(0), static_cast<int32_t>(sizeof(float) * SWI_FACTOR),
                         static_cast<int32_t>(ubMaxPair_));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> xOffsetLocalU32 = xOffsetLocalI32.template ReinterpretCast<uint32_t>();
        Gather(tmpB, xFloatLocal, xOffsetLocalU32, static_cast<uint32_t>(4), pairNum_);
        Gather(tmpA, xFloatLocal, xOffsetLocalU32, static_cast<uint32_t>(0), pairNum_);
        PipeBarrier<PIPE_V>();
    }
}

} // namespace ClippedSwigluGradOps
#endif // OPP_CLIPPED_SWIGLU_GRAD_H
