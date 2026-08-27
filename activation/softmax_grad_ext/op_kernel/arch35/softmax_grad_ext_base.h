/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_ext_base.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_EXT_BASE_H
#define SOFTMAX_GRAD_EXT_BASE_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace SoftmaxGradExt {
using namespace AscendC;
constexpr static AscendC::Reg::CastTrait castTraitFp16ToFp32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::Reg::CastTrait castTraitFp32ToFp16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_THREE = 3;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_FIVE = 5;
constexpr static int64_t CONST_SIX = 6;
constexpr static int64_t CONST_SEVEN = 7;
constexpr static int64_t CONST_EIGHT = 8;
constexpr static int64_t CONST_SIXTY_THREE = 63;
constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) /
                                    sizeof(float); // 一个向量寄存器可以容纳多少个float元素

class SoftmaxGradExtBase {
public:
    __aicore__ inline SoftmaxGradExtBase() : pipe_(nullptr){};

protected:
    __aicore__ inline static int64_t FindNearestPower2(const int64_t value);
    __aicore__ inline static int64_t GetCacheID(const int64_t idx);

protected:
    __aicore__ inline static void LastReduceSumSmallR(const LocalTensor<float>& dstTensor,
                                                      const LocalTensor<float>& srcTensor, const int64_t aSize,
                                                      const int64_t rSize, const int64_t stride);
    __aicore__ inline static void LastReduceSum(const LocalTensor<float>& dstTensor,
                                                const LocalTensor<float>& srcTensor,
                                                const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize,
                                                const int64_t rSize, const int64_t stride);
    __aicore__ inline static void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                              const int64_t cacheID, const int64_t stride, const int64_t count);

protected:
    TPipe* pipe_;
}; // class SoftmaxGradExtBase

// IMPL
__aicore__ inline int64_t SoftmaxGradExtBase::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - clz(num);
        return (CONST_ONE << pow);
    }
}

__aicore__ inline int64_t SoftmaxGradExtBase::GetCacheID(const int64_t idx)
{
    return bcnt1(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

__aicore__ inline void SoftmaxGradExtBase::LastReduceSumSmallR(const LocalTensor<float>& dstTensor,
                                                               const LocalTensor<float>& srcTensor, const int64_t aSize,
                                                               const int64_t rSize, const int64_t stride)
{
    // LastReduceSumSmallR
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    if (rSize <= VL_FP32) {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::Reg::RegTensor<float> aReg, bReg;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::UnalignRegForStore UReg;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                Reg::LoadAlign(aReg, (__ubuf__ float*)src + i * stride);
                Reduce<ReduceType::SUM>(bReg, aReg, pMask);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::Reg::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    } else {
        __VEC_SCOPE__
        {
            __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
            __ubuf__ float* src0 = (__ubuf__ float*)srcTensor.GetPhyAddr();
            __ubuf__ float* src1 = (__ubuf__ float*)srcTensor.GetPhyAddr() + VL_FP32;
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            AscendC::Reg::UnalignRegForStore UReg;
            AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < loopTimes; ++i) {
                Reg::LoadAlign(aReg, (__ubuf__ float*)src0 + i * stride);
                Reg::LoadAlign(bReg, (__ubuf__ float*)src1 + i * stride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::Reg::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                Reduce<ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::Reg::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    }
}

__aicore__ inline void SoftmaxGradExtBase::LastReduceSum(const LocalTensor<float>& dstTensor,
                                                         const LocalTensor<float>& srcTensor,
                                                         const LocalTensor<float>& reduceSumTempTensor,
                                                         const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    // LastReduceSum
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize <= CONST_TWO * VL_FP32) {
        LastReduceSumSmallR(dstTensor, srcTensor, aSize, rSize, stride);
        return;
    }

    int64_t ceilVLCount = ops::CeilDiv(static_cast<int64_t>(rSize * sizeof(float)),
                                       static_cast<int64_t>(platform::GetVRegSize()));
    int64_t floorVLCount = ops::FloorDiv(static_cast<int64_t>(rSize * sizeof(float)),
                                         static_cast<int64_t>(platform::GetVRegSize()));
    int64_t foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = ceilVLCount - floorVLCount;
    uint32_t tailFoldElemCount = static_cast<uint32_t>(rSize - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = floorVLCount - foldPoint;
    uint16_t unFoldLoopTimes = foldPoint + foldPoint - ceilVLCount;
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL_FP32;
    uint32_t outerLoopDstStride = ops::Aligned(static_cast<int64_t>(foldPoint),
                                               static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __VEC_SCOPE__
    {
        __ubuf__ float* dst = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr();
        __ubuf__ float* foldSrcA = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ float* foldSrcB = (__ubuf__ float*)srcTensor.GetPhyAddr() + foldSrcBOffset;
        __ubuf__ float* tailSrcA = (__ubuf__ float*)srcTensor.GetPhyAddr() + tailSrcAOffset;
        __ubuf__ float* tailSrcB = (__ubuf__ float*)srcTensor.GetPhyAddr() + tailSrcBOffset;
        __ubuf__ float* unFoldSrc = (__ubuf__ float*)srcTensor.GetPhyAddr() + unFoldSrcOffset;
        AscendC::Reg::MaskReg pFull = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::UnalignRegForStore UReg;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__ubuf__ float*)reduceSumTempTensor.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::Reg::RegTensor<float> aReg, bReg, cReg, dReg;
                Reg::LoadAlign(aReg, (__ubuf__ float*)foldSrcA + i * outerLoopStride + j * innerLoopStride);
                Reg::LoadAlign(bReg, (__ubuf__ float*)foldSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pFull);
                Reduce<ReduceType::SUM>(dReg, cReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, dReg, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
                AscendC::Reg::MaskReg pMask = AscendC::Reg::UpdateMask<float>(count);
                Reg::LoadAlign(aReg, (__ubuf__ float*)tailSrcA + i * outerLoopStride + j * innerLoopStride);
                Reg::LoadAlign(bReg, (__ubuf__ float*)tailSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Move<float, AscendC::Reg::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                Reduce<ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                Reg::LoadAlign(aReg, (__ubuf__ float*)unFoldSrc + i * outerLoopStride + j * innerLoopStride);
                Reduce<ReduceType::SUM>(bReg, aReg, pFull);
                AscendC::Reg::StoreUnAlign((__ubuf__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::Reg::StoreUnAlignPost((__ubuf__ float*&)dst, UReg, 0);
        }
    }
    LastReduceSumSmallR(dstTensor, reduceSumTempTensor, aSize, foldPoint, outerLoopDstStride);
}

template <uint32_t RSize, int32_t TailCount = -1, int32_t Index = 0, int32_t Depth = 1>
struct NlastDichotomyAdd {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        __ubuf__ float* srcAOffset = srcA + stride * CONST_TWO;
        __ubuf__ float* srcBOffset = srcB + stride * CONST_TWO;
        if constexpr (TailCount <= 0) {
            NlastDichotomyAdd<(RSize + 1) / CONST_TWO>::LoadAndAccumulate(aReg, srcA, srcAOffset, pMask,
                                                                          stride * CONST_TWO);
            NlastDichotomyAdd<RSize / CONST_TWO>::LoadAndAccumulate(bReg, srcB, srcBOffset, pMask, stride * CONST_TWO);
        }
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        __ubuf__ float* srcAOffset = srcA + stride * CONST_TWO;
        __ubuf__ float* srcBOffset = srcB + stride * CONST_TWO;
        if constexpr (TailCount <= 0) {
            NlastDichotomyAdd<(RSize + 1) / CONST_TWO>::LoadAndAccumulate(aReg, srcA, srcAOffset, pMask,
                                                                          stride * CONST_TWO, offset);
            NlastDichotomyAdd<RSize / CONST_TWO>::LoadAndAccumulate(bReg, srcB, srcBOffset, pMask, stride * CONST_TWO,
                                                                    offset);
        } else {
            NlastDichotomyAdd<(RSize + 1) / CONST_TWO, TailCount, Index, Depth * CONST_TWO>::LoadAndAccumulate(
                aReg, srcA, srcAOffset, pMask, stride * CONST_TWO, offset);
            NlastDichotomyAdd<RSize / CONST_TWO, TailCount, Index + Depth, Depth * CONST_TWO>::LoadAndAccumulate(
                bReg, srcB, srcBOffset, pMask, stride * CONST_TWO, offset);
        }
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
};

template <int32_t TailCount, int32_t Index, int32_t Depth>
struct NlastDichotomyAdd<CONST_TWO, TailCount, Index, Depth> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        Reg::LoadAlign(aReg, (__ubuf__ float*)srcA);
        Reg::LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        if constexpr (TailCount <= 0) {
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            Reg::LoadAlign(aReg, (__ubuf__ float*)srcA);
            Reg::LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            Reg::LoadAlign(bReg, (__ubuf__ float*)srcB);
            Reg::LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
        } else {
            if constexpr (Index + Depth < TailCount) {
                AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
                Reg::LoadAlign(aReg, (__ubuf__ float*)srcA);
                Reg::LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                Reg::LoadAlign(bReg, (__ubuf__ float*)srcB);
                Reg::LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else if constexpr (Index < TailCount) {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                Reg::LoadAlign(aReg, (__ubuf__ float*)srcA);
                Reg::LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                Reg::LoadAlign(bReg, (__ubuf__ float*)srcB);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                Reg::LoadAlign(aReg, (__ubuf__ float*)srcA);
                Reg::LoadAlign(bReg, (__ubuf__ float*)srcB);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            }
        }
    }
};

template <>
struct NlastDichotomyAdd<CONST_TWO> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        Reg::LoadAlign(aReg, (__ubuf__ float*)srcA);
        Reg::LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
        Reg::LoadAlign(aReg, (__ubuf__ float*)srcA);
        Reg::LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
        Reg::LoadAlign(bReg, (__ubuf__ float*)srcB);
        Reg::LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
};

template <>
struct NlastDichotomyAdd<1> {
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride)
    {
        Reg::LoadAlign(acc, (__ubuf__ float*)srcA);
    }
};

__aicore__ inline void SoftmaxGradExtBase::UpdateCache(const LocalTensor<float>& dstTensor,
                                                       const LocalTensor<float>& srcTensor, const int64_t cacheID,
                                                       const int64_t stride, const int64_t count)
{
    // UpdateCache
    uint16_t outerLoopTimes = ops::CeilDiv(static_cast<int64_t>(count * sizeof(float)),
                                           static_cast<int64_t>(platform::GetVRegSize()));
    uint16_t innerLoopTimes = cacheID;
    uint32_t outerLoopStride = VL_FP32;
    uint32_t innerLoopStride = stride;
    __VEC_SCOPE__
    {
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ float* cah = (__ubuf__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
        __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::Reg::RegTensor<float> aReg, bReg;
        AscendC::Reg::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(sreg);
            Reg::LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                Reg::LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            Reg::StoreAlign((__ubuf__ float*)cah + i * outerLoopStride, aReg, pMask);
        }
    }
}
} // namespace SoftmaxGradExt
#endif
