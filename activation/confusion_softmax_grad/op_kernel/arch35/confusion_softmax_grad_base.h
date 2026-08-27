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
 * \file confusion_softmax_grad_base.h
 * \brief
 */

#ifndef NORM_CONFUSION_SOFTMAX_GRAD_BASE_H
#define NORM_CONFUSION_SOFTMAX_GRAD_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ConfusionSoftmaxGradOps {
using namespace AscendC;
using AscendC::Reg::LoadAlign;

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
constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);

class ConfusionSoftmaxGradOpsBase {
public:
    __aicore__ inline ConfusionSoftmaxGradOpsBase() : pipe_(nullptr){};

protected:
    __aicore__ inline static int64_t FindNearestPower2(const int64_t value);
    __aicore__ inline static int64_t GetCacheID(const int64_t idx);

protected:
    __aicore__ inline static void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                              const int64_t cacheID, const int64_t stride, const int64_t count);
    __aicore__ inline static void Normalize(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                            const LocalTensor<float>& meanTensor, const LocalTensor<float>& rstdTensor,
                                            const int64_t rowSize, const int64_t colSize);

protected:
    TPipe* pipe_;
}; // class ConfusionSoftmaxGradOpsBase

// IMPL
__aicore__ inline int64_t ConfusionSoftmaxGradOpsBase::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - AscendC::ScalarCountLeadingZero(num);
        return (CONST_ONE << pow);
    }
}

__aicore__ inline int64_t ConfusionSoftmaxGradOpsBase::GetCacheID(const int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + CONST_ONE)) - CONST_ONE;
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
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        if constexpr (TailCount <= 0) {
            AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
            LoadAlign(aReg, (__ubuf__ float*)srcA);
            LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            LoadAlign(bReg, (__ubuf__ float*)srcB);
            LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
        } else {
            if constexpr (Index + Depth < TailCount) {
                AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, bReg, cReg, pMask);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else if constexpr (Index < TailCount) {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
            } else {
                AscendC::Reg::RegTensor<float> aReg, bReg;
                LoadAlign(aReg, (__ubuf__ float*)srcA);
                LoadAlign(bReg, (__ubuf__ float*)srcB);
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
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(acc, aReg, bReg, pMask);
    }
    __aicore__ static inline void LoadAndAccumulate(AscendC::Reg::RegTensor<float>& acc, __ubuf__ float*& srcA,
                                                    __ubuf__ float*& srcB, AscendC::Reg::MaskReg& pMask,
                                                    uint32_t stride, uint32_t offset)
    {
        AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
        LoadAlign(aReg, (__ubuf__ float*)srcA);
        LoadAlign(bReg, (__ubuf__ float*)srcA + offset);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
        LoadAlign(bReg, (__ubuf__ float*)srcB);
        LoadAlign(cReg, (__ubuf__ float*)srcB + offset);
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
        LoadAlign(acc, (__ubuf__ float*)srcA);
    }
};

__aicore__ inline void ConfusionSoftmaxGradOpsBase::UpdateCache(const LocalTensor<float>& dstTensor,
                                                                const LocalTensor<float>& srcTensor,
                                                                const int64_t cacheID, const int64_t stride,
                                                                const int64_t count)
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
            LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                LoadAlign(bReg, (__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            StoreAlign((__ubuf__ float*)cah + i * outerLoopStride, aReg, pMask);
        }
    }
}

__aicore__ inline void ConfusionSoftmaxGradOpsBase::Normalize(const LocalTensor<float>& dstTensor,
                                                              const LocalTensor<float>& srcTensor,
                                                              const LocalTensor<float>& meanTensor,
                                                              const LocalTensor<float>& rstdTensor,
                                                              const int64_t rowSize, const int64_t colSize)
{
    // Normalize
    uint16_t outerLoopTimes = rowSize;
    uint16_t innerLoopTimes = ops::CeilDiv(static_cast<int64_t>(colSize * sizeof(float)),
                                           static_cast<int64_t>(platform::GetVRegSize()));
    uint32_t outerLoopStride = colSize;
    uint32_t innerLoopStride = VL_FP32;
    __VEC_SCOPE__
    {
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ float* src = (__ubuf__ float*)srcTensor.GetPhyAddr();
        __ubuf__ float* mean = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* rstd = (__ubuf__ float*)rstdTensor.GetPhyAddr();
        uint32_t count;
        AscendC::Reg::RegTensor<float> aReg, bReg, cReg;
        AscendC::Reg::RegTensor<float> meanReg, rstdReg;
        AscendC::Reg::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            count = static_cast<uint32_t>(colSize);
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(meanReg, (__ubuf__ float*)mean + i);
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstdReg, (__ubuf__ float*)rstd + i);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                pMask = AscendC::Reg::UpdateMask<float>(count);
                LoadAlign(aReg, (__ubuf__ float*)src + i * outerLoopStride + j * innerLoopStride);
                Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(bReg, aReg, meanReg, pMask);
                Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(cReg, bReg, rstdReg, pMask);
                StoreAlign((__ubuf__ float*)dst + i * outerLoopStride + j * innerLoopStride, cReg, pMask);
            }
        }
    }
}
} // namespace ConfusionSoftmaxGradOps

#endif // NORM_CONFUSION_SOFTMAX_GRAD_BASE_H
