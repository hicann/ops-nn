/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_OPS_SOFT_MARGIN_LOSS_REDUCE_H
#define CANN_OPS_SOFT_MARGIN_LOSS_REDUCE_H

#include "atvoss/reduce/reduce_operator.h"

namespace SoftMarginLoss {
using namespace AscendC;
using namespace Ops::Base;

/**
 * PyTorch CPU's FP32 sum is sensitive to the accumulation tree.  For SoftMarginLoss sum, first combine adjacent
 * elements and then invoke the platform ReduceSum.  This is a shape-independent pairwise reduction, not a
 * case-specific output correction.  The ATVOSS cache, bisection and padding behavior remains inherited from
 * Vec::ReduceSumOp.
 */
template <typename T>
class PairwiseReduceSumOp : public Vec::ReduceSumOp<T> {
private:
#ifdef __CCE_AICORE__
    static __simd_vf__ inline void PairwiseAddInPlace(__ubuf__ T* srcAddr, uint32_t dimR)
    {
        constexpr uint32_t vlSize = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        const uint32_t pairCount = dimR >> 1;
        const uint16_t loopNum = static_cast<uint16_t>((pairCount + vlSize - 1U) / vlSize);
        uint32_t remain = pairCount;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> evenValue;
            AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> oddValue;
            AscendC::MicroAPI::RegTensor<uint32_t, AscendC::MicroAPI::RegTraitNumOne> evenIndex;
            AscendC::MicroAPI::RegTensor<uint32_t, AscendC::MicroAPI::RegTraitNumOne> oddIndex;
            AscendC::MicroAPI::MaskReg mask;

            for (uint16_t loopIdx = 0; loopIdx < loopNum; ++loopIdx) {
                mask = AscendC::MicroAPI::UpdateMask<uint32_t, AscendC::MicroAPI::RegTraitNumOne>(remain);
                const uint32_t outOffset = static_cast<uint32_t>(loopIdx) * vlSize;
                AscendC::MicroAPI::Arange(
                    reinterpret_cast<AscendC::MicroAPI::RegTensor<int32_t, AscendC::MicroAPI::RegTraitNumOne>&>(
                        evenIndex),
                    static_cast<int32_t>(outOffset));
                AscendC::MicroAPI::Muls(evenIndex, evenIndex, static_cast<uint32_t>(2), mask);
                AscendC::MicroAPI::Adds(oddIndex, evenIndex, static_cast<uint32_t>(1), mask);
                AscendC::MicroAPI::Gather(evenValue, srcAddr, evenIndex, mask);
                AscendC::MicroAPI::Gather(oddValue, srcAddr, oddIndex, mask);
                AscendC::MicroAPI::Add(evenValue, evenValue, oddValue, mask);
                AscendC::MicroAPI::DataCopy(srcAddr + outOffset, evenValue, mask);
            }

            if ((dimR & 1U) != 0U) {
                uint32_t one = 1;
                mask = AscendC::MicroAPI::UpdateMask<uint32_t, AscendC::MicroAPI::RegTraitNumOne>(one);
                AscendC::MicroAPI::Duplicate(evenIndex, dimR - 1U);
                AscendC::MicroAPI::Gather(evenValue, srcAddr, evenIndex, mask);
                AscendC::MicroAPI::DataCopy(srcAddr + pairCount, evenValue, mask);
            }

            AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE,
                                           AscendC::MicroAPI::MemType::VEC_LOAD>();
        }
    }
#endif

public:
    __aicore__ inline PairwiseReduceSumOp() {}

#ifdef __CCE_AICORE__
    template <class Pattern, bool isBatchInvariant,
              typename ReduceOpTmpl::IsSameV<Pattern, ReduceOpTmpl::__reducePattern::AR>::Type* dummy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<T>& dst, const LocalTensor<T>& src)
    {
        const uint32_t dimA = static_cast<uint32_t>(shape.value[0]);
        const uint32_t dimR = static_cast<uint32_t>(shape.value[1]);
        if (dimA != 1U || dimR <= 1U) {
            Vec::ReduceSumOp<T>::template Compute<Pattern, isBatchInvariant>(shape, dst, src);
            return;
        }

        LocalTensor<T> work = src;
        PairwiseAddInPlace((__ubuf__ T*)work.GetPhyAddr(), dimR);

        uint32_t pairShape[2] = {1U, (dimR + 1U) >> 1};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(dst, work, pairShape, false);
    }

    template <class Pattern, bool isBatchInvariant,
              typename ReduceOpTmpl::IsSameV<Pattern, ReduceOpTmpl::__reducePattern::RA>::Type* dummy = nullptr>
    __aicore__ inline void Compute(ReduceOpTmpl::Shape<2>& shape, const LocalTensor<T>& dst, const LocalTensor<T>& src)
    {
        Vec::ReduceSumOp<T>::template Compute<Pattern, isBatchInvariant>(shape, dst, src);
    }
#endif
};

} // namespace SoftMarginLoss

#endif // CANN_OPS_SOFT_MARGIN_LOSS_REDUCE_H
