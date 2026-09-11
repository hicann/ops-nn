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
 * \file foreach_unary_regbase_common.h
 * \brief Register-only runner shared by the Ascend950 Foreach unary operators.
 */

#ifndef FOREACH_UNARY_REGBASE_COMMON_H
#define FOREACH_UNARY_REGBASE_COMMON_H

#include "foreach_flat_unary_regbase.h"

namespace ForeachUnaryRegbase {
using namespace AscendC;

constexpr static Reg::CastTrait UNARY_CAST_TO_FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr static Reg::CastTrait UNARY_CAST_TO_STORAGE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

// BF16, and FP16 policies that need FP32 precision, use 64 logical lanes so
// unpack/cast/compute/cast/pack remain register-only.
template <typename T, typename Operation>
__simd_vf__ inline void UnaryPromoteVF(__ubuf__ T* inputAddr, __ubuf__ T* outputAddr, uint32_t dataCount)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataCount, VL));
    uint32_t remain = dataCount;
    Reg::MaskReg mask;
    Reg::RegTensor<T, Reg::RegTraitNumOne> inputStorage;
    Reg::RegTensor<T, Reg::RegTraitNumOne> outputStorage;
    Reg::RegTensor<float, Reg::RegTraitNumOne> inputFloat;
    Reg::RegTensor<float, Reg::RegTraitNumOne> outputFloat;
    for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
        mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(remain);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputStorage, inputAddr + repeatIdx * VL);
        Reg::Cast<float, T, UNARY_CAST_TO_FLOAT>(inputFloat, inputStorage, mask);
        Operation::template Apply<float>(outputFloat, inputFloat, mask);
        Reg::Cast<T, float, UNARY_CAST_TO_STORAGE>(outputStorage, outputFloat, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputAddr + repeatIdx * VL, outputStorage, mask);
    }
}

// The active native policies use FP16, FP32, or INT32. Keeping the native type
// in RegTensor also prevents integers from entering the BF16 conversion path.
template <typename T, typename Operation>
__simd_vf__ inline void UnaryNativeVF(__ubuf__ T* inputAddr, __ubuf__ T* outputAddr, uint32_t dataCount)
{
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "unsupported native unary storage width");
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(T);
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(dataCount, VL));
    uint32_t remain = dataCount;
    Reg::MaskReg mask;
    Reg::RegTensor<T, Reg::RegTraitNumOne> inputReg;
    Reg::RegTensor<T, Reg::RegTraitNumOne> outputReg;
    for (uint16_t repeatIdx = 0; repeatIdx < repeatTimes; ++repeatIdx) {
        mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(remain);
        Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(inputReg, inputAddr + repeatIdx * VL);
        Operation::template Apply<T>(outputReg, inputReg, mask);
        if constexpr (sizeof(T) == 2) {
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B16>(outputAddr + repeatIdx * VL, outputReg, mask);
        } else {
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(outputAddr + repeatIdx * VL, outputReg, mask);
        }
    }
}

/**
 * Compute policy used by FlatUnaryKernel.
 *
 * BF16 is always unpacked and computed as FP32. Operations that request it do
 * the same for FP16. All casts stay in vector registers, so Compute owns no UB.
 */
template <typename T, typename Operation>
struct UnaryRegComputePolicy {
    static constexpr bool kUsesExtraUb = false;

    __aicore__ static inline void Run(LocalTensor<T> input, LocalTensor<T> output, uint32_t dataCount)
    {
        __ubuf__ T* inputAddr = reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr());
        __ubuf__ T* outputAddr = reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr());

        constexpr bool PROMOTE_TO_FLOAT = IsSameType<T, bfloat16_t>::value ||
                                          (IsSameType<T, half>::value && Operation::kPromoteHalf);
        if constexpr (PROMOTE_TO_FLOAT) {
            asc_vf_call<UnaryPromoteVF<T, Operation>>(inputAddr, outputAddr, dataCount);
        } else {
            asc_vf_call<UnaryNativeVF<T, Operation>>(inputAddr, outputAddr, dataCount);
        }
    }
};

template <typename T, typename Tiling, typename Operation>
using UnaryApplyKernel = ForeachFlatRegbase::FlatUnaryKernel<T, Tiling, UnaryRegComputePolicy<T, Operation>>;

template <typename T, typename Tiling, typename ComputePolicy>
using UnaryCustomComputeKernel = ForeachFlatRegbase::FlatUnaryKernel<T, Tiling, ComputePolicy>;

// Compatibility alias for unary operators migrated before the two extension
// points were named explicitly. New simple unary operators should prefer
// UnaryApplyKernel; operators that own the full Run loop should use
// UnaryCustomComputeKernel.
template <typename T, typename Tiling, typename Operation>
using UnaryRegbaseKernel = UnaryApplyKernel<T, Tiling, Operation>;
} // namespace ForeachUnaryRegbase

#endif // FOREACH_UNARY_REGBASE_COMMON_H
