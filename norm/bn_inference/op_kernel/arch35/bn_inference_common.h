/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN_INFERENCE_COMMON_H
#define BN_INFERENCE_COMMON_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace BNInferenceOps {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();
constexpr uint16_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait CAST_B16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline int64_t AlignUpBlock(int64_t bytes)
{
    return (bytes + UB_BLOCK_BYTES - 1) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
}

template <typename T>
__aicore__ inline void LoadToFp32(RegTensor<float>& dst, __ubuf__ T* src, MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::LoadAlign<float, LoadDist::DIST_NORM>(dst, (__ubuf__ float*)src + offset);
    } else {
        RegTensor<T> b16;
        AscendC::Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(b16, src + offset);
        AscendC::Reg::Cast<float, T, CAST_B16_TO_FP32>(dst, b16, mask);
    }
}

template <typename T>
__aicore__ inline void LoadBroadcastToFp32(RegTensor<float>& dst, __ubuf__ T* src, MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(dst, (__ubuf__ float*)src + offset);
    } else {
        RegTensor<T> b16;
        AscendC::Reg::LoadAlign<T, LoadDist::DIST_BRC_B16>(b16, src + offset);
        AscendC::Reg::Cast<float, T, CAST_B16_TO_FP32>(dst, b16, mask);
    }
}

template <typename T>
__aicore__ inline void StoreFromFp32(__ubuf__ T* dst, RegTensor<float>& src, MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::StoreAlign<float, StoreDist::DIST_NORM>((__ubuf__ float*)dst + offset, src, mask);
    } else {
        RegTensor<T> b16;
        AscendC::Reg::Cast<T, float, CAST_FP32_TO_B16>(b16, src, mask);
        AscendC::Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + offset, b16, mask);
    }
}

template <typename T>
__aicore__ inline void RoundTripFp32(RegTensor<float>& value, MaskReg& mask)
{
    if constexpr (!IsSameType<T, float>::value) {
        RegTensor<T> rounded;
        AscendC::Reg::Cast<T, float, CAST_FP32_TO_B16>(rounded, value, mask);
        AscendC::Reg::Cast<float, T, CAST_B16_TO_FP32>(value, rounded, mask);
    }
}

template <typename T>
__aicore__ inline void LoadUnalignedToFp32(__ubuf__ T*& src, RegTensor<float>& dst,
                                           AscendC::MicroAPI::UnalignRegForLoad& state, MaskReg& mask,
                                           uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::MicroAPI::LoadUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, state, src,
                                                                                                postUpdateStride);
    } else {
        RegTensor<T> b16;
        RegTensor<T> unpacked;
        AscendC::MicroAPI::LoadUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(b16, state, src,
                                                                                            postUpdateStride);
        AscendC::MicroAPI::UnPack((RegTensor<uint32_t>&)unpacked, (RegTensor<uint16_t>&)b16);
        AscendC::Reg::Cast<float, T, CAST_B16_TO_FP32>(dst, unpacked, mask);
    }
}

template <typename T>
__aicore__ inline void StoreUnalignedFromFp32(__ubuf__ T*& dst, RegTensor<float>& src,
                                              AscendC::MicroAPI::UnalignRegForStore& state, MaskReg& mask,
                                              uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::MicroAPI::StoreUnAlign<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, src, state,
                                                                                                 postUpdateStride);
    } else {
        RegTensor<T> b16;
        RegTensor<T> packed;
        AscendC::Reg::Cast<T, float, CAST_FP32_TO_B16>(b16, src, mask);
        AscendC::MicroAPI::Pack((RegTensor<uint16_t>&)packed, (RegTensor<uint32_t>&)b16);
        AscendC::MicroAPI::StoreUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, packed, state,
                                                                                             postUpdateStride);
    }
}

template <typename T>
__aicore__ inline void LoadUnalignedOnce(__ubuf__ T* src, RegTensor<float>& dst, MaskReg& mask, uint32_t count)
{
    AscendC::MicroAPI::UnalignRegForLoad state;
    __ubuf__ T* current = src;
    AscendC::MicroAPI::LoadUnAlignPre(state, current);
    LoadUnalignedToFp32(current, dst, state, mask, count);
}

__aicore__ inline void LoadOffsetUnaligned(__ubuf__ uint32_t* src, RegTensor<uint32_t>& dst, uint32_t count)
{
    AscendC::MicroAPI::UnalignRegForLoad state;
    __ubuf__ uint32_t* current = src;
    AscendC::MicroAPI::LoadUnAlignPre(state, current);
    AscendC::MicroAPI::LoadUnAlign<uint32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, state, current,
                                                                                               count);
}

template <typename T>
__aicore__ inline void GatherToFp32(__ubuf__ T* src, RegTensor<float>& dst, RegTensor<uint32_t>& offsets,
                                    MaskReg& fp32Mask, uint32_t count)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::MicroAPI::Gather(dst, (__ubuf__ float*)src, offsets, fp32Mask);
    } else {
        uint32_t maskCount = count;
        MaskReg b16Mask = AscendC::MicroAPI::UpdateMask<T>(maskCount);
        RegTensor<uint16_t> offsetsB16;
        RegTensor<T> gathered;
        RegTensor<T> unpacked;
        AscendC::MicroAPI::Pack(offsetsB16, offsets);
        AscendC::MicroAPI::Gather(gathered, src, offsetsB16, b16Mask);
        AscendC::MicroAPI::UnPack((RegTensor<uint32_t>&)unpacked, (RegTensor<uint16_t>&)gathered);
        AscendC::Reg::Cast<float, T, CAST_B16_TO_FP32>(dst, unpacked, fp32Mask);
    }
}

__aicore__ inline void ComputeRstdExact(RegTensor<float>& variance, RegTensor<float>& rstd, MaskReg& mask,
                                        float epsilon)
{
    // The intrinsic FP32 division may differ by 1 ULP.  That difference is
    // observable around zero after the affine add, so use the 0-ULP division
    // mode for the reciprocal while retaining Sqrt's full input range.
    static constexpr AscendC::Reg::DivSpecificMode kHighPrecisionDiv = {AscendC::Reg::MaskMergeMode::ZEROING, true};
    RegTensor<float> epsilonReg;
    RegTensor<float> oneReg;
    AscendC::Reg::Duplicate(epsilonReg, epsilon, mask);
    AscendC::Reg::Add(variance, variance, epsilonReg, mask);
    AscendC::Reg::Sqrt(rstd, variance, mask);
    AscendC::Reg::Duplicate(oneReg, 1.0f, mask);
    AscendC::Reg::Div<float, &kHighPrecisionDiv>(rstd, oneReg, rstd, mask);
}

template <bool HAS_SCALE, bool HAS_OFFSET>
__aicore__ inline void NormalizeOneSided(RegTensor<float>& x, RegTensor<float>& mean, RegTensor<float>& rstd,
                                         RegTensor<float>& gamma, RegTensor<float>& beta, RegTensor<float>& y,
                                         MaskReg& mask)
{
    static_assert(HAS_SCALE != HAS_OFFSET, "NormalizeOneSided requires exactly one affine input");
    AscendC::Reg::Sub(y, x, mean, mask);
    if constexpr (HAS_OFFSET) {
        // Native inference BatchNorm evaluates (x - mean) * rstd + offset as
        // one fused multiply-add.  MulDstAdd preserves that rounding and also
        // removes the redundant multiplication by the implicit scale 1.
        AscendC::Reg::MulDstAdd(y, rstd, beta, mask);
    } else {
        AscendC::Reg::Mul(y, y, rstd, mask);
        AscendC::Reg::Mul(y, y, gamma, mask);
    }
}

template <typename T_MEAN, typename T_VARIANCE>
__aicore__ inline void FoldWithoutAffine(RegTensor<float>& mean, RegTensor<float>& variance, RegTensor<float>& alpha,
                                         RegTensor<float>& beta, MaskReg& mask, float epsilon, float factor)
{
    RegTensor<float> factorReg;
    RegTensor<float> negFactorReg;
    RegTensor<float> epsilonReg;
    RegTensor<float> oneReg;
    AscendC::Reg::Duplicate(factorReg, factor, mask);
    AscendC::Reg::Duplicate(negFactorReg, -factor, mask);
    AscendC::Reg::Mul(alpha, negFactorReg, mean, mask);
    AscendC::Reg::Mul(variance, variance, factorReg, mask);
    AscendC::Reg::Duplicate(epsilonReg, epsilon, mask);
    AscendC::Reg::Add(variance, variance, epsilonReg, mask);
    AscendC::Reg::Sqrt(beta, variance, mask);
    AscendC::Reg::Duplicate(oneReg, 1.0f, mask);
    AscendC::Reg::Div(beta, oneReg, beta, mask);
    RoundTripFp32<T_MEAN>(alpha, mask);
    RoundTripFp32<T_VARIANCE>(beta, mask);
}

template <typename T_MEAN, typename T_VARIANCE>
__aicore__ inline void FoldWithAffine(RegTensor<float>& mean, RegTensor<float>& variance, RegTensor<float>& scale,
                                      RegTensor<float>& offset, RegTensor<float>& alpha, RegTensor<float>& beta,
                                      MaskReg& mask, float epsilon)
{
    RegTensor<float> epsilonReg;
    RegTensor<float> oneReg;
    AscendC::Reg::Duplicate(epsilonReg, epsilon, mask);
    AscendC::Reg::Add(variance, variance, epsilonReg, mask);
    AscendC::Reg::Sqrt(variance, variance, mask);
    AscendC::Reg::Duplicate(oneReg, 1.0f, mask);
    AscendC::Reg::Div(beta, oneReg, variance, mask);
    AscendC::Reg::Mul(beta, scale, beta, mask);
    AscendC::Reg::Div(alpha, offset, scale, mask);
    AscendC::Reg::Mul(alpha, alpha, variance, mask);
    AscendC::Reg::Sub(alpha, alpha, mean, mask);
    RoundTripFp32<T_MEAN>(alpha, mask);
    RoundTripFp32<T_VARIANCE>(beta, mask);
}

template <typename T_X>
__aicore__ inline void ApplyFoldedAffine(RegTensor<float>& x, RegTensor<float>& alpha, RegTensor<float>& beta,
                                         RegTensor<float>& y, MaskReg& mask)
{
    AscendC::Reg::Add(y, x, alpha, mask);
    RoundTripFp32<T_X>(y, mask);
    AscendC::Reg::Mul(y, y, beta, mask);
}

template <typename T_X, bool HAS_SCALE, bool HAS_OFFSET>
__aicore__ inline void ApplyPreFoldedAffine(RegTensor<float>& x, RegTensor<float>& alpha, RegTensor<float>& beta,
                                            RegTensor<float>& scale, RegTensor<float>& offset, RegTensor<float>& y,
                                            MaskReg& mask)
{
    static_assert(!HAS_OFFSET || HAS_SCALE, "mode=0 does not support offset without scale");
    ApplyFoldedAffine<T_X>(x, alpha, beta, y, mask);
    if constexpr (HAS_SCALE) {
        // BNInferenceD materializes the base Mul in x.dtype before the optional
        // affine stage promotes it back to FP32 on supported products.
        RoundTripFp32<T_X>(y, mask);
        AscendC::Reg::Mul(y, y, scale, mask);
        if constexpr (HAS_OFFSET) {
            AscendC::Reg::Add(y, y, offset, mask);
        }
    }
}
} // namespace BNInferenceOps

#endif // BN_INFERENCE_COMMON_H
