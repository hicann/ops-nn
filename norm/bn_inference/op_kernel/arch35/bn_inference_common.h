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
#include "bn_inference_compensated.h"
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
constexpr float FP32_MAX_FINITE = 0x1.fffffep127f;
// Preserve subnormal radicands while retaining the full FP32 input range.
constexpr AscendC::Reg::SqrtSpecificMode SQRT_NO_FTZ_MODE = {MaskMergeMode::ZEROING, false,
                                                             AscendC::SqrtAlgo::PRECISION_1ULP_FTZ_FALSE};

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
    AscendC::Reg::Sqrt<float, &SQRT_NO_FTZ_MODE>(rstd, variance, mask);
    AscendC::Reg::Duplicate(oneReg, 1.0f, mask);
    AscendC::Reg::Div<float, &kHighPrecisionDiv>(rstd, oneReg, rstd, mask);
}

template <typename T_X, typename T_MEAN, typename T_VARIANCE, bool HAS_SCALE, bool HAS_OFFSET>
__aicore__ inline void NormalizeOneSided(RegTensor<float>& x, RegTensor<float>& mean, RegTensor<float>& rstd,
                                         RegTensor<float>& gamma, RegTensor<float>& beta, RegTensor<float>& y,
                                         MaskReg& mask)
{
    static_assert(HAS_SCALE != HAS_OFFSET, "NormalizeOneSided requires exactly one affine input");
    if constexpr (HAS_OFFSET) {
        // Native inference BatchNorm evaluates (x - mean) * rstd + offset as
        // one fused multiply-add.  MulDstAdd preserves that rounding and also
        // removes the redundant multiplication by the implicit scale 1.
        AscendC::Reg::Sub(y, x, mean, mask);
        AscendC::Reg::MulDstAdd(y, rstd, beta, mask);
    } else if constexpr (!IsSameType<T_X, float>::value && !IsSameType<T_MEAN, float>::value &&
                         !IsSameType<T_VARIANCE, float>::value) {
        // The promoted reference normalizes in FP32 only when x, mean and
        // variance are all FP16/BF16. Preserve that stage's overflow before
        // applying scale. Any original FP32 operand makes this stage FP64,
        // so the mixed-type paths below must recover representable results.
        AscendC::Reg::Sub(y, x, mean, mask);
        AscendC::Reg::Mul(y, y, rstd, mask);
        AscendC::Reg::Mul(y, y, gamma, mask);
    } else {
        RegTensor<float> centered;
        RegTensor<float> coefficient;
        RegTensor<float> stable;
        RegTensor<float> scaledX;
        RegTensor<float> scaledMean;
        RegTensor<float> magnitude;
        MaskReg centeredOverflow;
        MaskReg normalizedOverflow;
        MaskReg overflow;
        MaskReg finiteStable;
        MaskReg recover;
        AscendC::Reg::Sub(centered, x, mean, mask);
        AscendC::Reg::Abs(magnitude, centered, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(centeredOverflow, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Mul(y, centered, rstd, mask);
        AscendC::Reg::Abs(magnitude, y, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(normalizedOverflow, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Mul(y, y, gamma, mask);

        // Recover a representable result from an overflowing intermediate.
        // This also supplies the fallback for unrepresentable compensated lanes.
        AscendC::Reg::Mul(coefficient, rstd, gamma, mask);
        AscendC::Reg::Mul(stable, centered, coefficient, mask);
        AscendC::Reg::Mul(scaledX, x, coefficient, mask);
        AscendC::Reg::Mul(scaledMean, mean, coefficient, mask);
        AscendC::Reg::Sub(scaledX, scaledX, scaledMean, mask);
        AscendC::Reg::Select(stable, scaledX, stable, centeredOverflow);
        AscendC::Reg::Abs(magnitude, stable, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteStable, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Or(overflow, centeredOverflow, normalizedOverflow, mask);
        AscendC::Reg::And(recover, overflow, finiteStable, mask);
        AscendC::Reg::Select(y, stable, y, recover);
        if constexpr (IsSameType<T_X, float>::value && IsSameType<T_VARIANCE, float>::value) {
            RegTensor<float> negativeMean, zero, high, low, candidate;
            RegTensor<float> centeredHigh, centeredLow, alternateHigh, alternateLow;
            MaskReg finiteCandidate, smallProduct, largeProduct, alternateOrder;
            AscendC::Reg::Muls(negativeMean, mean, -1.0f, mask);
            AscendC::Reg::Duplicate(zero, 0.0f, mask);
            Compensated::TwoSum(centeredHigh, centeredLow, x, negativeMean, mask);
            // beta carries the reciprocal-square-root residual on this path.
            Compensated::MulPair(high, low, centeredHigh, centeredLow, rstd, beta, mask);
            AscendC::Reg::Abs(magnitude, high, mask);
            // The low component needs another FP32 significand below the
            // leading product. Change order before either component loses range.
            AscendC::Reg::Compares<float, CMPMODE::LT>(smallProduct, magnitude, 0x1p-102f, mask);
            AscendC::Reg::Compares<float, CMPMODE::GT>(largeProduct, magnitude, FP32_MAX_FINITE, mask);
            AscendC::Reg::Or(alternateOrder, smallProduct, largeProduct, mask);
            Compensated::MulPair(alternateHigh, alternateLow, centeredHigh, centeredLow, gamma, zero, alternateOrder);
            Compensated::MulPair(alternateHigh, alternateLow, alternateHigh, alternateLow, rstd, beta, alternateOrder);
            Compensated::MulPair(high, low, high, low, gamma, zero, mask);
            AscendC::Reg::Select(high, alternateHigh, high, alternateOrder);
            AscendC::Reg::Select(low, alternateLow, low, alternateOrder);
            AscendC::Reg::Add(candidate, high, low, mask);
            AscendC::Reg::Abs(magnitude, candidate, mask);
            AscendC::Reg::Compares<float, CMPMODE::LE>(finiteCandidate, magnitude, FP32_MAX_FINITE, mask);
            AscendC::Reg::Select(y, candidate, y, finiteCandidate);
        }
    }
}

template <typename T_MEAN, typename T_VARIANCE>
__aicore__ inline void FoldWithoutAffine(RegTensor<float>& mean, RegTensor<float>& variance, RegTensor<float>& alpha,
                                         RegTensor<float>& beta, RegTensor<float>& correction, MaskReg& mask,
                                         float epsilon, float factor)
{
    RegTensor<float> factorReg;
    RegTensor<float> negFactorReg;
    RegTensor<float> epsilonReg;
    RegTensor<float> oneReg;
    RegTensor<float> originalVariance;
    AscendC::Reg::Adds(originalVariance, variance, 0.0f, mask);
    AscendC::Reg::Duplicate(factorReg, factor, mask);
    AscendC::Reg::Duplicate(negFactorReg, -factor, mask);
    AscendC::Reg::Mul(alpha, negFactorReg, mean, mask);
    AscendC::Reg::Mul(variance, variance, factorReg, mask);
    AscendC::Reg::Duplicate(epsilonReg, epsilon, mask);
    AscendC::Reg::Add(variance, variance, epsilonReg, mask);
    AscendC::Reg::Sqrt<float, &SQRT_NO_FTZ_MODE>(beta, variance, mask);
    AscendC::Reg::Duplicate(oneReg, 1.0f, mask);
    AscendC::Reg::Div(beta, oneReg, beta, mask);
    if constexpr (IsSameType<T_MEAN, float>::value && IsSameType<T_VARIANCE, float>::value) {
        if (factor > 1.0f && factor <= FP32_MAX_FINITE) {
            // factor * variance can overflow although its reciprocal square
            // root is representable. Scale epsilon before the square root.
            RegTensor<float> scaledEpsilon;
            RegTensor<float> root;
            RegTensor<float> factorRoot;
            RegTensor<float> stableBeta;
            MaskReg overflow;
            MaskReg finiteVariance;
            MaskReg recover;
            AscendC::Reg::Compares<float, CMPMODE::GT>(overflow, variance, FP32_MAX_FINITE, mask);
            AscendC::Reg::Compares<float, CMPMODE::LE>(finiteVariance, originalVariance, FP32_MAX_FINITE, mask);
            AscendC::Reg::And(recover, overflow, finiteVariance, mask);
            AscendC::Reg::Div(scaledEpsilon, epsilonReg, factorReg, mask);
            AscendC::Reg::Add(root, originalVariance, scaledEpsilon, mask);
            AscendC::Reg::Sqrt(root, root, mask);
            AscendC::Reg::Sqrt(factorRoot, factorReg, mask);
            AscendC::Reg::Div(stableBeta, oneReg, factorRoot, mask);
            AscendC::Reg::Div(stableBeta, stableBeta, root, mask);
            AscendC::Reg::Select(beta, stableBeta, beta, recover);
        }
    }
    AscendC::Reg::Duplicate(correction, 0.0f, mask);
    if constexpr (IsSameType<T_MEAN, float>::value && IsSameType<T_VARIANCE, float>::value) {
        // For an overflowing shift, apply its finite contribution after the
        // multiply: x * beta - mean * (factor * beta).
        RegTensor<float> magnitude;
        RegTensor<float> stableCorrection;
        RegTensor<float> zero;
        MaskReg overflow;
        MaskReg finiteMean;
        MaskReg finiteShift;
        MaskReg finiteCorrection;
        MaskReg recover;
        AscendC::Reg::Abs(magnitude, alpha, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(overflow, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Abs(magnitude, mean, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteMean, magnitude, FP32_MAX_FINITE, overflow);
        AscendC::Reg::Abs(magnitude, factorReg, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteShift, magnitude, FP32_MAX_FINITE, finiteMean);
        AscendC::Reg::Mul(stableCorrection, negFactorReg, beta, mask);
        AscendC::Reg::Mul(stableCorrection, mean, stableCorrection, mask);
        AscendC::Reg::Abs(magnitude, stableCorrection, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteCorrection, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::And(recover, finiteShift, finiteCorrection, mask);
        AscendC::Reg::Duplicate(zero, 0.0f, mask);
        AscendC::Reg::Select(alpha, zero, alpha, recover);
        // A nonfinite correction also records that the shift overflow came
        // from finite inputs, so an infinite x keeps its own IEEE sign.
        AscendC::Reg::Select(correction, stableCorrection, zero, finiteShift);
    }
}

template <typename T_MEAN, typename T_VARIANCE, typename T_SCALE, typename T_OFFSET>
__aicore__ inline void FoldWithAffine(RegTensor<float>& mean, RegTensor<float>& variance, RegTensor<float>& scale,
                                      RegTensor<float>& offset, RegTensor<float>& alpha, RegTensor<float>& beta,
                                      MaskReg& mask, float epsilon)
{
    RegTensor<float> epsilonReg;
    RegTensor<float> oneReg;
    RegTensor<float> inverseSqrt;
    RegTensor<float> magnitude;
    MaskReg divisionOverflow;
    MaskReg finiteQuotient;
    MaskReg shiftOverflow;
    MaskReg alphaOverflow;
    MaskReg overflow;
    AscendC::Reg::Duplicate(epsilonReg, epsilon, mask);
    AscendC::Reg::Add(variance, variance, epsilonReg, mask);
    AscendC::Reg::Sqrt<float, &SQRT_NO_FTZ_MODE>(variance, variance, mask);
    AscendC::Reg::Duplicate(oneReg, 1.0f, mask);
    // A reciprocal error can move the final low-precision output across a
    // rounding boundary after folding. Preserve subnormal division inputs.
    static constexpr AscendC::Reg::DivSpecificMode kHighPrecisionDiv = {MaskMergeMode::ZEROING, false,
                                                                        AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    AscendC::Reg::Div<float, &kHighPrecisionDiv>(inverseSqrt, oneReg, variance, mask);
    AscendC::Reg::Mul(beta, scale, inverseSqrt, mask);
    AscendC::Reg::Div(alpha, offset, scale, mask);
    if constexpr (IsSameType<T_MEAN, float>::value && IsSameType<T_VARIANCE, float>::value) {
        // Record the division before multiplying by sqrt: an overflowing
        // quotient followed by sqrt(0) would otherwise become a false NaN.
        AscendC::Reg::Abs(magnitude, alpha, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(divisionOverflow, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteQuotient, magnitude, FP32_MAX_FINITE, mask);
    }
    AscendC::Reg::Mul(alpha, alpha, variance, mask);
    if constexpr (IsSameType<T_MEAN, float>::value && IsSameType<T_VARIANCE, float>::value) {
        AscendC::Reg::Abs(magnitude, alpha, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(shiftOverflow, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Or(overflow, divisionOverflow, shiftOverflow, mask);
    }
    AscendC::Reg::Sub(alpha, alpha, mean, mask);
    RegTensor<float> zero;
    AscendC::Reg::Duplicate(zero, 0.0f, mask);
    if constexpr (IsSameType<T_MEAN, float>::value && IsSameType<T_VARIANCE, float>::value) {
        MaskReg combinedOverflow;
        AscendC::Reg::Abs(magnitude, alpha, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(alphaOverflow, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Or(combinedOverflow, overflow, alphaOverflow, mask);

        // Keep scale separate when folding FP32 parameters would lose range.
        // The cached scale is one for the ordinary folded path.
        MaskReg betaOverflow;
        MaskReg betaUnderflow;
        MaskReg unsafeBeta;
        MaskReg finiteScale;
        MaskReg nonzeroScale;
        MaskReg finiteReciprocal;
        MaskReg splitCandidate;
        MaskReg splitBeta;
        AscendC::Reg::Abs(magnitude, beta, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(betaOverflow, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Compares<float, CMPMODE::LT>(betaUnderflow, magnitude, 0x1p-126f, mask);
        AscendC::Reg::Or(unsafeBeta, betaOverflow, betaUnderflow, mask);
        AscendC::Reg::Abs(magnitude, scale, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteScale, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(nonzeroScale, magnitude, 0.0f, finiteScale);
        AscendC::Reg::Abs(magnitude, inverseSqrt, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteReciprocal, magnitude, FP32_MAX_FINITE, nonzeroScale);
        AscendC::Reg::Compares<float, CMPMODE::GT>(splitCandidate, magnitude, 0.0f, finiteReciprocal);
        AscendC::Reg::And(splitBeta, splitCandidate, unsafeBeta, mask);
        AscendC::Reg::Select(beta, inverseSqrt, beta, splitBeta);
        AscendC::Reg::Select(scale, scale, oneReg, splitBeta);

        RegTensor<float> centeredShift;
        MaskReg finiteRoot;
        MaskReg recover;
        MaskReg finiteOffset;
        AscendC::Reg::Abs(magnitude, variance, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteRoot, magnitude, FP32_MAX_FINITE, combinedOverflow);
        AscendC::Reg::And(recover, finiteRoot, nonzeroScale, mask);
        AscendC::Reg::Abs(magnitude, offset, mask);
        AscendC::Reg::Compares<float, CMPMODE::LE>(finiteOffset, magnitude, FP32_MAX_FINITE, recover);
        if constexpr (!IsSameType<T_SCALE, float>::value && !IsSameType<T_OFFSET, float>::value) {
            // The affine quotient is evaluated before it meets the wider
            // statistics. Its FP32 overflow remains part of the folding.
            AscendC::Reg::And(finiteOffset, finiteOffset, finiteQuotient, mask);
        }
        AscendC::Reg::Sub(centeredShift, zero, mean, mask);
        AscendC::Reg::Select(alpha, centeredShift, alpha, finiteOffset);
        AscendC::Reg::Select(offset, offset, zero, finiteOffset);
    } else {
        // Low-precision statistics retain FP32 folding, including its IEEE
        // special values; only their extra FP16/BF16 materializations are removed.
        AscendC::Reg::Duplicate(scale, 1.0f, mask);
        AscendC::Reg::Duplicate(offset, 0.0f, mask);
    }
}

template <typename T_X, bool HAS_SCALE>
__aicore__ inline void ApplyCorrectedFoldedAffine(RegTensor<float>& x, RegTensor<float>& alpha, RegTensor<float>& beta,
                                                  RegTensor<float>& scale, RegTensor<float>& correction,
                                                  RegTensor<float>& y, MaskReg& mask)
{
    AscendC::Reg::Add(y, x, alpha, mask);
    AscendC::Reg::Mul(y, y, beta, mask);
    if constexpr (HAS_SCALE) {
        AscendC::Reg::Mul(y, y, scale, mask);
    }
    AscendC::Reg::Add(y, y, correction, mask);
    if constexpr (!HAS_SCALE) {
        RegTensor<float> magnitude;
        RegTensor<float> scaledX;
        MaskReg infiniteX;
        MaskReg infiniteCorrection;
        MaskReg recover;
        AscendC::Reg::Abs(magnitude, x, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(infiniteX, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::Abs(magnitude, correction, mask);
        AscendC::Reg::Compares<float, CMPMODE::GT>(infiniteCorrection, magnitude, FP32_MAX_FINITE, mask);
        AscendC::Reg::And(recover, infiniteX, infiniteCorrection, mask);
        AscendC::Reg::Mul(scaledX, x, beta, mask);
        AscendC::Reg::Select(y, scaledX, y, recover);
    }
}

template <typename T_X>
__aicore__ inline void ApplyFoldedAffine(RegTensor<float>& x, RegTensor<float>& alpha, RegTensor<float>& beta,
                                         RegTensor<float>& y, MaskReg& mask)
{
    AscendC::Reg::Add(y, x, alpha, mask);
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
        // Keep intermediates in FP32 until the final output conversion.
        AscendC::Reg::Mul(y, y, scale, mask);
        if constexpr (HAS_OFFSET) {
            AscendC::Reg::Add(y, y, offset, mask);
        }
    }
}
} // namespace BNInferenceOps

#endif // BN_INFERENCE_COMMON_H
