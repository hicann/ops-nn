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
 * \file in_training_update_v2_common.h
 * \brief Shared RegBase primitives and statistic staging for INTrainingUpdateV2.
 */

#ifndef IN_TRAINING_UPDATE_V2_COMMON_H
#define IN_TRAINING_UPDATE_V2_COMMON_H

#include <cmath>
#include <type_traits>
#include "kernel_operator.h"
#include "in_training_update_v2_tiling_data.h"

namespace INTrainingUpdateV2Ops {
using namespace AscendC;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr uint32_t VL_FP32 = 64;
constexpr int64_t STAT_CHUNK = 64;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr int64_t CHANNEL_TILE = 64;
constexpr int64_t DMA_BLOCK_BYTES = 32;
constexpr int64_t MAX_EXACT_FP32_INTEGER = 1LL << 24;
static constexpr AscendC::Reg::SqrtSpecificMode SQRT_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    false,
    AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE,
};

constexpr AscendC::Reg::CastTrait CAST_B16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait CAST_FP32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void LoadXToFp32(__ubuf__ T* src, RegTensor<float>& dst, MaskReg& mask, uint32_t offset)
{
    if constexpr (std::is_same<T, float>::value) {
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T> packed;
        Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(packed, src + offset);
        Reg::Cast<float, T, CAST_B16_TO_FP32>(dst, packed, mask);
    }
}

template <typename T>
__aicore__ inline void StoreFp32ToY(__ubuf__ T* dst, RegTensor<float>& src, MaskReg& mask, uint32_t offset)
{
    if constexpr (std::is_same<T, float>::value) {
        Reg::StoreAlign<T, StoreDist::DIST_NORM>(dst + offset, src, mask);
    } else {
        RegTensor<T> packed;
        Reg::Cast<T, float, CAST_FP32_TO_B16>(packed, src, mask);
        Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(dst + offset, packed, mask);
    }
}

// Error of a+b relative to an already rounded result. Callers keep a and b
// distinct from error so the low part survives cancellation.
__aicore__ inline void SumError(RegTensor<float>& error, RegTensor<float>& a, RegTensor<float>& b,
                                RegTensor<float>& rounded, MaskReg& mask)
{
    RegTensor<float> virtualB;
    RegTensor<float> temp;
    Reg::Sub(virtualB, rounded, a, mask);
    Reg::Sub(temp, rounded, virtualB, mask);
    Reg::Sub(error, a, temp, mask);
    Reg::Sub(virtualB, b, virtualB, mask);
    Reg::Add(error, error, virtualB, mask);
}

template <typename T>
__aicore__ inline void CorrectRoundedY(RegTensor<float>& value, RegTensor<float>& error, MaskReg& mask)
{
    RegTensor<float> corrected;
    RegTensor<float> tail;
    RegTensor<float> check;
    MaskReg finiteValue;
    MaskReg finiteError;
    Reg::Sub(check, value, value, mask);
    Reg::Compares<float, CMPMODE::EQ>(finiteValue, check, 0.0f, mask);
    Reg::Sub(check, error, error, mask);
    Reg::Compares<float, CMPMODE::EQ>(finiteError, check, 0.0f, mask);
    Reg::And(finiteValue, finiteValue, finiteError, mask);
    Reg::Add(corrected, value, error, mask);
    if constexpr (!std::is_same<T, float>::value) {
        // FP64->FP32->FP16 can round a value below 65520 to the FP32 tie
        // 65520 and then to Inf. Preserve the low-part direction at that tie.
        Reg::Sub(tail, value, corrected, mask);
        Reg::Add(tail, tail, error, mask);
        MaskReg boundary;
        MaskReg inward;
        Reg::Compares<float, CMPMODE::EQ>(boundary, corrected, 65520.0f, mask);
        Reg::Compares<float, CMPMODE::LT>(inward, tail, 0.0f, mask);
        Reg::And(boundary, boundary, inward, mask);
        Reg::Adds(check, corrected, -0x1p-8f, mask);
        Reg::Select(corrected, check, corrected, boundary);
        Reg::Compares<float, CMPMODE::EQ>(boundary, corrected, -65520.0f, mask);
        Reg::Compares<float, CMPMODE::GT>(inward, tail, 0.0f, mask);
        Reg::And(boundary, boundary, inward, mask);
        Reg::Adds(check, corrected, 0x1p-8f, mask);
        Reg::Select(corrected, check, corrected, boundary);
    }
    Reg::Select(value, corrected, value, finiteValue);
}

template <typename T, bool HAS_AFFINE, bool ZERO_EPSILON>
__aicore__ inline void ComputeNormalizedY(RegTensor<float>& dst, RegTensor<float>& xReg, RegTensor<float>& sumReg,
                                          RegTensor<float>& meanReg, RegTensor<float>& scaleReg,
                                          RegTensor<float>& betaReg, RegTensor<float>& restoreReg,
                                          RegTensor<float>& scaleErrorReg, RegTensor<float>& biasErrorReg,
                                          MaskReg& validMask)
{
    if constexpr (HAS_AFFINE) {
        // ComputeAffine selects the centering point and incorporates its
        // low part in beta. Fuse the product and offset to avoid rounding
        // the product before cancellation with beta.
        Reg::Sub(dst, xReg, meanReg, validMask);
        RegTensor<float> centered;
        RegTensor<float> centerError;
        RegTensor<float> product;
        RegTensor<float> productError;
        RegTensor<float> outputError;
        RegTensor<float> negativeMean;
        Reg::Muls(centered, dst, 1.0f, validMask);
        Reg::Muls(negativeMean, meanReg, -1.0f, validMask);
        SumError(centerError, xReg, negativeMean, centered, validMask);
        Reg::Mul(product, centered, scaleReg, validMask);
        Reg::Muls(productError, product, -1.0f, validMask);
        Reg::Muls(outputError, centered, 1.0f, validMask);
        Reg::MulDstAdd(outputError, scaleReg, productError, validMask);
        Reg::Muls(productError, outputError, 1.0f, validMask);
        Reg::MulDstAdd(dst, scaleReg, betaReg, validMask);
        SumError(outputError, product, betaReg, dst, validMask);
        Reg::Add(outputError, outputError, productError, validMask);
        Reg::MulDstAdd(centerError, scaleReg, outputError, validMask);
        Reg::MulDstAdd(centered, scaleErrorReg, centerError, validMask);
        Reg::Add(outputError, centered, biasErrorReg, validMask);

        // Reassociation is not valid for an infinite or NaN scale. Retain
        // x * scale + (beta - mean * scale) in these lanes.
        RegTensor<float> finiteCheckReg;
        RegTensor<float> directReg;
        RegTensor<float> biasReg;
        MaskReg finiteScaleMask;
        MaskReg nonzeroScaleMask;
        Reg::Sub(finiteCheckReg, scaleReg, scaleReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(finiteScaleMask, finiteCheckReg, 0.0f, validMask);
        Reg::Compares<float, CMPMODE::NE>(nonzeroScaleMask, scaleReg, 0.0f, validMask);
        Reg::And(finiteScaleMask, finiteScaleMask, nonzeroScaleMask, validMask);
        Reg::Mul(biasReg, meanReg, scaleReg, validMask);
        Reg::Sub(biasReg, betaReg, biasReg, validMask);
        Reg::Mul(directReg, xReg, scaleReg, validMask);
        Reg::Add(directReg, directReg, biasReg, validMask);
        Reg::Select(dst, dst, directReg, finiteScaleMask);
        // Extreme finite scales are stored with a power-of-two exponent.
        Reg::Mul(dst, dst, restoreReg, validMask);
        Reg::Mul(outputError, outputError, restoreReg, validMask);
        if constexpr (std::is_same<T, float>::value) {
            // Combine the low part before the final exponent restoration:
            // the rounded high part alone may overflow while the sum is finite.
            CorrectRoundedY<T>(dst, outputError, validMask);
            Reg::Mul(dst, dst, restoreReg, validMask);
        } else {
            Reg::Mul(dst, dst, restoreReg, validMask);
            Reg::Mul(outputError, outputError, restoreReg, validMask);
            CorrectRoundedY<T>(dst, outputError, validMask);
        }
    } else {
        RegTensor<float> simpleCenteredReg;
        MaskReg finiteSumMask;
        Reg::Sub(simpleCenteredReg, sumReg, sumReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(finiteSumMask, simpleCenteredReg, 0.0f, validMask);
        RegTensor<float> centered;
        RegTensor<float> centerError;
        RegTensor<float> negativeMean;
        RegTensor<float> outputError;
        Reg::Muls(negativeMean, meanReg, -1.0f, validMask);
        Reg::Sub(centered, xReg, meanReg, validMask);
        SumError(centerError, xReg, negativeMean, centered, validMask);
        // Without affine, biasError stores the low part of the mean. It
        // remains significant when the rounded mean is exactly equal to x.
        Reg::Sub(centerError, centerError, biasErrorReg, validMask);
        if constexpr (ZERO_EPSILON) {
            MaskReg zeroStdMask;
            MaskReg zeroCenteredMask;
            Reg::Compares<float, CMPMODE::EQ>(zeroStdMask, scaleReg, 0.0f, validMask);
            Reg::Compares<float, CMPMODE::EQ>(zeroCenteredMask, centered, 0.0f, validMask);
            Reg::And(zeroStdMask, zeroStdMask, zeroCenteredMask, validMask);
            Reg::Select(centered, centerError, centered, zeroStdMask);
        }
        Reg::Div(dst, centered, scaleReg, validMask);
        Reg::Muls(outputError, dst, -1.0f, validMask);
        Reg::MulDstAdd(outputError, scaleReg, centered, validMask);
        Reg::Add(outputError, outputError, centerError, validMask);
        Reg::Muls(centered, dst, -1.0f, validMask);
        Reg::MulDstAdd(centered, scaleErrorReg, outputError, validMask);
        Reg::Div(outputError, centered, scaleReg, validMask);
        // The direct fallback remains authoritative for non-finite sums.
        Reg::Sub(simpleCenteredReg, xReg, sumReg, validMask);
        Reg::Div(simpleCenteredReg, simpleCenteredReg, scaleReg, validMask);
        Reg::Select(dst, dst, simpleCenteredReg, finiteSumMask);
        CorrectRoundedY<T>(dst, outputError, validMask);
        // With an infinite standard deviation, finite original x and mean
        // normalize to zero even if their FP32 difference overflowed.
        RegTensor<float> infiniteStdResult;
        MaskReg infiniteStdMask;
        Reg::Compares<float, CMPMODE::EQ>(infiniteStdMask, scaleReg, static_cast<float>(INFINITY), validMask);
        Reg::And(infiniteStdMask, infiniteStdMask, finiteSumMask, validMask);
        Reg::Muls(infiniteStdResult, xReg, 0.0f, validMask);
        Reg::Select(dst, infiniteStdResult, dst, infiniteStdMask);
    }
}

__aicore__ inline void ComputeBaseStats(__ubuf__ float* sum, __ubuf__ float* squareSum, __ubuf__ float* mean,
                                        __ubuf__ float* unbiasedVar, __ubuf__ float* stdValue,
                                        __ubuf__ float* sumForNormalize, __ubuf__ float* stdError,
                                        __ubuf__ float* meanLow, int64_t count, float invR, float invRCorrection,
                                        float rForZeroCheck, float besselExtra, float epsilon)
{
    __VEC_SCOPE__
    {
        RegTensor<float> sumReg;
        RegTensor<float> squareReg;
        RegTensor<float> meanReg;
        RegTensor<float> meanErrorReg;
        RegTensor<float> varReg;
        RegTensor<float> varErrorReg;
        RegTensor<float> meanSquareReg;
        RegTensor<float> meanSquareErrorReg;
        RegTensor<float> simpleVarReg;
        RegTensor<float> tempReg;
        RegTensor<float> finiteCheckReg;
        RegTensor<float> zeroReg;
        RegTensor<float> stdReg;
        RegTensor<float> specialReg;
        RegTensor<float> nanReg;
        RegTensor<float> varLowReg;
        RegTensor<float> stdErrorReg;
        RegTensor<float> varHighReg;
        MaskReg negativeMask;
        MaskReg positiveMask;
        MaskReg zeroMask;
        MaskReg besselZeroMask;
        MaskReg scaledSquareFiniteMask;
        MaskReg meanSquareFiniteMask;
        uint32_t validCount = static_cast<uint32_t>(count);
        MaskReg validMask = UpdateMask<float>(validCount);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(sumReg, sum);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(squareReg, squareSum);

        // Evaluate square_sum / R - (sum / R)^2 as a two-float expansion.
        // The direct FP32 subtraction can lose every useful low bit when the
        // variance is small relative to the mean. On DAV_3510 these Axpy and
        // MulDstAdd sequences map to multiply-add instructions; target-binary
        // and hardware regressions verify the required residual recovery.
        Reg::Muls(meanReg, sumReg, invR, validMask);
        Reg::Muls(meanErrorReg, meanReg, -1.0f, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invR, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invRCorrection, validMask);

        Reg::Muls(varReg, squareReg, invR, validMask);
        Reg::Muls(varErrorReg, varReg, -1.0f, validMask);
        Reg::Axpy(varErrorReg, squareReg, invR, validMask);
        Reg::Axpy(varErrorReg, squareReg, invRCorrection, validMask);

        Reg::Mul(meanSquareReg, meanReg, meanReg, validMask);
        Reg::Sub(simpleVarReg, varReg, meanSquareReg, validMask);

        Reg::Muls(tempReg, meanSquareReg, -1.0f, validMask);
        Reg::Muls(meanSquareErrorReg, meanReg, 1.0f, validMask);
        Reg::MulDstAdd(meanSquareErrorReg, meanReg, tempReg, validMask);
        Reg::Muls(tempReg, meanReg, 2.0f, validMask);
        Reg::MulDstAdd(tempReg, meanErrorReg, meanSquareErrorReg, validMask);
        Reg::Muls(meanSquareErrorReg, meanErrorReg, 1.0f, validMask);
        Reg::MulDstAdd(meanSquareErrorReg, meanErrorReg, tempReg, validMask);

        Reg::Sub(varReg, varReg, meanSquareReg, validMask);
        Reg::Sub(varErrorReg, varErrorReg, meanSquareErrorReg, validMask);
        Reg::Muls(varHighReg, varReg, 1.0f, validMask);
        Reg::Add(varReg, varReg, varErrorReg, validMask);
        SumError(varLowReg, varHighReg, varErrorReg, varReg, validMask);

        // Residual recovery intentionally forms Inf-Inf for non-finite inputs.
        // Fall back to the public direct expression in those lanes so the
        // existing IEEE special-value categories remain unchanged.
        Reg::Sub(finiteCheckReg, squareReg, squareReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(scaledSquareFiniteMask, finiteCheckReg, 0.0f, validMask);
        Reg::Select(varReg, varReg, simpleVarReg, scaledSquareFiniteMask);
        Reg::Sub(finiteCheckReg, meanSquareReg, meanSquareReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(meanSquareFiniteMask, finiteCheckReg, 0.0f, validMask);
        Reg::Select(varReg, varReg, simpleVarReg, meanSquareFiniteMask);
        // A finite FP32 sum has a finite square in the promoted expression.
        // If square_sum is non-finite, an overflow of mean*mean in FP32 must
        // not turn (+Inf - finite) into (Inf - Inf).
        Reg::Sub(finiteCheckReg, sumReg, sumReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(meanSquareFiniteMask, finiteCheckReg, 0.0f, validMask);
        Reg::Select(tempReg, squareReg, varReg, meanSquareFiniteMask);
        Reg::Select(varReg, varReg, tempReg, scaledSquareFiniteMask);

        Reg::Duplicate(zeroReg, 0.0f, validMask);
        Reg::Select(varLowReg, varLowReg, zeroReg, scaledSquareFiniteMask);
        Reg::Select(varLowReg, varLowReg, zeroReg, meanSquareFiniteMask);

        // A two-float reciprocal still has a third-order residual. It can
        // manufacture a tiny positive variance for an exactly constant plane
        // (for example sum=square_sum=R=3). Use two exact FMA products to
        // recognize square_sum * R == sum * sum. In those lanes retain the
        // public FP32 expression, including its rounding sign, rather than
        // replacing a genuine small variance whose products are not equal.
        Reg::Duplicate(specialReg, rForZeroCheck, validMask);
        Reg::Mul(meanSquareErrorReg, squareReg, specialReg, validMask);
        Reg::Muls(meanErrorReg, squareReg, 1.0f, validMask);
        Reg::Muls(tempReg, meanSquareErrorReg, -1.0f, validMask);
        Reg::MulDstAdd(meanErrorReg, specialReg, tempReg, validMask);
        Reg::Mul(finiteCheckReg, sumReg, sumReg, validMask);
        Reg::Muls(varErrorReg, sumReg, 1.0f, validMask);
        Reg::Muls(tempReg, finiteCheckReg, -1.0f, validMask);
        Reg::MulDstAdd(varErrorReg, sumReg, tempReg, validMask);
        Reg::Compare<float, CMPMODE::EQ>(scaledSquareFiniteMask, meanSquareErrorReg, finiteCheckReg, validMask);
        Reg::Compare<float, CMPMODE::EQ>(meanSquareFiniteMask, meanErrorReg, varErrorReg, validMask);
        Reg::Compares<float, CMPMODE::GT>(negativeMask, meanSquareErrorReg, 0.0f, validMask);
        Reg::Sub(tempReg, meanSquareErrorReg, meanSquareErrorReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(positiveMask, tempReg, 0.0f, validMask);
        Reg::And(zeroMask, scaledSquareFiniteMask, meanSquareFiniteMask, validMask);
        Reg::And(zeroMask, zeroMask, positiveMask, validMask);
        Reg::And(zeroMask, zeroMask, negativeMask, validMask);
        Reg::Select(varReg, simpleVarReg, varReg, zeroMask);
        Reg::Select(varLowReg, zeroReg, varLowReg, zeroMask);

        Reg::Compares<float, CMPMODE::LT>(negativeMask, varReg, 0.0f, validMask);
        Reg::Select(varReg, zeroReg, varReg, negativeMask);
        Reg::Select(varLowReg, zeroReg, varLowReg, negativeMask);
        Reg::Adds(tempReg, varReg, epsilon, validMask);
        Reg::Duplicate(specialReg, epsilon, validMask);
        SumError(stdErrorReg, varReg, specialReg, tempReg, validMask);
        Reg::Add(stdErrorReg, stdErrorReg, varLowReg, validMask);
        Reg::Compares<float, CMPMODE::GT>(positiveMask, tempReg, 0.0f, validMask);
        Reg::Compares<float, CMPMODE::EQ>(zeroMask, tempReg, 0.0f, validMask);
        Reg::Duplicate(specialReg, 1.0f, validMask);
        Reg::Select(stdReg, tempReg, specialReg, positiveMask);
        Reg::Sqrt<float, &SQRT_MODE>(stdReg, stdReg, validMask);
        Reg::Duplicate(nanReg, static_cast<float>(NAN), validMask);
        Reg::Select(stdReg, stdReg, nanReg, positiveMask);
        Reg::Select(stdReg, tempReg, stdReg, zeroMask);
        Reg::Muls(varHighReg, stdReg, -1.0f, validMask);
        Reg::MulDstAdd(varHighReg, stdReg, tempReg, validMask);
        Reg::Add(varHighReg, varHighReg, stdErrorReg, validMask);
        Reg::Muls(specialReg, stdReg, 2.0f, validMask);
        Reg::Div(stdErrorReg, varHighReg, specialReg, validMask);
        Reg::Sub(finiteCheckReg, stdErrorReg, stdErrorReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(positiveMask, finiteCheckReg, 0.0f, validMask);
        Reg::Select(stdErrorReg, stdErrorReg, zeroReg, positiveMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(stdError, stdErrorReg, validMask);
        Reg::Duplicate(specialReg, besselExtra, validMask);
        Reg::Compares<float, CMPMODE::EQ>(besselZeroMask, specialReg, 0.0f, validMask);
        // Preserve both parts until the final rounding of the unbiased variance.
        Reg::Muls(tempReg, varLowReg, 1.0f, validMask);
        Reg::Axpy(tempReg, varLowReg, besselExtra, validMask);
        Reg::Axpy(tempReg, varReg, besselExtra, validMask);
        Reg::Add(tempReg, varReg, tempReg, validMask);
        Reg::Select(tempReg, zeroReg, tempReg, besselZeroMask);
        // Round the compensated mean once for the public FP32 statistic and
        // the zero-denominator centering path.
        Reg::Muls(meanErrorReg, meanReg, -1.0f, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invR, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invRCorrection, validMask);
        Reg::Add(specialReg, meanReg, meanErrorReg, validMask);
        Reg::Sub(meanSquareReg, meanReg, specialReg, validMask);
        Reg::Add(meanErrorReg, meanSquareReg, meanErrorReg, validMask);
        Reg::Sub(finiteCheckReg, meanReg, meanReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(positiveMask, finiteCheckReg, 0.0f, validMask);
        Reg::Select(meanReg, specialReg, meanReg, positiveMask);
        Reg::Select(meanErrorReg, meanErrorReg, zeroReg, positiveMask);
        // Clear the split reciprocal's third-order residual when the mean
        // is exactly representable. Otherwise gamma or a zero denominator
        // could amplify that artificial residual into a different category.
        Reg::Duplicate(specialReg, rForZeroCheck, validMask);
        Reg::Mul(meanSquareReg, meanReg, specialReg, validMask);
        Reg::Muls(finiteCheckReg, meanSquareReg, -1.0f, validMask);
        Reg::Muls(meanSquareErrorReg, meanReg, 1.0f, validMask);
        Reg::MulDstAdd(meanSquareErrorReg, specialReg, finiteCheckReg, validMask);
        Reg::Compare<float, CMPMODE::EQ>(scaledSquareFiniteMask, meanSquareReg, sumReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(meanSquareFiniteMask, meanSquareErrorReg, 0.0f, validMask);
        Reg::And(scaledSquareFiniteMask, scaledSquareFiniteMask, meanSquareFiniteMask, validMask);
        Reg::Select(meanErrorReg, zeroReg, meanErrorReg, scaledSquareFiniteMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(meanLow, meanErrorReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(mean, meanReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(unbiasedVar, tempReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(stdValue, stdReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(sumForNormalize, sumReg, validMask);
    }
}

template <typename T>
__aicore__ inline void ComputeAffine(__ubuf__ float* gamma, __ubuf__ float* beta, __ubuf__ float* stdValue,
                                     __ubuf__ float* betaValue, __ubuf__ float* mean, __ubuf__ float* restore,
                                     __ubuf__ float* scaleError, __ubuf__ float* biasError, int64_t count,
                                     float rForMeanCorrection, float invR)
{
    __VEC_SCOPE__
    {
        RegTensor<float> gammaReg;
        RegTensor<float> betaReg;
        RegTensor<float> stdReg;
        RegTensor<float> scaleReg;
        RegTensor<float> meanReg;
        RegTensor<float> meanErrorReg;
        RegTensor<float> correctedBetaReg;
        RegTensor<float> finiteCheckReg;
        RegTensor<float> restoreReg;
        RegTensor<float> scaledGammaReg;
        RegTensor<float> gammaWorkReg;
        RegTensor<float> betaOriginalReg;
        RegTensor<float> meanOriginalReg;
        RegTensor<float> meanLowReg;
        RegTensor<float> meanTailReg;
        RegTensor<float> scaleErrorReg;
        RegTensor<float> biasErrorReg;
        RegTensor<float> stdErrorReg;
        MaskReg gammaFiniteMask;
        MaskReg scaleOverflowMask;
        MaskReg positiveStdMask;
        MaskReg finiteScaleMask;
        uint32_t validCount = static_cast<uint32_t>(count);
        MaskReg validMask = UpdateMask<float>(validCount);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(gammaReg, gamma);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(betaReg, beta);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(stdReg, stdValue);
        Reg::Div(scaleReg, gammaReg, stdReg, validMask);
        Reg::Sub(finiteCheckReg, scaleReg, scaleReg, validMask);
        Reg::Compares<float, CMPMODE::NE>(scaleOverflowMask, finiteCheckReg, 0.0f, validMask);
        Reg::Sub(finiteCheckReg, gammaReg, gammaReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(gammaFiniteMask, finiteCheckReg, 0.0f, validMask);
        Reg::Compares<float, CMPMODE::GT>(positiveStdMask, stdReg, 0.0f, validMask);
        Reg::And(scaleOverflowMask, scaleOverflowMask, gammaFiniteMask, validMask);
        Reg::And(scaleOverflowMask, scaleOverflowMask, positiveStdMask, validMask);
        Reg::Muls(scaledGammaReg, gammaReg, 0x1p-96f, validMask);
        Reg::Select(gammaWorkReg, scaledGammaReg, gammaReg, scaleOverflowMask);
        Reg::Div(scaledGammaReg, scaledGammaReg, stdReg, validMask);
        Reg::Select(scaleReg, scaledGammaReg, scaleReg, scaleOverflowMask);
        Reg::Muls(scaledGammaReg, betaReg, 0x1p-96f, validMask);
        Reg::Select(betaReg, scaledGammaReg, betaReg, scaleOverflowMask);
        Reg::Muls(betaOriginalReg, betaReg, 1.0f, validMask);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(stdErrorReg, scaleError);
        Reg::Muls(scaleErrorReg, scaleReg, -1.0f, validMask);
        Reg::MulDstAdd(scaleErrorReg, stdReg, gammaWorkReg, validMask);
        Reg::Muls(gammaWorkReg, scaleReg, -1.0f, validMask);
        Reg::MulDstAdd(gammaWorkReg, stdErrorReg, scaleErrorReg, validMask);
        Reg::Div(scaleErrorReg, gammaWorkReg, stdReg, validMask);
        Reg::Duplicate(restoreReg, 1.0f, validMask);
        Reg::Duplicate(scaledGammaReg, 0x1p48f, validMask);
        Reg::Select(restoreReg, scaledGammaReg, restoreReg, scaleOverflowMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(restore, restoreReg, validMask);
        // Base statistics stage the mean low part in biasError. Correct the
        // affine offset before reusing that buffer for the affine low part.
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(meanReg, mean);
        Reg::Muls(meanOriginalReg, meanReg, 1.0f, validMask);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(meanErrorReg, biasError);
        Reg::Muls(meanLowReg, meanErrorReg, 1.0f, validMask);
        // A rounded mean low part can still matter after near-mean
        // cancellation with an extreme gamma. Recover its division residual
        // from the original sum before betaValue is reused for the offset.
        if constexpr (std::is_same<T, float>::value) {
            Reg::LoadAlign<float, LoadDist::DIST_NORM>(meanTailReg, betaValue);
            Reg::Duplicate(gammaWorkReg, rForMeanCorrection, validMask);
            Reg::Muls(finiteCheckReg, meanReg, -1.0f, validMask);
            Reg::MulDstAdd(finiteCheckReg, gammaWorkReg, meanTailReg, validMask);
            Reg::Muls(meanTailReg, meanLowReg, -1.0f, validMask);
            Reg::MulDstAdd(meanTailReg, gammaWorkReg, finiteCheckReg, validMask);
            Reg::Muls(meanTailReg, meanTailReg, invR, validMask);
            Reg::Sub(finiteCheckReg, meanTailReg, meanTailReg, validMask);
            Reg::Compares<float, CMPMODE::EQ>(positiveStdMask, finiteCheckReg, 0.0f, validMask);
            Reg::Duplicate(gammaWorkReg, 0.0f, validMask);
            Reg::Select(meanTailReg, meanTailReg, gammaWorkReg, positiveStdMask);
        }
        Reg::Muls(correctedBetaReg, meanErrorReg, -1.0f, validMask);
        Reg::MulDstAdd(correctedBetaReg, scaleReg, betaReg, validMask);
        Reg::Sub(finiteCheckReg, scaleReg, scaleReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(finiteScaleMask, finiteCheckReg, 0.0f, validMask);
        // Either center at mean or at zero. Prefer the representation with
        // the smaller affine offset, reducing cancellation without discarding
        // mean centering when the original beta is small. Statistics have
        // already been emitted (or are recomputed by ProcessOwnedStats).
        Reg::Muls(scaledGammaReg, meanReg, -1.0f, validMask);
        Reg::MulDstAdd(scaledGammaReg, scaleReg, betaReg, validMask);
        Reg::Muls(finiteCheckReg, meanErrorReg, -1.0f, validMask);
        Reg::MulDstAdd(finiteCheckReg, scaleReg, scaledGammaReg, validMask);
        Reg::Abs(scaledGammaReg, finiteCheckReg, validMask);
        Reg::Abs(meanErrorReg, correctedBetaReg, validMask);
        Reg::Compare<float, CMPMODE::LT>(positiveStdMask, scaledGammaReg, meanErrorReg, validMask);
        Reg::And(positiveStdMask, positiveStdMask, finiteScaleMask, validMask);
        Reg::Select(correctedBetaReg, finiteCheckReg, correctedBetaReg, positiveStdMask);
        Reg::Duplicate(scaledGammaReg, 0.0f, validMask);
        Reg::Select(meanReg, scaledGammaReg, meanReg, positiveStdMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(mean, meanReg, validMask);
        Reg::Select(betaReg, correctedBetaReg, betaReg, finiteScaleMask);
        // Recover the affine offset low part for the chosen centering point.
        Reg::Sub(meanOriginalReg, meanOriginalReg, meanReg, validMask);
        Reg::Mul(gammaWorkReg, scaleReg, meanOriginalReg, validMask);
        Reg::Muls(stdErrorReg, gammaWorkReg, -1.0f, validMask);
        Reg::Muls(biasErrorReg, scaleReg, 1.0f, validMask);
        Reg::MulDstAdd(biasErrorReg, meanOriginalReg, stdErrorReg, validMask);
        Reg::Muls(gammaWorkReg, gammaWorkReg, -1.0f, validMask);
        Reg::Add(correctedBetaReg, betaOriginalReg, gammaWorkReg, validMask);
        SumError(stdErrorReg, betaOriginalReg, gammaWorkReg, correctedBetaReg, validMask);
        Reg::Sub(biasErrorReg, stdErrorReg, biasErrorReg, validMask);
        Reg::Muls(gammaWorkReg, scaleReg, -1.0f, validMask);
        Reg::MulDstAdd(gammaWorkReg, meanLowReg, biasErrorReg, validMask);
        Reg::Muls(biasErrorReg, meanOriginalReg, -1.0f, validMask);
        Reg::MulDstAdd(biasErrorReg, scaleErrorReg, gammaWorkReg, validMask);
        Reg::Muls(gammaWorkReg, meanLowReg, -1.0f, validMask);
        Reg::MulDstAdd(gammaWorkReg, scaleErrorReg, biasErrorReg, validMask);
        Reg::Sub(correctedBetaReg, correctedBetaReg, betaReg, validMask);
        Reg::Add(biasErrorReg, correctedBetaReg, gammaWorkReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(scaleError, scaleErrorReg, validMask);
        if constexpr (std::is_same<T, float>::value) {
            Reg::Muls(meanTailReg, meanTailReg, -1.0f, validMask);
            Reg::MulDstAdd(meanTailReg, scaleReg, biasErrorReg, validMask);
            Reg::StoreAlign<float, StoreDist::DIST_NORM>(biasError, meanTailReg, validMask);
        } else {
            Reg::StoreAlign<float, StoreDist::DIST_NORM>(biasError, biasErrorReg, validMask);
        }
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(stdValue, scaleReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(betaValue, betaReg, validMask);
    }
}

template <bool HAS_RUNNING>
__aicore__ inline void ComputeOutputStats(__ubuf__ float* currentMean, __ubuf__ float* currentVar,
                                          __ubuf__ float* runningMean, __ubuf__ float* runningVar,
                                          __ubuf__ float* outputMean, __ubuf__ float* outputVar, int64_t count,
                                          float momentum, float oneMinusMomentum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> meanReg;
        RegTensor<float> varReg;
        RegTensor<float> outputMeanReg;
        RegTensor<float> outputVarReg;
        uint32_t validCount = static_cast<uint32_t>(count);
        MaskReg validMask = UpdateMask<float>(validCount);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(meanReg, currentMean);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(varReg, currentVar);
        if constexpr (HAS_RUNNING) {
            RegTensor<float> runningMeanReg;
            RegTensor<float> runningVarReg;
            RegTensor<float> tempReg;
            Reg::LoadAlign<float, LoadDist::DIST_NORM>(runningMeanReg, runningMean);
            Reg::LoadAlign<float, LoadDist::DIST_NORM>(runningVarReg, runningVar);
            Reg::Muls(outputMeanReg, meanReg, momentum, validMask);
            Reg::Muls(tempReg, runningMeanReg, oneMinusMomentum, validMask);
            Reg::Add(outputMeanReg, outputMeanReg, tempReg, validMask);
            Reg::Muls(outputVarReg, varReg, momentum, validMask);
            Reg::Muls(tempReg, runningVarReg, oneMinusMomentum, validMask);
            Reg::Add(outputVarReg, outputVarReg, tempReg, validMask);
        } else {
            Reg::Muls(outputMeanReg, meanReg, 1.0f, validMask);
            Reg::Muls(outputVarReg, varReg, 1.0f, validMask);
        }
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(outputMean, outputMeanReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(outputVar, outputVarReg, validMask);
    }
}

template <typename T, bool HAS_AFFINE, bool HAS_RUNNING>
class INTrainingUpdateV2Base {
protected:
    __aicore__ inline void InitCommon(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR gamma, GM_ADDR beta,
                                      GM_ADDR mean, GM_ADDR variance, GM_ADDR y, GM_ADDR batchMean,
                                      GM_ADDR batchVariance, const INTrainingUpdateV2TilingData* tilingData,
                                      TPipe* pipe)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        const int64_t statElements = tiling_->n * tiling_->c;
        xGm_.SetGlobalBuffer((__gm__ T*)x, tiling_->totalElements);
        yGm_.SetGlobalBuffer((__gm__ T*)y, tiling_->totalElements);
        sumGm_.SetGlobalBuffer((__gm__ float*)sum, statElements);
        squareSumGm_.SetGlobalBuffer((__gm__ float*)squareSum, statElements);
        batchMeanGm_.SetGlobalBuffer((__gm__ float*)batchMean, statElements);
        batchVarianceGm_.SetGlobalBuffer((__gm__ float*)batchVariance, statElements);
        if constexpr (HAS_AFFINE) {
            const int64_t gammaElements = (tiling_->gammaBatchStride == 0) ? tiling_->c : statElements;
            const int64_t betaElements = (tiling_->betaBatchStride == 0) ? tiling_->c : statElements;
            gammaGm_.SetGlobalBuffer((__gm__ float*)gamma, gammaElements);
            betaGm_.SetGlobalBuffer((__gm__ float*)beta, betaElements);
        }
        if constexpr (HAS_RUNNING) {
            runningMeanGm_.SetGlobalBuffer((__gm__ float*)mean, statElements);
            runningVarianceGm_.SetGlobalBuffer((__gm__ float*)variance, statElements);
        }

        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, tiling_->xyBufferBytes);
        pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, tiling_->xyBufferBytes);
        pipe_->InitBuffer(statAQue_, 1, tiling_->statBufferBytes);
        pipe_->InitBuffer(statBQue_, 1, tiling_->statBufferBytes);
        pipe_->InitBuffer(batchMeanQue_, 1, tiling_->statBufferBytes);
        pipe_->InitBuffer(batchVarQue_, 1, tiling_->statBufferBytes);
        pipe_->InitBuffer(meanBuf_, tiling_->statBufferBytes);
        pipe_->InitBuffer(unbiasedVarBuf_, tiling_->statBufferBytes);
        pipe_->InitBuffer(scaleBuf_, tiling_->statBufferBytes);
        pipe_->InitBuffer(biasBuf_, tiling_->statBufferBytes);
        pipe_->InitBuffer(scaleErrorBuf_, tiling_->statBufferBytes);
        pipe_->InitBuffer(biasErrorBuf_, tiling_->statBufferBytes);
    }

    __aicore__ inline LocalTensor<float> StageStat(TQue<QuePosition::VECIN, 1>& queue, const GlobalTensor<float>& gm,
                                                   int64_t offset, int64_t count)
    {
        LocalTensor<float> local = queue.AllocTensor<float>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[offset], copy, pad);
        queue.EnQue(local);
        return queue.DeQue<float>();
    }

    __aicore__ inline void PrepareBaseStats(int64_t statOffset, int64_t count)
    {
        LocalTensor<float> sumLocal = StageStat(statAQue_, sumGm_, statOffset, count);
        LocalTensor<float> squareLocal = StageStat(statBQue_, squareSumGm_, statOffset, count);
        // Bias staging holds the original sum for no-affine centering. It is
        // overwritten by beta when the complete affine pair is present.
        // Large R is also exact in FP32 when removing trailing zero bits
        // leaves at most 24 significant bits. Keep exact-mean/variance
        // detection for those R values without accepting a rounded R.
        int64_t rSignificand = tiling_->r;
        while (rSignificand > MAX_EXACT_FP32_INTEGER && (rSignificand & 1) == 0) {
            rSignificand >>= 1;
        }
        const float rForZeroCheck = (rSignificand <= MAX_EXACT_FP32_INTEGER) ? static_cast<float>(tiling_->r) :
                                                                               static_cast<float>(NAN);
        // Derive 1/(R-1) directly; invR/(1-invR) compounds the rounded 1/R.
        const float besselExtra = tiling_->r > 1 ? 1.0f / static_cast<float>(tiling_->r - 1) : 0.0f;
        ComputeBaseStats((__ubuf__ float*)sumLocal.GetPhyAddr(), (__ubuf__ float*)squareLocal.GetPhyAddr(), MeanAddr(),
                         VarAddr(), ScaleAddr(), BiasAddr(), ScaleErrorAddr(), BiasErrorAddr(), count, tiling_->invR,
                         tiling_->invRCorrection, rForZeroCheck, besselExtra, tiling_->epsilon);
        statAQue_.FreeTensor(sumLocal);
        statBQue_.FreeTensor(squareLocal);
    }

    __aicore__ inline void PrepareStats(int64_t statOffset, int64_t gammaOffset, int64_t betaOffset, int64_t count,
                                        bool writeStats = false)
    {
        PrepareBaseStats(statOffset, count);
        if (writeStats) {
            WriteStats(statOffset, count);
        }
        if constexpr (HAS_AFFINE) {
            LocalTensor<float> gammaLocal = StageStat(statAQue_, gammaGm_, gammaOffset, count);
            LocalTensor<float> betaLocal = StageStat(statBQue_, betaGm_, betaOffset, count);
            const float rForMeanCorrection = tiling_->r <= MAX_EXACT_FP32_INTEGER ? static_cast<float>(tiling_->r) :
                                                                                    static_cast<float>(NAN);
            ComputeAffine<T>((__ubuf__ float*)gammaLocal.GetPhyAddr(), (__ubuf__ float*)betaLocal.GetPhyAddr(),
                             ScaleAddr(), BiasAddr(), MeanAddr(), VarAddr(), ScaleErrorAddr(), BiasErrorAddr(), count,
                             rForMeanCorrection, tiling_->invR);
            statAQue_.FreeTensor(gammaLocal);
            statBQue_.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void WriteStats(int64_t statOffset, int64_t count)
    {
        LocalTensor<float> runningMeanLocal;
        LocalTensor<float> runningVarLocal;
        __ubuf__ float* runningMeanAddr = nullptr;
        __ubuf__ float* runningVarAddr = nullptr;
        if constexpr (HAS_RUNNING) {
            runningMeanLocal = StageStat(statAQue_, runningMeanGm_, statOffset, count);
            runningVarLocal = StageStat(statBQue_, runningVarianceGm_, statOffset, count);
            runningMeanAddr = (__ubuf__ float*)runningMeanLocal.GetPhyAddr();
            runningVarAddr = (__ubuf__ float*)runningVarLocal.GetPhyAddr();
        }

        LocalTensor<float> meanOutput = batchMeanQue_.AllocTensor<float>();
        LocalTensor<float> varOutput = batchVarQue_.AllocTensor<float>();
        ComputeOutputStats<HAS_RUNNING>(
            MeanAddr(), VarAddr(), runningMeanAddr, runningVarAddr, (__ubuf__ float*)meanOutput.GetPhyAddr(),
            (__ubuf__ float*)varOutput.GetPhyAddr(), count, tiling_->momentum, tiling_->oneMinusMomentum);
        batchMeanQue_.EnQue(meanOutput);
        batchVarQue_.EnQue(varOutput);
        meanOutput = batchMeanQue_.DeQue<float>();
        varOutput = batchVarQue_.DeQue<float>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(batchMeanGm_[statOffset], meanOutput, copy);
        DataCopyPad(batchVarianceGm_[statOffset], varOutput, copy);
        batchMeanQue_.FreeTensor(meanOutput);
        batchVarQue_.FreeTensor(varOutput);
        if constexpr (HAS_RUNNING) {
            statAQue_.FreeTensor(runningMeanLocal);
            statBQue_.FreeTensor(runningVarLocal);
        }
    }

    // DataCopyPad writes a partial GM block through a block-granular read/modify/write.
    // Complete 32-byte ownership blocks prevent different AI Vector cores from
    // updating neighbouring logical elements in the same physical GM block.
    __aicore__ inline void GetOwnedAlignedRange(int64_t totalElements, int64_t alignmentElements, int64_t& ownedStart,
                                                int64_t& ownedCount) const
    {
        const int64_t blockIndex = static_cast<int64_t>(GetBlockIdx());
        const int64_t usedCores = tiling_->unitBlocks * tiling_->rCores;
        const int64_t totalBlocks = totalElements / alignmentElements +
                                    ((totalElements % alignmentElements) != 0 ? 1 : 0);
        if (blockIndex >= usedCores || blockIndex >= totalBlocks) {
            ownedStart = 0;
            ownedCount = 0;
            return;
        }
        const int64_t baseBlocks = totalBlocks / usedCores;
        const int64_t extraBlocks = totalBlocks % usedCores;
        const int64_t ownedBlocks = baseBlocks + ((blockIndex < extraBlocks) ? 1 : 0);
        const int64_t startBlock = blockIndex * baseBlocks + ((blockIndex < extraBlocks) ? blockIndex : extraBlocks);
        const int64_t endBlock = startBlock + ownedBlocks;
        ownedStart = startBlock * alignmentElements;
        const int64_t ownedEnd = (endBlock >= totalBlocks) ? totalElements : endBlock * alignmentElements;
        ownedCount = ownedEnd - ownedStart;
    }

    __aicore__ inline void GetOwnedYRange(int64_t alignmentElements, int64_t& ownedStart, int64_t& ownedCount) const
    {
        const int64_t blockIndex = static_cast<int64_t>(GetBlockIdx());
        const int64_t unitBlock = blockIndex / tiling_->rCores;
        if (unitBlock >= tiling_->unitBlocks) {
            ownedStart = 0;
            ownedCount = 0;
            return;
        }
        int64_t startBlock = 0;
        int64_t ownedBlocks = 0;
        if (unitBlock < tiling_->formerBlockNum) {
            startBlock = unitBlock * tiling_->formerUnits;
            ownedBlocks = tiling_->formerUnits;
        } else {
            startBlock = tiling_->formerBlockNum * tiling_->formerUnits +
                         (unitBlock - tiling_->formerBlockNum) * tiling_->latterUnits;
            ownedBlocks = tiling_->latterUnits;
        }
        ownedStart = startBlock * alignmentElements;
        const int64_t endBlock = startBlock + ownedBlocks;
        const int64_t totalBlocks = tiling_->totalElements / alignmentElements +
                                    ((tiling_->totalElements % alignmentElements) != 0 ? 1 : 0);
        const int64_t ownedEnd = (endBlock >= totalBlocks) ? tiling_->totalElements : endBlock * alignmentElements;
        ownedCount = ownedEnd - ownedStart;
    }

    __aicore__ inline void ProcessOwnedStats()
    {
        constexpr int64_t fp32ElementsPerGmBlock = DMA_BLOCK_BYTES / static_cast<int64_t>(sizeof(float));
        const int64_t statElements = tiling_->n * tiling_->c;
        int64_t statStart = 0;
        int64_t statCount = 0;
        GetOwnedAlignedRange(statElements, fp32ElementsPerGmBlock, statStart, statCount);
        const int64_t statEnd = statStart + statCount;
        for (int64_t position = statStart; position < statEnd;) {
            const int64_t cStart = position % tiling_->c;
            int64_t count = statEnd - position;
            if (count > STAT_CHUNK) {
                count = STAT_CHUNK;
            }
            if (count > tiling_->c - cStart) {
                count = tiling_->c - cStart;
            }
            PrepareBaseStats(position, count);
            WriteStats(position, count);
            position += count;
        }
    }

    __aicore__ inline __ubuf__ float* MeanAddr() { return (__ubuf__ float*)meanBuf_.Get<float>().GetPhyAddr(); }

    __aicore__ inline __ubuf__ float* VarAddr() { return (__ubuf__ float*)unbiasedVarBuf_.Get<float>().GetPhyAddr(); }

    __aicore__ inline __ubuf__ float* ScaleAddr() { return (__ubuf__ float*)scaleBuf_.Get<float>().GetPhyAddr(); }

    __aicore__ inline __ubuf__ float* BiasAddr() { return (__ubuf__ float*)biasBuf_.Get<float>().GetPhyAddr(); }

    __aicore__ inline __ubuf__ float* ScaleErrorAddr()
    {
        return (__ubuf__ float*)scaleErrorBuf_.Get<float>().GetPhyAddr();
    }

    __aicore__ inline __ubuf__ float* BiasErrorAddr()
    {
        return (__ubuf__ float*)biasErrorBuf_.Get<float>().GetPhyAddr();
    }

    TPipe* pipe_ = nullptr;
    const INTrainingUpdateV2TilingData* tiling_ = nullptr;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> betaGm_;
    GlobalTensor<float> runningMeanGm_;
    GlobalTensor<float> runningVarianceGm_;
    GlobalTensor<float> batchMeanGm_;
    GlobalTensor<float> batchVarianceGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue_;
    TQue<QuePosition::VECIN, 1> statAQue_;
    TQue<QuePosition::VECIN, 1> statBQue_;
    TQue<QuePosition::VECOUT, 1> batchMeanQue_;
    TQue<QuePosition::VECOUT, 1> batchVarQue_;
    TBuf<> meanBuf_;
    TBuf<> unbiasedVarBuf_;
    TBuf<> scaleBuf_;
    TBuf<> biasBuf_;
    TBuf<> scaleErrorBuf_;
    TBuf<> biasErrorBuf_;
};

} // namespace INTrainingUpdateV2Ops

#endif // IN_TRAINING_UPDATE_V2_COMMON_H
