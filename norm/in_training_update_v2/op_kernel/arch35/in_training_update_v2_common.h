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

template <bool HAS_AFFINE, bool ZERO_EPSILON>
__aicore__ inline void ComputeNormalizedY(RegTensor<float>& dst, RegTensor<float>& xReg, RegTensor<float>& sumReg,
                                          RegTensor<float>& meanReg, RegTensor<float>& scaleReg,
                                          RegTensor<float>& betaReg, RegTensor<float>& restoreReg, float negativeInvR,
                                          float negativeInvRCorrection, MaskReg& validMask)
{
    if constexpr (HAS_AFFINE) {
        // The low part of sum / R is incorporated in beta by ComputeAffine.
        Reg::Sub(dst, xReg, meanReg, validMask);
        Reg::Mul(dst, dst, scaleReg, validMask);
        Reg::Add(dst, dst, betaReg, validMask);

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
        Reg::Mul(dst, dst, restoreReg, validMask);
    } else {
        Reg::Muls(dst, xReg, 1.0f, validMask);
        Reg::Axpy(dst, sumReg, negativeInvR, validMask);
        Reg::Axpy(dst, sumReg, negativeInvRCorrection, validMask);
        if constexpr (ZERO_EPSILON) {
            RegTensor<float> simpleCenteredReg;
            MaskReg zeroStdMask;
            Reg::Sub(simpleCenteredReg, xReg, meanReg, validMask);
            Reg::Compares<float, CMPMODE::EQ>(zeroStdMask, scaleReg, 0.0f, validMask);
            Reg::Select(dst, simpleCenteredReg, dst, zeroStdMask);
        }
        Reg::Div(dst, dst, scaleReg, validMask);
        // With an infinite standard deviation, finite original x and mean
        // normalize to zero even if their FP32 difference overflowed.
        RegTensor<float> infiniteStdResult;
        MaskReg infiniteStdMask;
        MaskReg finiteSumMask;
        Reg::Sub(infiniteStdResult, sumReg, sumReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(finiteSumMask, infiniteStdResult, 0.0f, validMask);
        Reg::Compares<float, CMPMODE::EQ>(infiniteStdMask, scaleReg, static_cast<float>(INFINITY), validMask);
        Reg::And(infiniteStdMask, infiniteStdMask, finiteSumMask, validMask);
        Reg::Muls(infiniteStdResult, xReg, 0.0f, validMask);
        Reg::Select(dst, infiniteStdResult, dst, infiniteStdMask);
    }
}

__aicore__ inline void ComputeBaseStats(__ubuf__ float* sum, __ubuf__ float* squareSum, __ubuf__ float* mean,
                                        __ubuf__ float* unbiasedVar, __ubuf__ float* stdValue,
                                        __ubuf__ float* sumForNormalize, int64_t count, float invR,
                                        float invRCorrection, float rForZeroCheck, float bessel, float epsilon)
{
    // Form the small correction separately: rounding 1 + 1/(R-1) to FP32
    // before multiplying can discard significant variance output bits.
    const float besselExtra = (bessel == 0.0f) ? 0.0f : invR / (1.0f - invR);
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
        Reg::Add(varReg, varReg, varErrorReg, validMask);

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

        Reg::Compares<float, CMPMODE::LT>(negativeMask, varReg, 0.0f, validMask);
        Reg::Select(varReg, zeroReg, varReg, negativeMask);
        Reg::Adds(tempReg, varReg, epsilon, validMask);
        Reg::Compares<float, CMPMODE::GT>(positiveMask, tempReg, 0.0f, validMask);
        Reg::Compares<float, CMPMODE::EQ>(zeroMask, tempReg, 0.0f, validMask);
        Reg::Duplicate(specialReg, 1.0f, validMask);
        Reg::Select(stdReg, tempReg, specialReg, positiveMask);
        Reg::Sqrt<float, &SQRT_MODE>(stdReg, stdReg, validMask);
        Reg::Duplicate(nanReg, static_cast<float>(NAN), validMask);
        Reg::Select(stdReg, stdReg, nanReg, positiveMask);
        Reg::Select(stdReg, tempReg, stdReg, zeroMask);
        Reg::Duplicate(specialReg, bessel, validMask);
        Reg::Compares<float, CMPMODE::EQ>(besselZeroMask, specialReg, 0.0f, validMask);
        Reg::Select(tempReg, zeroReg, varReg, besselZeroMask);
        Reg::Axpy(tempReg, tempReg, besselExtra, validMask);
        Reg::Select(tempReg, zeroReg, tempReg, besselZeroMask);
        // Round the compensated mean once for the public FP32 statistic and
        // the zero-denominator centering path.
        Reg::Muls(meanErrorReg, meanReg, -1.0f, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invR, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invRCorrection, validMask);
        Reg::Add(specialReg, meanReg, meanErrorReg, validMask);
        Reg::Sub(finiteCheckReg, meanReg, meanReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(positiveMask, finiteCheckReg, 0.0f, validMask);
        Reg::Select(meanReg, specialReg, meanReg, positiveMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(mean, meanReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(unbiasedVar, tempReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(stdValue, stdReg, validMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(sumForNormalize, sumReg, validMask);
    }
}

__aicore__ inline void ComputeAffine(__ubuf__ float* gamma, __ubuf__ float* beta, __ubuf__ float* stdValue,
                                     __ubuf__ float* betaValue, __ubuf__ float* mean, __ubuf__ float* restore,
                                     int64_t count, float invR, float invRCorrection, float exactR)
{
    __VEC_SCOPE__
    {
        RegTensor<float> gammaReg;
        RegTensor<float> betaReg;
        RegTensor<float> stdReg;
        RegTensor<float> scaleReg;
        RegTensor<float> sumReg;
        RegTensor<float> meanReg;
        RegTensor<float> meanErrorReg;
        RegTensor<float> correctedBetaReg;
        RegTensor<float> finiteCheckReg;
        RegTensor<float> restoreReg;
        RegTensor<float> scaledGammaReg;
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
        Reg::Div(scaledGammaReg, scaledGammaReg, stdReg, validMask);
        Reg::Select(scaleReg, scaledGammaReg, scaleReg, scaleOverflowMask);
        Reg::Muls(scaledGammaReg, betaReg, 0x1p-96f, validMask);
        Reg::Select(betaReg, scaledGammaReg, betaReg, scaleOverflowMask);
        Reg::Duplicate(restoreReg, 1.0f, validMask);
        Reg::Duplicate(scaledGammaReg, 0x1p48f, validMask);
        Reg::Select(restoreReg, scaledGammaReg, restoreReg, scaleOverflowMask);
        Reg::StoreAlign<float, StoreDist::DIST_NORM>(restore, restoreReg, validMask);
        // betaValue still holds the original sum from PrepareBaseStats.
        // Correct the affine offset before reusing this buffer for beta, so
        // the spatial loop does not require an additional statistic buffer.
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(sumReg, betaValue);
        Reg::LoadAlign<float, LoadDist::DIST_NORM>(meanReg, mean);
        Reg::Muls(meanErrorReg, meanReg, -1.0f, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invR, validMask);
        Reg::Axpy(meanErrorReg, sumReg, invRCorrection, validMask);
        // Exact representable means must not retain the third-order error
        // of the split reciprocal, which an extreme gamma could amplify.
        Reg::Muls(scaledGammaReg, meanReg, exactR, validMask);
        Reg::Muls(finiteCheckReg, scaledGammaReg, -1.0f, validMask);
        Reg::Axpy(finiteCheckReg, meanReg, exactR, validMask);
        Reg::Compare<float, CMPMODE::EQ>(gammaFiniteMask, scaledGammaReg, sumReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(positiveStdMask, finiteCheckReg, 0.0f, validMask);
        Reg::And(gammaFiniteMask, gammaFiniteMask, positiveStdMask, validMask);
        Reg::Duplicate(scaledGammaReg, 0.0f, validMask);
        Reg::Select(meanErrorReg, scaledGammaReg, meanErrorReg, gammaFiniteMask);
        Reg::Muls(correctedBetaReg, meanErrorReg, -1.0f, validMask);
        Reg::MulDstAdd(correctedBetaReg, scaleReg, betaReg, validMask);
        Reg::Sub(finiteCheckReg, scaleReg, scaleReg, validMask);
        Reg::Compares<float, CMPMODE::EQ>(finiteScaleMask, finiteCheckReg, 0.0f, validMask);
        Reg::Select(betaReg, correctedBetaReg, betaReg, finiteScaleMask);
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
        const float rForZeroCheck = (tiling_->r <= MAX_EXACT_FP32_INTEGER) ? static_cast<float>(tiling_->r) :
                                                                             static_cast<float>(NAN);
        ComputeBaseStats((__ubuf__ float*)sumLocal.GetPhyAddr(), (__ubuf__ float*)squareLocal.GetPhyAddr(), MeanAddr(),
                         VarAddr(), ScaleAddr(), BiasAddr(), count, tiling_->invR, tiling_->invRCorrection,
                         rForZeroCheck, tiling_->bessel, tiling_->epsilon);
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
            ComputeAffine(
                (__ubuf__ float*)gammaLocal.GetPhyAddr(), (__ubuf__ float*)betaLocal.GetPhyAddr(), ScaleAddr(),
                BiasAddr(), MeanAddr(), VarAddr(), count, tiling_->invR, tiling_->invRCorrection,
                tiling_->r <= MAX_EXACT_FP32_INTEGER ? static_cast<float>(tiling_->r) : static_cast<float>(NAN));
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
};

} // namespace INTrainingUpdateV2Ops

#endif // IN_TRAINING_UPDATE_V2_COMMON_H
