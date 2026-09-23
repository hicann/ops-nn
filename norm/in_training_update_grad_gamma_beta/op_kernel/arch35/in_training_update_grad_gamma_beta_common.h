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
 * \file in_training_update_grad_gamma_beta_common.h
 * \brief RegBase helpers for stable FP32 axis-zero reduction.
 */

#ifndef IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_COMMON_H_
#define IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_COMMON_H_

#include "kernel_operator.h"

namespace NsINTrainingUpdateGradGammaBeta {
using namespace AscendC;
using namespace AscendC::Reg;

constexpr uint32_t VL_FP32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
constexpr float MAX_FINITE_FP32 = 3.402823466e+38F;
constexpr float FAST_PATH_ERROR_SAFETY_FACTOR = 4.0F;
constexpr float PRECISION_ABSOLUTE_TOLERANCE = 1.0e-8F;
constexpr float PRECISION_RELATIVE_TOLERANCE = 1.0e-4F;
constexpr int64_t FAST_PATH_MAX_REDUCE_COUNT = 1000000;
constexpr CastTrait PART_INDEX_CAST = {RegLayout::UNKNOWN, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                       RoundMode::CAST_RINT};

__simd_callee__ inline void CompareOccupied(MaskReg& occupied, RegTensor<float>& length, uint32_t part, MaskReg& mask)
{
    RegTensor<int32_t> integerIndex;
    RegTensor<float> floatIndex;
    // Runtime scalar integer-to-float conversion is not supported inside VF.
    Duplicate(integerIndex, static_cast<int32_t>(part));
    Cast<float, int32_t, PART_INDEX_CAST>(floatIndex, integerIndex, mask);
    Compare<float, CMPMODE::GT>(occupied, length, floatIndex, mask);
}

__simd_callee__ inline void AddWithResidual(RegTensor<float>& high, RegTensor<float>& low, RegTensor<float>& lhs,
                                            RegTensor<float>& rhs, MaskReg& mask)
{
    RegTensor<float> absLhs;
    RegTensor<float> absRhs;
    RegTensor<float> larger;
    RegTensor<float> smaller;
    RegTensor<float> zero;
    MaskReg lhsDominates;
    MaskReg finite;
    Abs(absLhs, lhs, mask);
    Abs(absRhs, rhs, mask);
    Compare<float, CMPMODE::GE>(lhsDominates, absLhs, absRhs, mask);
    Select(larger, lhs, rhs, lhsDominates);
    Select(smaller, rhs, lhs, lhsDominates);
    Add(high, larger, smaller, mask);
    Sub(low, high, larger, mask);
    Sub(low, smaller, low, mask);
    Abs(absLhs, low, mask);
    Compares<float, CMPMODE::LE>(finite, absLhs, MAX_FINITE_FP32, mask);
    Duplicate(zero, 0.0F, mask);
    Select(low, low, zero, finite);
}

// Two sweeps remove gaps in the expansion. Without this normalization,
// non-overlapping but sparse residuals can exhaust a fixed-size partial array.
__simd_callee__ inline void CompressExpansion(__ubuf__ float* accumulator, RegTensor<float>& length, uint32_t offset,
                                              uint32_t stateStride, uint32_t partialCount, MaskReg& mask)
{
    RegTensor<float> carry;
    RegTensor<float> partial;
    RegTensor<float> high;
    RegTensor<float> low;
    RegTensor<float> zero;
    RegTensor<float> nextLength;
    RegTensor<float> advancedLength;
    RegTensor<int32_t> cursor;
    RegTensor<int32_t> nextCursor;
    RegTensor<int32_t> bottom;
    RegTensor<int32_t> readIndex;
    MaskReg occupied;
    MaskReg nonzero;
    Duplicate(zero, 0.0F, mask);
    Duplicate(carry, 0.0F, mask);
    Arange(cursor, static_cast<int32_t>(offset + (partialCount - 1U) * stateStride));
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    for (uint16_t reverse = static_cast<uint16_t>(partialCount); reverse > 0U; --reverse) {
        const uint32_t part = static_cast<uint32_t>(reverse - 1U);
        LoadAlign(partial, accumulator + part * stateStride + offset);
        CompareOccupied(occupied, length, part, mask);
        Select(partial, partial, zero, occupied);
        AddWithResidual(high, low, carry, partial, mask);
        Compares<float, CMPMODE::NE>(nonzero, low, 0.0F, mask);
        // Pack from the top: destinations are never below the current read.
        Scatter<float, uint32_t>(accumulator, high, reinterpret_cast<RegTensor<uint32_t>&>(cursor), nonzero);
        Adds(nextCursor, cursor, -static_cast<int32_t>(stateStride), mask);
        Select(cursor, nextCursor, cursor, nonzero);
        Select(carry, low, high, nonzero);
    }
    Scatter<float, uint32_t>(accumulator, carry, reinterpret_cast<RegTensor<uint32_t>&>(cursor), mask);
    bottom = cursor;

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    Duplicate(carry, 0.0F, mask);
    Duplicate(nextLength, 0.0F, mask);
    Arange(cursor, static_cast<int32_t>(offset));
    for (uint16_t part = 0U; part < partialCount; ++part) {
        const uint32_t readOffset = static_cast<uint32_t>(part) * stateStride + offset;
        LoadAlign(partial, accumulator + readOffset);
        Arange(readIndex, static_cast<int32_t>(readOffset));
        Compare<int32_t, CMPMODE::GE>(occupied, readIndex, bottom, mask);
        Select(partial, partial, zero, occupied);
        AddWithResidual(high, low, partial, carry, mask);
        Compares<float, CMPMODE::NE>(nonzero, low, 0.0F, mask);
        // Pack from the bottom: destinations are never above the current read.
        Scatter<float, uint32_t>(accumulator, low, reinterpret_cast<RegTensor<uint32_t>&>(cursor), nonzero);
        Adds(nextCursor, cursor, static_cast<int32_t>(stateStride), mask);
        Select(cursor, nextCursor, cursor, nonzero);
        Adds(advancedLength, nextLength, 1.0F, mask);
        Select(nextLength, advancedLength, nextLength, nonzero);
        carry = high;
    }
    Compares<float, CMPMODE::NE>(nonzero, carry, 0.0F, mask);
    Scatter<float, uint32_t>(accumulator, carry, reinterpret_cast<RegTensor<uint32_t>&>(cursor), nonzero);
    Adds(advancedLength, nextLength, 1.0F, mask);
    Select(length, advancedLength, nextLength, nonzero);
}

__simd_vf__ inline void ZeroFloatBuffer(__ubuf__ float* buffer, uint32_t count)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> zero;
    MaskReg mask;
    uint32_t remaining = count;
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        Duplicate(zero, 0.0F, mask);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        StoreAlign(buffer + offset, zero, mask);
    }
}

__simd_vf__ inline void ZeroFastReductionState(__ubuf__ float* accumulator, uint32_t count, uint32_t stateStride)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> zero;
    MaskReg mask;
    uint32_t remaining = count;
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        Duplicate(zero, 0.0F, mask);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        StoreAlign(accumulator + offset, zero, mask);
        StoreAlign(accumulator + stateStride + offset, zero, mask);
        StoreAlign(accumulator + 2U * stateStride + offset, zero, mask);
    }
}

__simd_vf__ inline void ZeroExpansionState(__ubuf__ float* accumulator, __ubuf__ float* lengths, uint32_t count,
                                           uint32_t stateStride, uint32_t partialCount)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> zero;
    MaskReg mask;
    uint32_t remaining = count;
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        Duplicate(zero, 0.0F, mask);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        for (uint16_t part = 0U; part < partialCount; ++part) {
            StoreAlign(accumulator + static_cast<uint32_t>(part) * stateStride + offset, zero, mask);
        }
        StoreAlign(lengths + offset, zero, mask);
    }
}

__simd_vf__ inline void AccumulateDirectRows(const __ubuf__ float* input, __ubuf__ float* accumulator, uint32_t count,
                                             uint32_t rowPitchElements, uint16_t rowCount)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> sum;
    RegTensor<float> value;
    MaskReg mask;
    uint32_t remaining = count;
    __ubuf__ float* readableInput = const_cast<__ubuf__ float*>(input);
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        LoadAlign(sum, accumulator + offset);
        for (uint16_t row = 0U; row < rowCount; ++row) {
            LoadAlign(value, readableInput + static_cast<uint32_t>(row) * rowPitchElements + offset);
            Add(sum, sum, value, mask);
        }
        StoreAlign(accumulator + offset, sum, mask);
    }
}

__simd_vf__ inline void UpdateMagnitudeRows(const __ubuf__ float* input, __ubuf__ float* magnitude, uint32_t count,
                                            uint32_t rowPitchElements, uint16_t rowCount)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> maxAbs;
    RegTensor<float> value;
    RegTensor<float> absValue;
    MaskReg mask;
    uint32_t remaining = count;
    __ubuf__ float* readableInput = const_cast<__ubuf__ float*>(input);
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        LoadAlign(maxAbs, magnitude + offset);
        for (uint16_t row = 0U; row < rowCount; ++row) {
            LoadAlign(value, readableInput + static_cast<uint32_t>(row) * rowPitchElements + offset);
            Abs(absValue, value, mask);
            Max(maxAbs, maxAbs, absValue, mask);
        }
        StoreAlign(magnitude + offset, maxAbs, mask);
    }
}

// The common path keeps Neumaier's primary sum and compensation while also
// collecting the column magnitude in the same GM pass.  A second error-free
// transform observes information that would be discarded while adding each
// correction to the compensation.  The absolute discarded amount is
// accumulated as a conservative retry signal; the numerical result remains
// bit-compatible with the original two-component path for safe finite inputs.
__simd_vf__ inline void AccumulateFastRows(const __ubuf__ float* input, __ubuf__ float* accumulator,
                                           __ubuf__ float* magnitude, uint32_t count, uint32_t rowPitchElements,
                                           uint16_t rowCount, uint32_t stateStride)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> sum;
    RegTensor<float> compensation;
    RegTensor<float> errorBound;
    RegTensor<float> maxAbs;
    RegTensor<float> value;
    RegTensor<float> updated;
    RegTensor<float> absSum;
    RegTensor<float> absValue;
    RegTensor<float> correctionFromSum;
    RegTensor<float> correctionFromValue;
    RegTensor<float> correction;
    RegTensor<float> updatedCompensation;
    RegTensor<float> lostResidual;
    RegTensor<float> absLostResidual;
    RegTensor<float> zero;
    RegTensor<float> absCompensation;
    MaskReg mask;
    MaskReg sumDominatesMask;
    MaskReg finiteCompensationMask;
    uint32_t remaining = count;
    __ubuf__ float* readableInput = const_cast<__ubuf__ float*>(input);
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        Duplicate(zero, 0.0F, mask);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        LoadAlign(sum, accumulator + offset);
        LoadAlign(compensation, accumulator + stateStride + offset);
        LoadAlign(errorBound, accumulator + 2U * stateStride + offset);
        LoadAlign(maxAbs, magnitude + offset);
        for (uint16_t row = 0U; row < rowCount; ++row) {
            LoadAlign(value, readableInput + static_cast<uint32_t>(row) * rowPitchElements + offset);
            Abs(absValue, value, mask);
            Max(maxAbs, maxAbs, absValue, mask);
            Add(updated, sum, value, mask);
            Abs(absSum, sum, mask);
            Compare<float, CMPMODE::GE>(sumDominatesMask, absSum, absValue, mask);
            Sub(correctionFromSum, sum, updated, mask);
            Add(correctionFromSum, correctionFromSum, value, mask);
            Sub(correctionFromValue, value, updated, mask);
            Add(correctionFromValue, correctionFromValue, sum, mask);
            Select(correction, correctionFromSum, correctionFromValue, sumDominatesMask);

            AddWithResidual(updatedCompensation, lostResidual, compensation, correction, mask);
            Abs(absCompensation, updatedCompensation, mask);
            Compares<float, CMPMODE::LE>(finiteCompensationMask, absCompensation, MAX_FINITE_FP32, mask);
            Select(compensation, updatedCompensation, zero, finiteCompensationMask);
            Abs(absLostResidual, lostResidual, mask);
            Add(errorBound, errorBound, absLostResidual, mask);
            sum = updated;
        }
        StoreAlign(accumulator + offset, sum, mask);
        StoreAlign(accumulator + stateStride + offset, compensation, mask);
        StoreAlign(accumulator + 2U * stateStride + offset, errorBound, mask);
        StoreAlign(magnitude + offset, maxAbs, mask);
    }
}

__simd_vf__ inline void FinalizeFastAndMarkRetry(__ubuf__ float* accumulator, const __ubuf__ float* magnitude,
                                                 uint32_t count, uint32_t stateStride, float safeMagnitude)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> sum;
    RegTensor<float> compensation;
    RegTensor<float> errorBound;
    RegTensor<float> maxAbs;
    RegTensor<float> result;
    RegTensor<float> guardedError;
    RegTensor<float> absResult;
    RegTensor<float> tolerance;
    RegTensor<float> retry;
    RegTensor<float> zero;
    RegTensor<float> one;
    MaskReg mask;
    MaskReg safeMask;
    MaskReg retryMask;
    uint32_t remaining = count;
    __ubuf__ float* readableMagnitude = const_cast<__ubuf__ float*>(magnitude);
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        LoadAlign(sum, accumulator + offset);
        LoadAlign(compensation, accumulator + stateStride + offset);
        LoadAlign(errorBound, accumulator + 2U * stateStride + offset);
        LoadAlign(maxAbs, readableMagnitude + offset);
        Compares<float, CMPMODE::LE>(safeMask, maxAbs, safeMagnitude, mask);
        Add(result, sum, compensation, mask);
        Muls(guardedError, errorBound, FAST_PATH_ERROR_SAFETY_FACTOR, mask);
        Abs(absResult, result, mask);
        Muls(tolerance, absResult, PRECISION_RELATIVE_TOLERANCE, mask);
        Adds(tolerance, tolerance, PRECISION_ABSOLUTE_TOLERANCE, mask);
        Compare<float, CMPMODE::GT>(retryMask, guardedError, tolerance, mask);
        Duplicate(zero, 0.0F, mask);
        Duplicate(one, 1.0F, mask);
        Select(retry, zero, one, safeMask);
        Select(retry, one, retry, retryMask);
        StoreAlign(accumulator + offset, result, mask);
        StoreAlign(accumulator + 2U * stateStride + offset, retry, mask);
    }
}

__simd_vf__ inline void AccumulateScaledRows(const __ubuf__ float* input, __ubuf__ float* accumulator,
                                             __ubuf__ float* lengths, const __ubuf__ float* magnitude, uint32_t count,
                                             uint32_t rowPitchElements, uint16_t rowCount, uint32_t stateStride,
                                             uint32_t partialCount, float safeMagnitude, float inputScale)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> maxAbs;
    RegTensor<float> value;
    RegTensor<float> scaledValue;
    RegTensor<float> partial;
    RegTensor<float> length;
    RegTensor<float> nextLength;
    RegTensor<float> advancedLength;
    RegTensor<float> updated;
    RegTensor<float> correction;
    RegTensor<float> zero;
    RegTensor<int32_t> cursor;
    RegTensor<int32_t> advancedCursor;
    MaskReg mask;
    MaskReg safeMask;
    MaskReg occupiedMask;
    MaskReg nonzeroMask;
    uint32_t remaining = count;
    __ubuf__ float* readableInput = const_cast<__ubuf__ float*>(input);
    __ubuf__ float* readableMagnitude = const_cast<__ubuf__ float*>(magnitude);
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        Duplicate(zero, 0.0F, mask);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        LoadAlign(length, lengths + offset);
        LoadAlign(maxAbs, readableMagnitude + offset);
        Compares<float, CMPMODE::LE>(safeMask, maxAbs, safeMagnitude, mask);
        for (uint16_t row = 0U; row < rowCount; ++row) {
            // The previous row compacted its expansion into the same UB slots.
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            LoadAlign(value, readableInput + static_cast<uint32_t>(row) * rowPitchElements + offset);
            Muls(scaledValue, value, inputScale, mask);
            Select(value, value, scaledValue, safeMask);
            Arange(cursor, static_cast<int32_t>(offset));
            Duplicate(nextLength, 0.0F, mask);
            for (uint16_t part = 0U; part < partialCount; ++part) {
                LoadAlign(partial, accumulator + static_cast<uint32_t>(part) * stateStride + offset);
                CompareOccupied(occupiedMask, length, part, mask);
                Select(partial, partial, zero, occupiedMask);

                // FastTwoSum after magnitude ordering: updated + correction
                // retains the exact finite sum of this pair. Unlike a single
                // compensation accumulator, nonzero residuals are not rounded
                // together; compact them from smallest to largest instead.
                AddWithResidual(updated, correction, value, partial, mask);
                Compares<float, CMPMODE::NE>(nonzeroMask, correction, 0.0F, mask);

                // Each lane keeps its column offset, so scatter indexes cannot
                // collide. The output slot is at most part: no unread partial
                // is overwritten. A non-finite updated value propagates while
                // its meaningless residual is discarded.
                Scatter<float, uint32_t>(accumulator, correction, reinterpret_cast<RegTensor<uint32_t>&>(cursor),
                                         nonzeroMask);
                Adds(advancedCursor, cursor, static_cast<int32_t>(stateStride), mask);
                Select(cursor, advancedCursor, cursor, nonzeroMask);
                Adds(advancedLength, nextLength, 1.0F, mask);
                Select(nextLength, advancedLength, nextLength, nonzeroMask);
                value = updated;
            }
            Compares<float, CMPMODE::NE>(nonzeroMask, value, 0.0F, mask);
            Scatter<float, uint32_t>(accumulator, value, reinterpret_cast<RegTensor<uint32_t>&>(cursor), nonzeroMask);
            Adds(advancedLength, nextLength, 1.0F, mask);
            Select(length, advancedLength, nextLength, nonzeroMask);
            CompressExpansion(accumulator, length, offset, stateStride, partialCount, mask);
        }
        StoreAlign(lengths + offset, length, mask);
    }
}

__simd_vf__ inline void FinalizeReduction(__ubuf__ float* accumulator, const __ubuf__ float* lengths,
                                          const __ubuf__ float* magnitude, uint32_t count, uint32_t stateStride,
                                          uint32_t partialCount, float safeMagnitude, float outputScale)
{
    const uint16_t loopCount = static_cast<uint16_t>((count + VL_FP32 - 1U) / VL_FP32);
    RegTensor<float> partial;
    RegTensor<float> length;
    RegTensor<float> zero;
    RegTensor<float> maxAbs;
    RegTensor<float> result;
    RegTensor<float> restored;
    MaskReg mask;
    MaskReg safeMask;
    MaskReg occupiedMask;
    uint32_t remaining = count;
    __ubuf__ float* readableLengths = const_cast<__ubuf__ float*>(lengths);
    __ubuf__ float* readableMagnitude = const_cast<__ubuf__ float*>(magnitude);
    for (uint16_t index = 0U; index < loopCount; ++index) {
        mask = UpdateMask<float>(remaining);
        const uint32_t offset = static_cast<uint32_t>(index) * VL_FP32;
        LoadAlign(length, readableLengths + offset);
        Duplicate(zero, 0.0F, mask);
        Duplicate(result, 0.0F, mask);
        for (uint16_t part = 0U; part < partialCount; ++part) {
            LoadAlign(partial, accumulator + static_cast<uint32_t>(part) * stateStride + offset);
            CompareOccupied(occupiedMask, length, part, mask);
            Select(partial, partial, zero, occupiedMask);
            Add(result, result, partial, mask);
        }
        LoadAlign(maxAbs, readableMagnitude + offset);
        Compares<float, CMPMODE::LE>(safeMask, maxAbs, safeMagnitude, mask);
        Muls(restored, result, outputScale, mask);
        Select(result, result, restored, safeMask);
        StoreAlign(accumulator + offset, result, mask);
    }
}
} // namespace NsINTrainingUpdateGradGammaBeta

#endif // IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_COMMON_H_
