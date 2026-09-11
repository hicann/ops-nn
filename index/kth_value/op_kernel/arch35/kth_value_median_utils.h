/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file kth_value_median_utils.h
 * @brief Median-mode helper primitives shared by kth_value schedule kernels
 *
 * @details This file provides the common primitives required by median semantics
 *          (NaN-aware median), reused by all schedule kernels:
 *          1. Sort-value canonicalization: replaces every NaN with the canonical quiet NaN
 *             bit pattern so that NaNs sort identically into a deterministic suffix. The
 *             scalar CanonicalizeMedianSortValue additionally normalizes both signed zeros
 *             to +0 so stable sorting preserves their original order.
 *          2. Non-NaN counting: vector reduction of the non-NaN element count of a UB
 *             segment into a scalar slot, with cross-chunk accumulation support
 *             (AccumulateNonNanCount / CountNonNan)
 *          3. Median K resolution: maps the median to a sorted-array index according to
 *             medianMode (PROPAGATE_NAN/IGNORE_NAN/STATIC); the non-NaN count is either
 *             passed in by the caller or derived by binary search over sorted rows
 *             (ResolveMedianK / ResolveMedianKFromSorted / IsNanValue)
 *
 *          All primitives operate on contiguous UB segments only and are axis/row agnostic;
 *          row iteration, multi-core splitting and tiling orchestration are owned by the
 *          callers (radix/merge/small_axis/non_last schedule files).
 */

#ifndef KTH_VALUE_MEDIAN_UTILS_H
#define KTH_VALUE_MEDIAN_UTILS_H

#include <cmath>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "common/util_type_simd.h"

namespace KthValue {
using namespace AscendC;

constexpr Reg::CastTrait MEDIAN_CAST_TO_FLOAT = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                 Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr Reg::CastTrait MEDIAN_CAST_FLOAT_TO_INT32 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                       Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

template <typename T>
constexpr bool IS_MEDIAN_FLOAT_TYPE = IsSameType<T, float>::value || IsSameType<T, half>::value ||
                                      IsSameType<T, bfloat16_t>::value;

// Replaces every NaN in-place with the canonical quiet NaN bit pattern so that all
// NaNs compare/sort identically and form a deterministic suffix after sorting.
template <typename T>
__aicore__ inline void CanonicalizeNanValues(LocalTensor<T> input, uint32_t count)
{
    static_assert(IS_MEDIAN_FLOAT_TYPE<T>, "CanonicalizeNanValues only supports floating-point types");
    if constexpr (IsSameType<float, T>::value) {
        constexpr uint32_t VECTOR_LEN = Ops::Base::GetVRegSize() / sizeof(uint32_t);
        constexpr uint32_t ABS_MASK = 0x7FFFFFFFU;
        constexpr uint32_t INF_BITS = 0x7F800000U;
        constexpr uint32_t CANONICAL_NAN_BITS = 0x7FC00000U;
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, VECTOR_LEN));
        uint32_t remaining = count;
        __ubuf__ uint32_t* inputPtr = reinterpret_cast<__ubuf__ uint32_t*>(input.GetPhyAddr());
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint32_t> rawBits, absBits, absMask, canonicalNan, result;
            Reg::MaskReg validMask, nanMask;
            Reg::Duplicate(absMask, ABS_MASK);
            Reg::Duplicate(canonicalNan, CANONICAL_NAN_BITS);
            for (uint16_t i = 0; i < repeatTime; ++i) {
                validMask = Reg::UpdateMask<uint32_t>(remaining);
                Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(rawBits, inputPtr + i * VECTOR_LEN);
                Reg::And(absBits, rawBits, absMask, validMask);
                Reg::Compares<uint32_t, CMPMODE::GT>(nanMask, absBits, INF_BITS, validMask);
                Reg::Select(result, canonicalNan, rawBits, nanMask);
                Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM_B32>(inputPtr + i * VECTOR_LEN, result, validMask);
            }
        }
    } else {
        constexpr uint32_t VECTOR_LEN = Ops::Base::GetVRegSize() / sizeof(uint16_t);
        constexpr uint16_t ABS_MASK = 0x7FFFU;
        constexpr uint16_t INF_BITS = IsSameType<half, T>::value ? 0x7C00U : 0x7F80U;
        constexpr uint16_t CANONICAL_NAN_BITS = IsSameType<half, T>::value ? 0x7E00U : 0x7FC0U;
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, VECTOR_LEN));
        uint32_t remaining = count;
        __ubuf__ uint16_t* inputPtr = reinterpret_cast<__ubuf__ uint16_t*>(input.GetPhyAddr());
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint16_t> rawBits, absBits, absMask, canonicalNan, result;
            Reg::MaskReg validMask, nanMask;
            Reg::Duplicate(absMask, ABS_MASK);
            Reg::Duplicate(canonicalNan, CANONICAL_NAN_BITS);
            for (uint16_t i = 0; i < repeatTime; ++i) {
                validMask = Reg::UpdateMask<uint16_t>(remaining);
                Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(rawBits, inputPtr + i * VECTOR_LEN);
                Reg::And(absBits, rawBits, absMask, validMask);
                Reg::Compares<uint16_t, CMPMODE::GT>(nanMask, absBits, INF_BITS, validMask);
                Reg::Select(result, canonicalNan, rawBits, nanMask);
                Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_NORM_B16>(inputPtr + i * VECTOR_LEN, result, validMask);
            }
        }
    }
}

__aicore__ inline uint32_t ResolveMedianK(uint32_t staticK, uint32_t validCount, uint32_t nonNanCount,
                                          uint32_t medianMode)
{
    if (medianMode == MEDIAN_MODE_PROPAGATE_NAN && nonNanCount < validCount) {
        return nonNanCount;
    }
    if (medianMode == MEDIAN_MODE_IGNORE_NAN) {
        return nonNanCount == 0U ? 0U : (nonNanCount - 1U) / 2U;
    }
    return staticK;
}

template <typename T>
__simt_callee__ __aicore__ inline bool IsNanValue(T value)
{
    float valueFp32 = static_cast<float>(value);
    return valueFp32 != valueFp32;
}

template <typename T>
__simt_callee__ __aicore__ inline T CanonicalizeMedianSortValue(T value)
{
    float valueFp32 = static_cast<float>(value);
    if (valueFp32 != valueFp32) {
        return static_cast<T>(NAN);
    }
    if (valueFp32 == 0.0f) {
        return static_cast<T>(0.0f);
    }
    return value;
}

// Resolves the sorted-array index of the median for the given mode: derives the non-NaN
// count by lower-bound binary search (NaNs form a suffix) and maps it per medianMode.
template <typename T>
__simt_callee__ __aicore__ inline uint32_t ResolveMedianKFromSorted(const __ubuf__ T* sortedValues, uint32_t rowOffset,
                                                                    uint32_t validCount, uint32_t staticK,
                                                                    uint32_t medianMode)
{
    if (medianMode == MEDIAN_MODE_STATIC) {
        return staticK;
    }

    // NaNs are canonicalized before sorting and form a suffix. Find the first
    // NaN (which is also the number of non-NaN values) with a lower-bound search.
    uint32_t left = 0U;
    uint32_t right = validCount;
    while (left < right) {
        uint32_t mid = left + (right - left) / 2U;
        if (IsNanValue(sortedValues[rowOffset + mid])) {
            right = mid;
        } else {
            left = mid + 1U;
        }
    }
    uint32_t nonNanCount = left;
    if (medianMode == MEDIAN_MODE_PROPAGATE_NAN && nonNanCount < validCount) {
        return nonNanCount;
    }
    if (medianMode == MEDIAN_MODE_IGNORE_NAN) {
        return nonNanCount == 0U ? 0U : (nonNanCount - 1U) / 2U;
    }
    return staticK;
}

// Vector-reduces the number of non-NaN elements in input[count] into a single UB scalar slot;
// with Reset=false the result is accumulated on top of the slot's existing value.
template <bool Reset, typename T>
__aicore__ inline void AccumulateNonNanCount(LocalTensor<T> input, uint32_t count, LocalTensor<uint32_t> scalar)
{
    constexpr uint32_t FLOAT_VL = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t repeatTime = static_cast<uint16_t>((count + FLOAT_VL - 1U) / FLOAT_VL);
    uint32_t remaining = count;
    __ubuf__ T* inputPtr = reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr());
    __ubuf__ int32_t* scalarPtr = reinterpret_cast<__ubuf__ int32_t*>(scalar.GetPhyAddr());
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> value, one, zero, nonNanFlag, reduced, total;
        Reg::RegTensor<int32_t> totalInt;
        Reg::MaskReg validMask, nanMask;
        Reg::MaskReg oneMask = Reg::CreateMask<float, Reg::MaskPattern::VL1>();
        Reg::Duplicate(one, 1.0f);
        Reg::Duplicate(zero, 0.0f);
        Reg::Duplicate(total, 0.0f, oneMask);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            validMask = Reg::UpdateMask<float>(remaining);
            if constexpr (IsSameType<float, T>::value) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(
                    value, reinterpret_cast<__ubuf__ float*>(inputPtr) + i * FLOAT_VL);
            } else {
                Reg::RegTensor<T> valueB16;
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(valueB16, inputPtr + i * FLOAT_VL);
                Reg::Cast<float, T, MEDIAN_CAST_TO_FLOAT>(value, valueB16, validMask);
            }
            Reg::Compare<float, CMPMODE::NE>(nanMask, value, value, validMask);
            Reg::Select(nonNanFlag, zero, one, nanMask);
            Reg::Reduce<Reg::ReduceType::SUM>(reduced, nonNanFlag, validMask);
            Reg::Add(total, total, reduced, oneMask);
        }
        Reg::Cast<int32_t, float, MEDIAN_CAST_FLOAT_TO_INT32>(totalInt, total, oneMask);
        if constexpr (!Reset) {
            Reg::RegTensor<int32_t> accumulated;
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>(accumulated, scalarPtr);
            Reg::Add(totalInt, totalInt, accumulated, oneMask);
        }
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM_B32>(scalarPtr, totalInt, oneMask);
    }
}

template <typename T>
__aicore__ inline uint32_t CountNonNan(LocalTensor<T> input, uint32_t count, LocalTensor<float> scalar, TPipe* pipe)
{
    LocalTensor<uint32_t> scalarU32 = scalar.template ReinterpretCast<uint32_t>();
    AccumulateNonNanCount<true>(input, count, scalarU32);
    event_t countEvent = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(countEvent);
    WaitFlag<HardEvent::V_S>(countEvent);
    return scalarU32.GetValue(0);
}

} // namespace KthValue

#endif
