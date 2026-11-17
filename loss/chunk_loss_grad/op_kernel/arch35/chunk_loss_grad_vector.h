/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"

namespace ChunkLossGradArch35 {
namespace Reg = AscendC::MicroAPI;
constexpr uint32_t FLOAT_LANES = 64;
constexpr Reg::CastTrait LOW_CAST = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                     AscendC::RoundMode::CAST_RINT};
constexpr Reg::ExpSpecificMode PRECISE_EXP = {Reg::MaskMergeMode::ZEROING, AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
constexpr Reg::DivSpecificMode PRECISE_DIV = {Reg::MaskMergeMode::ZEROING, false,
                                              AscendC::DivAlgo::PRECISION_1ULP_FTZ_FALSE};

// GM/UB ownership belongs to the outer queues. This VF only touches UB/registers.
template <class InputType, bool precise>
__simd_vf__ inline void MakeGradientVF(__ubuf__ InputType* destination, __ubuf__ float* source, uint32_t count,
                                       uint32_t pitch, float maximum, float sum, float alpha, int32_t target,
                                       bool ignored)
{
    Reg::RegTensor<float> value, corrected, denominator;
    Reg::RegTensor<int32_t> indices;
    Reg::RegTensor<InputType> low;
    Reg::MaskReg targetMask;
    uint32_t remaining = count;
    uint32_t paddingRemaining = pitch;
    const uint16_t repeats = (pitch + FLOAT_LANES - 1) / FLOAT_LANES;
    const float inverseSum = 1.0f / sum;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeats; ++i) {
        auto valid = Reg::UpdateMask<float>(remaining);
        auto write = Reg::UpdateMask<float>(paddingRemaining);
        Reg::LoadAlign(value, source + i * FLOAT_LANES);
        Reg::Adds(value, value, -maximum, valid);
        if constexpr (precise) {
            // The software Exp path rereads its source after writing dst.
            Reg::Exp<float, &PRECISE_EXP>(corrected, value, valid);
            Reg::Move(value, corrected);
            Reg::Duplicate(denominator, sum, full);
            Reg::Div<float, &PRECISE_DIV>(corrected, value, denominator, valid);
            Reg::Move(value, corrected);
        } else {
            Reg::Exp(value, value, valid);
            Reg::Muls(value, value, inverseSum, valid);
        }
        Reg::Arange<int32_t>(indices, int32_t(i * FLOAT_LANES));
        Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(targetMask, indices, target, valid);
        Reg::Adds(corrected, value, -1.0f, valid);
        Reg::Select(value, corrected, value, targetMask);
        Reg::Muls(value, value, alpha, valid);
        if (ignored) {
            Reg::Duplicate(value, 0.0f, full);
        }
        // ZEROING casts clear the padded lanes; pitch is a multiple of 16 elements.
        Reg::Cast<InputType, float, LOW_CAST>(low, value, full);
        Reg::StoreAlign<InputType, Reg::StoreDist::DIST_PACK_B32>(destination + i * FLOAT_LANES, low, write);
    }
}

template <class InputType, bool addOld, bool final, bool applyAlpha = false>
__simd_vf__ inline void AccumulateVF(__ubuf__ float* partial, __ubuf__ float* previousPartial,
                                     __ubuf__ InputType* destination, uint32_t count, float upstreamScale,
                                     float finalAlpha)
{
    Reg::RegTensor<float> value, previous;
    Reg::RegTensor<InputType> low;
    const uint16_t repeats = (count + FLOAT_LANES - 1) / FLOAT_LANES;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeats; ++i) {
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(value, partial + i * FLOAT_LANES);
        if constexpr (addOld) {
            Reg::LoadAlign(previous, previousPartial + i * FLOAT_LANES);
            Reg::Add(value, value, previous, mask);
        }
        if constexpr (final) {
            Reg::Muls(value, value, upstreamScale, mask);
            if constexpr (applyAlpha) {
                Reg::Muls(value, value, finalAlpha, mask);
            }
            Reg::Cast<InputType, float, LOW_CAST>(low, value, mask);
            Reg::StoreAlign<InputType, Reg::StoreDist::DIST_PACK_B32>(destination + i * FLOAT_LANES, low, mask);
        } else {
            Reg::StoreAlign(partial + i * FLOAT_LANES, value, mask);
        }
    }
}

// Batch the old N-by-V layout without changing either Cube operation. Caches
// are loaded once by the caller; each row stays inside this one VF invocation.
template <class InputType, bool precise>
__simd_vf__ inline void MakeGradientRowsVF(__ubuf__ InputType* destination, __ubuf__ float* source,
                                           __ubuf__ float* maxima, __ubuf__ float* sums, __ubuf__ int64_t* labels,
                                           uint32_t rows, uint32_t count, uint32_t inPitch, uint32_t outPitch,
                                           float alpha, int32_t vocabStart, int64_t ignoreIndex)
{
    Reg::RegTensor<float> value, corrected, maximum, denominator, inverse, zero;
    Reg::RegTensor<int32_t> indices, target, highWord;
    Reg::RegTensor<InputType> low;
    Reg::MaskReg targetMask, ignoredMask, highMask;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zero, 0.0f, full);
    for (uint16_t row = 0; row < rows; ++row) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(maximum, maxima + row);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(denominator, sums + row);
        // Ascend950 is little endian. Legal labels are nonnegative INT32 values,
        // so broadcasting the low word of the INT64 cache is an exact narrowing.
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(target,
                                                             reinterpret_cast<__ubuf__ int32_t*>(labels) + 2 * row);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(highWord,
                                                             reinterpret_cast<__ubuf__ int32_t*>(labels) + 2 * row + 1);
        Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(ignoredMask, target,
                                                     static_cast<int32_t>(static_cast<uint64_t>(ignoreIndex)), full);
        Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(
            highMask, highWord, static_cast<int32_t>(static_cast<uint64_t>(ignoreIndex) >> 32), full);
        Reg::MaskAnd(ignoredMask, ignoredMask, highMask, full);
        Reg::Adds(target, target, -vocabStart, full);
        uint32_t remaining = count;
        uint32_t padding = outPitch;
        for (uint16_t i = 0; i < (outPitch + FLOAT_LANES - 1) / FLOAT_LANES; ++i) {
            auto valid = Reg::UpdateMask<float>(remaining);
            auto write = Reg::UpdateMask<float>(padding);
            Reg::LoadAlign(value, source + row * inPitch + i * FLOAT_LANES);
            Reg::Sub(value, value, maximum, valid);
            if constexpr (precise) {
                Reg::Exp<float, &PRECISE_EXP>(corrected, value, valid);
                Reg::Move(value, corrected);
                Reg::Div<float, &PRECISE_DIV>(corrected, value, denominator, valid);
                Reg::Move(value, corrected);
            } else {
                Reg::Exp(value, value, valid);
                Reg::Duplicate(inverse, 1.0f, full);
                Reg::Div(inverse, inverse, denominator, valid);
                Reg::Mul(value, value, inverse, valid);
            }
            Reg::Arange<int32_t>(indices, int32_t(i * FLOAT_LANES));
            Reg::Compare<int32_t, AscendC::CMPMODE::EQ>(targetMask, indices, target, valid);
            Reg::Adds(corrected, value, -1.0f, valid);
            Reg::Select(value, corrected, value, targetMask);
            Reg::Muls(value, value, alpha, valid);
            Reg::Select(value, zero, value, ignoredMask);
            Reg::Cast<InputType, float, LOW_CAST>(low, value, valid);
            Reg::StoreAlign<InputType, Reg::StoreDist::DIST_PACK_B32>(destination + row * outPitch + i * FLOAT_LANES,
                                                                      low, write);
        }
    }
}
} // namespace ChunkLossGradArch35

namespace ChunkLossGradOpt {
namespace Reg = AscendC::MicroAPI;
using ChunkLossGradArch35::LOW_CAST;
using ChunkLossGradArch35::PRECISE_DIV;
using ChunkLossGradArch35::PRECISE_EXP;
constexpr Reg::CastTrait LABEL_CAST = {Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING,
                                       AscendC::RoundMode::UNKNOWN};

__simd_vf__ inline void NarrowLabels(__ubuf__ int32_t* destination, __ubuf__ int64_t* source, uint32_t count,
                                     int64_t ignoreIndex)
{
    Reg::RegTensor<int64_t> wide, ignoredLabel;
    Reg::MaskReg ignoredMask;
    Reg::RegTensor<int32_t> narrow;
    uint32_t remaining64 = count;
    for (uint16_t i = 0; i < (count + 31) / 32; ++i) {
        auto valid64 = Reg::UpdateMask<int64_t>(remaining64);
        Reg::LoadAlign(wide, source + i * 32);
        Reg::Compares<int64_t, AscendC::CMPMODE::EQ>(ignoredMask, wide, ignoreIndex, valid64);
        Reg::Duplicate(ignoredLabel, int64_t(-1), valid64);
        Reg::Select(wide, ignoredLabel, wide, ignoredMask);
        Reg::Cast<int32_t, int64_t, LABEL_CAST>(narrow, wide, valid64);
        Reg::Pack((Reg::RegTensor<uint32_t>&)narrow, (Reg::RegTensor<uint64_t>&)narrow);
        uint32_t count32 = count - i * 32 < 32 ? count - i * 32 : 32;
        auto valid32 = Reg::UpdateMask<int32_t>(count32);
        Reg::StoreAlign(destination + i * 32, narrow, valid32);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <class InputType, bool precise>
__simd_vf__ inline void MakeGradientBlock(__ubuf__ InputType* destination, __ubuf__ float* source,
                                          __ubuf__ float* maxima, __ubuf__ float* sums, __ubuf__ int32_t* labels,
                                          uint32_t rows, uint32_t count, uint32_t inputPitch, uint32_t outputPitch,
                                          float alpha, int32_t vocabStart)
{
    Reg::RegTensor<float> value, corrected, maximum, denominator, one, inverse, zero;
    Reg::RegTensor<int32_t> target;
    Reg::RegTensor<InputType> low, lowZero;
    Reg::MaskReg targetMask, ignoredMask;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(one, 1.0f, full);
    Reg::Duplicate(zero, 0.0f, full);
    Reg::Cast<InputType, float, LOW_CAST>(lowZero, zero, full);
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t remaining = count, padded = outputPitch;
        const uint16_t activeBlocks = (count + 63) / 64;
        for (uint16_t i = 0; i < activeBlocks; ++i) {
            auto valid = Reg::UpdateMask<float>(remaining);
            auto write = Reg::UpdateMask<float>(padded);
            Reg::LoadAlign(value, source + row * inputPitch + i * 64);
            Reg::LoadAlign(maximum, maxima + i * 64);
            Reg::LoadAlign(denominator, sums + i * 64);
            Reg::LoadAlign(target, labels + i * 64);
            Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(ignoredMask, target, -1, valid);
            Reg::Sub(value, value, maximum, valid);
            if constexpr (precise) {
                Reg::Exp<float, &PRECISE_EXP>(corrected, value, valid);
                Reg::Move(value, corrected);
                Reg::Div<float, &PRECISE_DIV>(corrected, value, denominator, valid);
                Reg::Move(value, corrected);
            } else {
                Reg::Exp(value, value, valid);
                Reg::Div(inverse, one, denominator, valid);
                Reg::Mul(value, value, inverse, valid);
            }
            Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(targetMask, target, vocabStart + row, valid);
            Reg::Adds(corrected, value, -1.0f, valid);
            Reg::Select(value, corrected, value, targetMask);
            Reg::Muls(value, value, alpha, valid);
            Reg::Select(value, zero, value, ignoredMask);
            Reg::Cast<InputType, float, LOW_CAST>(low, value, valid);
            Reg::StoreAlign<InputType, Reg::StoreDist::DIST_PACK_B32>(destination + row * outputPitch + i * 64, low,
                                                                      write);
        }
        // Clear the fixed physical pitch without evaluating Exp/Div on empty
        // blocks when a short N tail follows a full window.
        for (uint16_t i = activeBlocks; i < (outputPitch + 63) / 64; ++i) {
            auto write = Reg::UpdateMask<float>(padded);
            Reg::StoreAlign<InputType, Reg::StoreDist::DIST_PACK_B32>(destination + row * outputPitch + i * 64, lowZero,
                                                                      write);
        }
    }
}

template <class InputType, bool addOld, bool final, bool applyAlpha>
__simd_vf__ inline void AccumulateBlock(__ubuf__ float* partial, __ubuf__ float* accumulator,
                                        __ubuf__ InputType* destination, uint32_t count, float upstreamScale,
                                        float alpha)
{
    Reg::RegTensor<float> value, previous;
    Reg::RegTensor<InputType> low;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < (count + 63) / 64; ++i) {
        auto mask = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(value, partial + i * 64);
        if constexpr (addOld) {
            Reg::LoadAlign(previous, accumulator + i * 64);
            Reg::Add(value, value, previous, mask);
        }
        if constexpr (final) {
            Reg::Muls(value, value, upstreamScale, mask);
            if constexpr (applyAlpha) {
                Reg::Muls(value, value, alpha, mask);
            }
            Reg::Cast<InputType, float, LOW_CAST>(low, value, mask);
            Reg::StoreAlign<InputType, Reg::StoreDist::DIST_PACK_B32>(destination + i * 64, low, mask);
        } else {
            Reg::StoreAlign(accumulator + i * 64, value, mask);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}
} // namespace ChunkLossGradOpt
