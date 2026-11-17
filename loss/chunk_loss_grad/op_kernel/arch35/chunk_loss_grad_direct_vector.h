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
#include "chunk_loss_grad_vector.h"

namespace ChunkLossGradDirect {
namespace Reg = AscendC::MicroAPI;

// P is split along V by Fixpipe. Keep the full physical H256 row pitch
// while only scaling/casting the real HC columns of each local row.
template <class InputType, bool applyAlpha>
__simd_vf__ inline void FinalizeRows(__ubuf__ InputType* destination, __ubuf__ float* source, uint32_t rows,
                                     uint32_t columns, uint32_t outPitch, float upstreamScale, float alpha)
{
    using namespace ChunkLossGradArch35;
    Reg::RegTensor<float> value;
    Reg::RegTensor<InputType> low;
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t remaining = columns;
        for (uint16_t i = 0; i < (columns + 63) / 64; ++i) {
            auto mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign(value, source + uint32_t(row) * 256 + i * 64);
            Reg::Muls(value, value, upstreamScale, mask);
            if constexpr (applyAlpha) {
                Reg::Muls(value, value, alpha, mask);
            }
            Reg::Cast<InputType, float, LOW_CAST>(low, value, mask);
            Reg::StoreAlign<InputType, Reg::StoreDist::DIST_PACK_B32>(destination + uint32_t(row) * outPitch + i * 64,
                                                                      low, mask);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// NZ UB: [outPitch/16, storageRows+1,16]. The extra row is skipped by UB->L1.
// Arithmetic and rounding match MakeGradientBlock. Pack only moves bit fields.
template <class InputType, bool precise>
__simd_vf__ inline void MakeGradientNZ(__ubuf__ InputType* destination, __ubuf__ float* source, __ubuf__ float* maxima,
                                       __ubuf__ float* sums, __ubuf__ int32_t* labels, uint32_t validRows,
                                       uint32_t storageRows, uint32_t count, uint32_t outPitch, float alpha,
                                       int32_t vocabStart)
{
    using namespace ChunkLossGradArch35;
    Reg::RegTensor<float> value, corrected, maximum, denominator, one, zero, inverse;
    Reg::RegTensor<int32_t> target;
    Reg::RegTensor<InputType> low, lowZero, packed;
    Reg::MaskReg targetMask, ignoredMask;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(one, 1.0f, full);
    Reg::Duplicate(zero, 0.0f, full);
    Reg::Cast<InputType, float, LOW_CAST>(lowZero, zero, full);
    for (uint16_t row = 0; row < storageRows; ++row) {
        uint32_t remaining = count;
        for (uint16_t i = 0; i < (outPitch + 63) / 64; ++i) {
            auto valid = Reg::UpdateMask<float>(remaining);
            if (row < validRows) {
                Reg::LoadAlign(value, source + row * 128 + i * 64);
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
            } else {
                Reg::Move(low, lowZero);
            }
            Reg::Pack((Reg::RegTensor<uint16_t>&)packed, (Reg::RegTensor<uint32_t>&)low);
            uint32_t lanes = outPitch - i * 64 < 64 ? outPitch - i * 64 : 64;
            auto write = Reg::UpdateMask<InputType>(lanes);
            __ubuf__ InputType* address = destination + uint32_t(i) * 64 * (storageRows + 1) + uint32_t(row) * 16;
            Reg::DataCopy<InputType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                address, packed, storageRows + 1, 1, write);
        }
    }
}
// Reuse the immutable N metadata registers across all local V rows.
// Reciprocal candidates keep the existing registers, Cast and NZ layout.
template <class InputType, bool precise, bool hoistSum = false>
__simd_vf__ inline void MakeGradientNZReuseMetadata(__ubuf__ InputType* destination, __ubuf__ float* source,
                                                    __ubuf__ float* maxima, __ubuf__ float* sums,
                                                    __ubuf__ int32_t* labels, uint32_t validRows, uint32_t storageRows,
                                                    uint32_t count, uint32_t outPitch, float alpha, int32_t vocabStart)
{
    using namespace ChunkLossGradArch35;
    Reg::RegTensor<float> value, corrected, maximum, denominator, one, zero, inverse;
    Reg::RegTensor<int32_t> target;
    Reg::RegTensor<InputType> low, lowZero, packed;
    Reg::MaskReg targetMask, ignoredMask;
    auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(one, 1.0f, full);
    Reg::Duplicate(zero, 0.0f, full);
    Reg::Cast<InputType, float, LOW_CAST>(lowZero, zero, full);
    uint32_t remaining = count;
    for (uint16_t i = 0; i < (outPitch + 63) / 64; ++i) {
        auto valid = Reg::UpdateMask<float>(remaining);
        Reg::LoadAlign(maximum, maxima + i * 64);
        Reg::LoadAlign(denominator, sums + i * 64);
        Reg::LoadAlign(target, labels + i * 64);
        Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(ignoredMask, target, -1, valid);
        if constexpr (precise && hoistSum) {
            Reg::Div<float, &PRECISE_DIV>(corrected, one, denominator, valid);
            Reg::Move(denominator, corrected);
        }
        uint32_t lanes = outPitch - i * 64 < 64 ? outPitch - i * 64 : 64;
        auto write = Reg::UpdateMask<InputType>(lanes);
        for (uint16_t row = 0; row < storageRows; ++row) {
            if (row < validRows) {
                Reg::LoadAlign(value, source + row * 128 + i * 64);
                Reg::Sub(value, value, maximum, valid);
                if constexpr (precise) {
                    Reg::Exp<float, &PRECISE_EXP>(corrected, value, valid);
                    Reg::Move(value, corrected);
                    if constexpr (hoistSum) {
                        Reg::Mul(value, value, denominator, valid);
                    } else {
                        Reg::Div<float, &PRECISE_DIV>(corrected, value, denominator, valid);
                        Reg::Move(value, corrected);
                    }
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
            } else {
                Reg::Move(low, lowZero);
            }
            Reg::Pack((Reg::RegTensor<uint16_t>&)packed, (Reg::RegTensor<uint32_t>&)low);
            __ubuf__ InputType* address = destination + uint32_t(i) * 64 * (storageRows + 1) + uint32_t(row) * 16;
            Reg::DataCopy<InputType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                address, packed, storageRows + 1, 1, write);
        }
    }
}
} // namespace ChunkLossGradDirect
