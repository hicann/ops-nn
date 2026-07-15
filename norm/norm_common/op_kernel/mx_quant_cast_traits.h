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
 * \file mx_quant_cast_traits.h
 * \brief shared CastTrait definitions for mx_quant operators, extracted to namespace scope
 *        to avoid GCC 14.2 ICE with static constexpr inside template functions
 */
#ifndef MX_QUANT_CAST_TRAITS_H
#define MX_QUANT_CAST_TRAITS_H
#include "kernel_operator.h"

namespace MxQuantCastTraits {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Float = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait castTraitZero = {AscendC::MicroAPI::RegLayout::ZERO,
                                                        AscendC::MicroAPI::SatMode::UNKNOWN,
                                                        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait castTraitOne = {AscendC::MicroAPI::RegLayout::ONE,
                                                       AscendC::MicroAPI::SatMode::UNKNOWN,
                                                       AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait castTrait32to80 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTrait32to81 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTrait32to82 = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTrait32to83 = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTraitF16toFp32Zero = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

constexpr AscendC::MicroAPI::CastTrait castTraitF16toFp32One = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN};

template <AscendC::RoundMode RM>
constexpr AscendC::MicroAPI::CastTrait castTraitRM = {AscendC::MicroAPI::RegLayout::ZERO,
                                                      AscendC::MicroAPI::SatMode::UNKNOWN,
                                                      AscendC::MicroAPI::MaskMergeMode::ZEROING, RM};

template <AscendC::RoundMode RM>
constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Bf16RM = {AscendC::MicroAPI::RegLayout::UNKNOWN,
                                                               AscendC::MicroAPI::SatMode::UNKNOWN,
                                                               AscendC::MicroAPI::MaskMergeMode::ZEROING, RM};

template <AscendC::RoundMode RM>
constexpr AscendC::MicroAPI::CastTrait castTraitFp32toBF16RM = {AscendC::MicroAPI::RegLayout::ZERO,
                                                                AscendC::MicroAPI::SatMode::NO_SAT,
                                                                AscendC::MicroAPI::MaskMergeMode::ZEROING, RM};

constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

} // namespace MxQuantCastTraits

#endif // MX_QUANT_CAST_TRAITS_H
