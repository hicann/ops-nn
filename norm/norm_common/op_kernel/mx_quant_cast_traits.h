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
using namespace AscendC::Reg;

constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16 = {AscendC::Reg::RegLayout::UNKNOWN,
                                                        AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

constexpr AscendC::Reg::CastTrait castTraitHalf2Float = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                         AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTraitZero = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                   AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTraitOne = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                  AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTrait32to80 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTrait32to81 = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTrait32to82 = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTrait32to83 = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTraitF16toFp32Zero = {AscendC::Reg::RegLayout::ZERO,
                                                            AscendC::Reg::SatMode::UNKNOWN,
                                                            AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTraitF16toFp32One = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                           AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <AscendC::RoundMode RM>
constexpr AscendC::Reg::CastTrait castTraitRM = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                 AscendC::Reg::MaskMergeMode::ZEROING, RM};

template <AscendC::RoundMode RM>
constexpr AscendC::Reg::CastTrait castTraitHalf2Bf16RM = {
    AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING, RM};

template <AscendC::RoundMode RM>
constexpr AscendC::Reg::CastTrait castTraitFp32toBF16RM = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING, RM};

constexpr AscendC::Reg::CastTrait castTrait32to8 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                    AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

} // namespace MxQuantCastTraits

#endif // MX_QUANT_CAST_TRAITS_H
