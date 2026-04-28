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
 * \file quant_max_common.h
 * \brief quantmax kernel base
 */

#ifndef QUANT_MAX_COMMON_H
#define QUANT_MAX_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "op_kernel/platform_util.h"
#include "quant_max_struct.h"

namespace QuantMax {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, typename U, uint64_t RoundMode>
class QuantMaxBase {
public:
    __aicore__ inline QuantMaxBase(){};

protected:
    constexpr static int32_t BLOCK_SIZE = GetUbBlockSize();

protected:
    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_HALF_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_BF16_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN};

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_HIFP8 = []() {
        if constexpr (RoundMode == TPL_ROUND_MODE_HYBRID) {
            return AscendC::Reg::CastTrait{
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
                RoundMode::CAST_HYBRID};
        } else {
            return AscendC::Reg::CastTrait{
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
                RoundMode::CAST_ROUND};
        }
    }();

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_FP8E5M2 = []() {
        return AscendC::Reg::CastTrait{
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_FP8E4M3 = []() {
        return AscendC::Reg::CastTrait{
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_BF16 = []() {
        return AscendC::Reg::CastTrait{
            AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
            RoundMode::CAST_RINT};
    }();
};

} // namespace QuantMax
#endif