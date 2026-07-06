/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_block_quant_common.h
 * \brief
 */

#ifndef DYNAMIC_BLOCK_QUANT_COMMON_H
#define DYNAMIC_BLOCK_QUANT_COMMON_H
namespace DynamicBlockQuant {

using namespace AscendC;

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

constexpr int64_t DB_BUFFER = 2;
constexpr uint16_t BLOCK_BYTE_32 = 32;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3_MAX_VALUE = 448.0f;
constexpr float HIFP8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr uint32_t FP32_INF_VALUE = 0x7f800000;
constexpr uint16_t FP16_INF_VALUE = 0x7c00;
constexpr uint16_t BF16_INF_VALUE = 0x7f80;
// 1 / max_dtype
constexpr uint32_t INV_FP8_E5M2_MAX_VALUE = 0x37924925;
constexpr uint32_t INV_FP8_E4M3_MAX_VALUE = 0x3b124925;
constexpr uint32_t INV_HIFP8_MAX_VALUE = 0x38000000;

template <typename OUT_TYPE>
__aicore__ inline float GetDstTypeMaxValue(float dstTypeMax)
{
    if constexpr (IsSameType<OUT_TYPE, fp8_e5m2_t>::value) {
        return FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<OUT_TYPE, fp8_e4m3fn_t>::value) {
        return FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<OUT_TYPE, hifloat8_t>::value) {
        return dstTypeMax != 0 ? dstTypeMax : HIFP8_MAX_VALUE;
    } else if constexpr (IsSameType<OUT_TYPE, int8_t>::value) {
        return INT8_MAX_VALUE;
    }
    return 0.0f;
}

constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

constexpr static AscendC::MicroAPI::CastTrait castTrait32tofp8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr static AscendC::MicroAPI::CastTrait castTraitF32ToI16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr static AscendC::MicroAPI::CastTrait castTraitI16ToF16 = {
    AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_FLOOR};

constexpr static AscendC::MicroAPI::CastTrait castTraitF16ToI8 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_FLOOR};
} // namespace DynamicBlockQuant
#endif // DYNAMIC_BLOCK_QUANT_COMMON_H
