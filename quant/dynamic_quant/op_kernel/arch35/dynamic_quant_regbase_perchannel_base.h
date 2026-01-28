/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_regbase_perchannel_base.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_PERCHANNEL_BASE_H
#define DYNAMIC_QUANT_REGBASE_PERCHANNEL_BASE_H

namespace DynamicQuantPerChannel {
using namespace AscendC;

constexpr static MicroAPI::DivSpecificMode divHighPrecisionMode = {MicroAPI::MaskMergeMode::ZEROING, true};
inline  constexpr  MicroAPI::CastTrait castTraitB16ToB32 = {MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
inline  constexpr MicroAPI::CastTrait castTraitF32ToI16 = {MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
inline  constexpr MicroAPI::CastTrait castTraitI16ToF16 = {MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND};
inline  constexpr MicroAPI::CastTrait castTraitF16ToI8 = {MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};
inline  constexpr MicroAPI::CastTrait castTraitF32tofp8 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
inline  constexpr MicroAPI::CastTrait castTraitF32toh8 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float INT4_MAX_VALUE = 7.0f;

// isSymmetric == false 使用
constexpr float FP8_E5M2_OFFSET_VALUE = 114688.0f;
constexpr float FP8_E4M3FN_OFFSET_VALUE = 896.0f;
constexpr float HIFLOAT8_OFFSET_VALUE = 65536.0f;
constexpr float INT8_OFFSET_VALUE = 255.0f;
constexpr float INT4_OFFSET_VALUE = 15.0f;
constexpr float NEGATIVE_ONE = -1.0f;
constexpr uint32_t REG_LEN = 64;
constexpr uint32_t USE_BUFFER_NUM = 2;

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif
constexpr float POS_INFINITY = INFINITY;
constexpr float NEG_INFINITY = -INFINITY;
// 设定每种数据类型对应的最大值和offset最大值
template <typename yDtype>
__aicore__ inline void SetMaxValue(float &maxValue, float &offsetValue, float &offsetDivValue) {
    if constexpr (IsSameType<yDtype, int8_t>::value) {
        maxValue = static_cast<float>(1.0) / INT8_MAX_VALUE;
        offsetValue = INT8_MAX_VALUE;
        offsetDivValue = static_cast<float>(1.0) / INT8_OFFSET_VALUE;
    } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
        maxValue = static_cast<float>(1.0) / INT4_MAX_VALUE;
        offsetValue = INT4_MAX_VALUE;
        offsetDivValue = static_cast<float>(1.0) / INT4_OFFSET_VALUE;
    } else if constexpr (IsSameType<yDtype, fp8_e5m2_t>::value) {
        maxValue = static_cast<float>(1.0) / FP8_E5M2_MAX_VALUE;
        offsetValue = FP8_E5M2_MAX_VALUE;
        offsetDivValue = static_cast<float>(1.0) / FP8_E5M2_OFFSET_VALUE;
    } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value) {
        maxValue= static_cast<float>(1.0) / FP8_E4M3FN_MAX_VALUE;
        offsetValue = FP8_E4M3FN_MAX_VALUE;
        offsetDivValue = static_cast<float>(1.0) / FP8_E4M3FN_OFFSET_VALUE;
    } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
        maxValue = static_cast<float>(1.0) / HIFLOAT8_MAX_VALUE;
        offsetValue = HIFLOAT8_MAX_VALUE;
        offsetDivValue = static_cast<float>(1.0) / HIFLOAT8_OFFSET_VALUE;
    }
}

template <typename yDtype, typename yCopyDtype>
__aicore__ inline void CastToDstType(
    MicroAPI::RegTensor<float>& vregIn, MicroAPI::RegTensor<yCopyDtype>& vregOut, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<int16_t> vregCastI16;
    MicroAPI::RegTensor<half> vregCastF16;
    if constexpr (IsSameType<yDtype, int8_t>::value) {
        MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregCastI16, vregIn, mask);
        MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregCastF16, vregCastI16, mask);
        MicroAPI::Cast<yDtype, half, castTraitF16ToI8>(vregOut, vregCastF16, mask);
    } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
        MicroAPI::Cast<yDtype, float, castTraitF32toh8>(vregOut, vregIn, mask);
    } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value || IsSameType<yDtype, fp8_e5m2_t>::value) {
        MicroAPI::Cast<yDtype, float, castTraitF32tofp8>(vregOut, vregIn, mask);
    } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
        MicroAPI::RegTensor<uint16_t> vregPacked;
        MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(vregCastI16, vregIn, mask);
        MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(vregCastF16, vregCastI16, mask);
        MicroAPI::Pack(vregPacked, (MicroAPI::RegTensor<uint32_t>&)vregCastF16);
        MicroAPI::Cast<int4x2_t, half, castTraitF16ToI8>(
            (MicroAPI::RegTensor<int4x2_t>&)vregOut, (MicroAPI::RegTensor<half>&)vregPacked, mask);
    }
}
} // namespace DynamicQuantPerChannel
#endif  // DYNAMIC_QUANT_REGBASE_PERCHANNEL_SPLIT_M_H