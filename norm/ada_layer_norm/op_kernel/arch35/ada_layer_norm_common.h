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
 * \file ada_layer_norm_common.h
 * \brief
 */
#ifndef ADA_LAYER_NORM_COMMON_H
#define ADA_LAYER_NORM_COMMON_H

#include "../ada_layer_norm_util.h"

namespace AdaLayerNormNS {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr uint16_t V_LENGTH = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint16_t TWO_V_LENGTH = V_LENGTH << 1;
constexpr float FACTOR_FP8_E5M2 = 1.0f / 57344.0f;
constexpr float FACTOR_FP8_E4M3FN = 1.0f / 448.0f;
constexpr float FACTOR_HIFLOAT8 = 1.0f / 32768.0f;

constexpr CastTrait castTraitB16ToB32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait castTraitB32ToB16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait castTraitF32ToI16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait castTraitI16ToF16 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr CastTrait castTraitF16ToI8 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
constexpr CastTrait castTraitF32Tofp8 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait castTraitF32Toh8 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr LayerNormConfig hasGammaBetaConfig = {false, false, false};
constexpr LayerNormConfig hasGammaNoBetaConfig = {true, false, false};
constexpr LayerNormConfig noGammaHasBetaConfig = {false, true, false};
constexpr LayerNormConfig noGammaNoBetaConfig = {true, true, false};

constexpr NormalizeConfig hasGammaBetaNormalizeConfig = {ReducePattern::AR, -1, false, false, false};
constexpr NormalizeConfig hasGammaNoBetaNormalizeConfig = {ReducePattern::AR, -1, true, false, false};
constexpr NormalizeConfig noGammaHasBetaNormalizeConfig = {ReducePattern::AR, -1, false, true, false};
constexpr NormalizeConfig noGammaNoBetaNormalizeConfig = {ReducePattern::AR, -1, true, true, false};

template <typename T>
__aicore__ inline void LoadTensor(RegTensor<float>& dst, __local_mem__ T* srcAddr, MaskReg& pregLoop)
{
    if constexpr (std::is_same_v<T, float>) {
        DataCopy(dst, srcAddr);
    } else {
        RegTensor<T> tmpFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(tmpFp16, srcAddr);
        Cast<float, T, castTraitB16ToB32>(dst, tmpFp16, pregLoop);
    }
}

template <typename T>
__aicore__ inline void CopyToTensor(__local_mem__ T* dstAddr, RegTensor<float>& src, MaskReg& pregLoop)
{
    if constexpr (std::is_same_v<T, float>) {
        DataCopy(dstAddr, src, pregLoop);
    } else {
        RegTensor<T> tmpFp16;
        Cast<T, float, castTraitB32ToB16>(tmpFp16, src, pregLoop);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr, tmpFp16, pregLoop);
    }
}

template <typename T>
__aicore__ inline float GetQuantFactor()
{
    if constexpr (std::is_same_v<T, hifloat8_t>) {
        return FACTOR_HIFLOAT8;
    } else if constexpr (std::is_same_v<T, fp8_e4m3fn_t>) {
        return FACTOR_FP8_E4M3FN;
    } else if constexpr (std::is_same_v<T, fp8_e5m2_t>) {
        return FACTOR_FP8_E5M2;
    } else {
        return FACTOR_INT8;
    }
}

} // namespace AdaLayerNormNS

#endif // ADA_LAYER_NORM_COMMON_H
