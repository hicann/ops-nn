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
constexpr int32_t WELFORD_COUNT = 5120;
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
__simd_callee__ inline void LoadTensor(RegTensor<float>& dst, __ubuf__ T* srcAddr, MaskReg& pregLoop)
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
__simd_callee__ inline void CopyToTensor(__ubuf__ T* dstAddr, RegTensor<float>& src, MaskReg& pregLoop)
{
    if constexpr (std::is_same_v<T, float>) {
        DataCopy(dstAddr, src, pregLoop);
    } else {
        RegTensor<T> tmpFp16;
        Cast<T, float, castTraitB32ToB16>(tmpFp16, src, pregLoop);
        DataCopy<T, StoreDist::DIST_PACK_B32>(dstAddr, tmpFp16, pregLoop);
    }
}

template <typename T, typename OUT_DTYPE, uint8_t OP_CODE, bool hasSmooth>
__simd_callee__ inline void SingleComputeAdaption(RegTensor<float>& tmpMax, __ubuf__ float* normAddr, 
    __ubuf__ T* scaleAddr, __ubuf__ T* shiftAddr, __ubuf__ T* smoothAddr, __ubuf__ OUT_DTYPE* outAddr,
    uint32_t dataCount, uint16_t colLoopTimes, MaskReg& pregFull)
{
    RegTensor<float> x;
    RegTensor<float> scale;
    RegTensor<float> shift;

    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes;j ++) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy(x, normAddr + j * V_LENGTH);
        LoadTensor(scale, scaleAddr + j * V_LENGTH, pregLoop);
        LoadTensor(shift, shiftAddr + j * V_LENGTH, pregLoop);
        Adds(scale, scale, 1.0f, pregLoop);
        FusedMulDstAdd(x, scale, shift, pregLoop);
        if constexpr (hasSmooth) {
            RegTensor<float> smooth;
            LoadTensor(smooth, smoothAddr + j * V_LENGTH, pregLoop);
            Mul(x, x, smooth, pregLoop);
        }
        CopyToTensor(outAddr + j * V_LENGTH, x, pregLoop);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            Abs(x, x, pregLoop);
            Max(tmpMax, tmpMax, x, pregFull);
        }
    }
}

template <typename T, typename OUT_DTYPE, uint8_t OP_CODE, bool hasSmooth>
__simd_callee__ inline void DoubleComputeAdaption(RegTensor<float>& tmpMax1, RegTensor<float>& tmpMax2,
    __ubuf__ float* normAddr, __ubuf__ T* scaleAddr, __ubuf__ T* shiftAddr, __ubuf__ T* smoothAddr, 
    __ubuf__ OUT_DTYPE* outAddr, uint32_t dataCount, uint16_t colLoopTimes, uint32_t halfLength, MaskReg& pregFull)
{
    RegTensor<float> x1;
    RegTensor<float> x2;
    RegTensor<float> scale1;
    RegTensor<float> scale2;
    RegTensor<float> shift1;
    RegTensor<float> shift2;

    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes;j ++) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy(x1, normAddr + j * V_LENGTH);
        DataCopy(x2, normAddr + halfLength + j * V_LENGTH);
        LoadTensor(scale1, scaleAddr + j * V_LENGTH, pregLoop);
        LoadTensor(scale2, scaleAddr + halfLength + j * V_LENGTH, pregLoop);
        LoadTensor(shift1, shiftAddr + j * V_LENGTH, pregLoop);
        LoadTensor(shift2, shiftAddr + halfLength + j * V_LENGTH, pregLoop);
        Adds(scale1, scale1, 1.0f, pregLoop);
        Adds(scale2, scale2, 1.0f, pregLoop);
        FusedMulDstAdd(x1, scale1, shift1, pregLoop);
        FusedMulDstAdd(x2, scale2, shift2, pregLoop);
        if constexpr (hasSmooth) {
            RegTensor<float> smooth;
            LoadTensor(smooth, smoothAddr + j * V_LENGTH, pregLoop);
            Mul(x1, x1, smooth, pregLoop);
            Mul(x2, x2, smooth, pregLoop);
        }
        CopyToTensor(outAddr + j * V_LENGTH, x1, pregLoop);
        CopyToTensor(outAddr + halfLength + j * V_LENGTH, x2, pregLoop);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            Abs(x1, x1, pregLoop);
            Abs(x2, x2, pregLoop);
            Max(tmpMax1, tmpMax1, x1, pregFull);
            Max(tmpMax2, tmpMax2, x2, pregFull);
        }
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

template <typename T, typename OUT_DTYPE, uint8_t OP_CODE, bool hasSmooth>
__aicore__ inline void WelfordAdaptionVF(uint32_t dataCount, __ubuf__ float* normAddr, __ubuf__ T* scaleAddr,
    __ubuf__ T* shiftAddr, __ubuf__ T* smoothAddr, __ubuf__ float* maxTmpAddr, __ubuf__ OUT_DTYPE* outAddr)
{
    uint16_t colLoopTimes = static_cast<uint16_t>(CeilA2B(dataCount, V_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<float> tmpMax;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            DataCopy<float, LoadDist::DIST_BRC_B32>(tmpMax, maxTmpAddr);
        }
        SingleComputeAdaption<T, OUT_DTYPE, OP_CODE, hasSmooth>(tmpMax, normAddr, scaleAddr, shiftAddr, smoothAddr, outAddr, dataCount, colLoopTimes, pregFull);
        if constexpr (OP_CODE == QUANT_OP_CODE) {
            ReduceMax(tmpMax, tmpMax, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(maxTmpAddr, tmpMax, pregMerge);
        }
    }
}

template <typename T, typename OUT_DTYPE, uint8_t OP_CODE, bool hasSmooth>
__aicore__ inline void AdaptionVF(uint16_t rowNum, uint32_t hiddenDim, uint32_t hiddenDimCeil, float quantFactor,
    __ubuf__ float* normAddr, __ubuf__ T* scaleAddr, __ubuf__ T* shiftAddr, __ubuf__ T* smoothAddr,
    __ubuf__ OUT_DTYPE* outAddr, __ubuf__ float* quantScaleAddr)
{
    uint16_t rowLoopTimes = rowNum >> 1;
    uint16_t tailLoopTimes = rowNum & 1;
    uint16_t colLoopTimes = static_cast<uint16_t>(CeilA2B(hiddenDim, V_LENGTH));
    uint32_t halfLength = rowLoopTimes * hiddenDimCeil;

    __VEC_SCOPE__
    {
        RegTensor<float> tmpMax1;
        RegTensor<float> tmpMax2;
        RegTensor<float> quantScale1;
        RegTensor<float> quantScale2;

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
        for (uint16_t i = 0; i < rowLoopTimes; i++) {
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                Duplicate(tmpMax1, 0.0f, pregFull);
                Duplicate(tmpMax2, 0.0f, pregFull);
            }
            DoubleComputeAdaption<T, OUT_DTYPE, OP_CODE, hasSmooth>(tmpMax1, tmpMax2, normAddr, scaleAddr, 
                shiftAddr, smoothAddr, outAddr, hiddenDim, colLoopTimes, halfLength, pregFull);
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                ReduceMax(quantScale1, tmpMax1, pregFull);
                ReduceMax(quantScale2, tmpMax2, pregFull);
                Muls(quantScale1, quantScale1, quantFactor, pregMerge);
                Muls(quantScale2, quantScale2, quantFactor, pregMerge);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(quantScaleAddr + i, quantScale1, pregMerge);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(quantScaleAddr + i + rowLoopTimes, quantScale2, pregMerge);
            }
            normAddr += hiddenDimCeil;
            scaleAddr += hiddenDimCeil;
            shiftAddr += hiddenDimCeil;
            outAddr += hiddenDimCeil;
        }

        for (uint16_t i = 0; i < tailLoopTimes; i++) {
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                Duplicate(tmpMax1, 0.0f, pregFull);
            }
            SingleComputeAdaption<T, OUT_DTYPE, OP_CODE, hasSmooth>(tmpMax1, normAddr + halfLength, scaleAddr + halfLength, 
                shiftAddr + halfLength, smoothAddr, outAddr + halfLength, hiddenDim, colLoopTimes, pregFull);
            if constexpr (OP_CODE == QUANT_OP_CODE) {
                ReduceMax(quantScale1, tmpMax1, pregFull);
                Muls(quantScale1, quantScale1, quantFactor, pregMerge);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(quantScaleAddr + rowLoopTimes + rowLoopTimes, quantScale1, pregMerge);
            }
        }
    }
}

} // namespace AdaLayerNormNS

#endif // ADA_LAYER_NORM_COMMON_H
