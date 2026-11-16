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
 * \file quantize_add_layer_norm_regbase_helper.h
 * \brief ascend950 (arch35/regbase) helpers for QuantizeAddLayerNorm.
 *        Mirrors add_layer_norm_quant_regbase_helper.h, single-path static quant only
 *        (no dual scales/offsets, no dynamic quant).
 *        The mean/var statistics family (VFCalcMeanVar*) lives in
 *        quantize_add_layer_norm_regbase_stats.h (header-size compliance).
 */

#ifndef QUANTIZE_ADD_LAYER_NORM_REGBASE_COMMON_H
#define QUANTIZE_ADD_LAYER_NORM_REGBASE_COMMON_H

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AddLayerNorm {
using AscendC::Reg::LoadAlign;
using AscendC::Reg::Reduce;
using AscendC::Reg::StoreAlign;
} // namespace AddLayerNorm

#include "../../add_layer_norm/arch35/add_layer_norm_regbase_common.h"
#include "quantize_add_layer_norm_regbase_residual.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace NsQuantizeAddLayerNorm {
using namespace AddLayerNorm;
using namespace AscendC;
using AscendC::Reg::Truncate;

// OPT_CODE bits: scales is required (always present), zero_points optional
#define OFFSET_CODE 0x1

#define IS_OFFSET_EXIST ((OPT_CODE & OFFSET_CODE) > 0)

// Quant mode digit of the RegBase tiling key: 0 = mul, 1 = per-channel div, 2 = per-tensor scalar mul.
#define IS_DIV_SCALE (((TILING_KEY / 10) % 10) == 1)
#define IS_PER_TENSOR_SCALE (((TILING_KEY / 10) % 10) == 2)

#define CONST_CONDITIONAL_EXPR(_cond, _expr) \
    if constexpr (_cond) {                   \
        _expr;                               \
    }

#define CONST_CONDITIONAL_ASSIGN(_cond, _var, _expr) \
    if constexpr (_cond) {                           \
        _var = _expr;                                \
    }

constexpr uint32_t QALN_VL_FP32 = GetVecLen() / sizeof(float);
constexpr uint32_t QALN_BLOCK_SIZE = GetDataBlockSizeInBytes();
static_assert(QALN_VL_FP32 == static_cast<uint32_t>(AddLayerNorm::VL_FP32),
              "QALN and AddLayerNorm must use the same FP32 vector length");

constexpr AscendC::Reg::CastTrait QUANT_CAST_TRAIT_F32_TO_F16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait QUANT_CAST_TRAIT_F16_TO_S8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr AscendC::Reg::DivSpecificMode DIV_HIGH_PREC_MODE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};

constexpr float QALN_MIN_NORMAL_FP32 = 1.1754943508222875e-38F;
constexpr AscendC::Reg::DivSpecificMode RSTD_DIV_0ULP_FTZ_FALSE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    false,
    AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE,
};
constexpr AscendC::Reg::SqrtSpecificMode RSTD_SQRT_0ULP_FTZ_FALSE = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    false,
    AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE,
};

template <typename T>
__simd_callee__ inline void LoadQuantParams(__ubuf__ T* ubAddr, RegTensor<float>& dstTensor, MaskReg& preg,
                                            uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dstTensor, (__ubuf__ float*)ubAddr + offset);
    } else {
        RegTensor<T> dstB16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(dstB16, (__ubuf__ T*)ubAddr + offset);
        Cast<float, T, castTraitB162B32>(dstTensor, dstB16, preg);
    }
}

// Per-tensor quantization uses scalar scales/zero_points in GM; the kernel copies only
// element 0 into UB, then broadcasts it to the whole register via a DIST_BRC load.
// (same mechanism as the mean/rstd broadcast loads).
template <typename T>
__simd_callee__ inline void LoadScalarQuantParam(__ubuf__ T* ubAddr, RegTensor<float>& dstTensor, MaskReg& preg)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstTensor, (__ubuf__ float*)ubAddr);
    } else {
        RegTensor<T> dstB16;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(dstB16, (__ubuf__ T*)ubAddr);
        Cast<float, T, castTraitB162B32>(dstTensor, dstB16, preg);
    }
}

__simd_callee__ inline void Round2Int8(RegTensor<int8_t>& dstTensor, RegTensor<float>& srcTensor, MaskReg& preg)
{
    RegTensor<half> tmpFp16;
    RegTensor<float> tmpFp32;
    Truncate<float, RoundMode::CAST_RINT>(tmpFp32, srcTensor, preg);
    Cast<half, float, QUANT_CAST_TRAIT_F32_TO_F16>(tmpFp16, tmpFp32, preg);
    Cast<int8_t, half, QUANT_CAST_TRAIT_F16_TO_S8>(dstTensor, tmpFp16, preg);
}

__simd_callee__ inline void DichotomyAddResidual(RegTensor<float>& dstReg, __ubuf__ float* src, uint16_t outerLoop,
                                                 uint16_t innerLoop, uint32_t lastNum)
{
    RegTensor<float> left;
    RegTensor<float> right;
    RegTensor<float> sum;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t k = 0; k < outerLoop; k++) {
        innerLoop /= 2;
        for (uint16_t i = 0; i < innerLoop; i++) {
            LoadAlign(left, src + i * VL_FP32);
            LoadAlign(right, src + (i + innerLoop) * VL_FP32);
            Add(sum, left, right, pregMain);
            StoreAlign(src + i * VL_FP32, sum, pregMain);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    uint32_t remaining = lastNum;
    MaskReg pregLoop = UpdateMask<float>(remaining);
    LoadAlign(sum, src);
    Reduce<AscendC::Reg::ReduceType::SUM>(dstReg, sum, pregLoop);
}

template <bool NEED_MAX = true>
__simd_callee__ inline void ComputeRstdNewtonRaphsonResidual(RegTensor<float>& var, RegTensor<float>& rstd,
                                                             MaskReg& preg, float epsilon)
{
    constexpr float maxFinite = __FLT_MAX__;
    constexpr float scalar1 = -0.5F;
    constexpr float scalar2 = 1.5F;
    constexpr float scalar3 = 0.5F;
    constexpr float scalar0 = -99.99F;

    RegTensor<float> reciprocal;
    RegTensor<float> estimate;
    RegTensor<float> correction;
    RegTensor<float> temp;
    RegTensor<float> one;
    RegTensor<float> half;
    RegTensor<float> oneAndHalf;
    RegTensor<float> residual;
    RegTensor<float> squareResidual;
    RegTensor<float> scalarInf;
    RegTensor<float> scalarZero;
    MaskReg cmpZero;
    MaskReg cmpInf;

    Duplicate(scalarInf, maxFinite, preg);
    Duplicate(scalarZero, 0.0F, preg);
    Duplicate(one, 1.0F, preg);
    Duplicate(half, scalar3, preg);
    Duplicate(oneAndHalf, scalar2, preg);
    Duplicate(correction, 1.0F, preg);

    Adds(var, var, epsilon, preg);
    if constexpr (NEED_MAX) {
        Maxs(var, var, scalar0, preg);
    }
    Div(reciprocal, one, var, preg);
    Sqrt(estimate, reciprocal, preg);
    Muls(temp, var, scalar1, preg);
    Mul(temp, temp, estimate, preg);
    Mula(oneAndHalf, temp, estimate, preg);
    Mul(rstd, estimate, oneAndHalf, preg);
    Muls(residual, var, -1.0F, preg);
    Mula(correction, residual, reciprocal, preg);
    Muls(squareResidual, rstd, -1.0F, preg);
    Mula(reciprocal, squareResidual, rstd, preg);
    Mula(correction, var, reciprocal, preg);
    Mul(correction, correction, rstd, preg);
    Mula(rstd, correction, half, preg);
    CompareScalar(cmpZero, var, maxFinite, preg);
    Select(rstd, scalarZero, rstd, cmpZero);
    CompareScalar(cmpInf, var, 0.0F, preg);
    Select(rstd, scalarInf, rstd, cmpInf);
}

template <bool NEED_MAX = true>
__simd_callee__ inline void ComputeRstdSqrtFirstResidual(RegTensor<float>& var, RegTensor<float>& rstd, MaskReg& preg,
                                                         float epsilon)
{
    constexpr float scalar0 = -99.99F;
    RegTensor<float> one;
    RegTensor<float> denominator;

    Duplicate(one, 1.0F, preg);
    Adds(denominator, var, epsilon, preg);
    if constexpr (NEED_MAX) {
        Maxs(denominator, denominator, scalar0, preg);
    }
    Sqrt<float, &RSTD_SQRT_0ULP_FTZ_FALSE>(denominator, denominator, preg);
    Div<float, &RSTD_DIV_0ULP_FTZ_FALSE>(rstd, one, denominator, preg);
}

template <bool NEED_MAX = true>
__aicore__ inline void ComputeRstdSqrtFirst(LocalTensor<float> srcLocal, LocalTensor<float> dstLocal, uint32_t rowCount,
                                            float epsilon, uint32_t vectorLen = QALN_VL_FP32)
{
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    uint16_t loopRows = static_cast<uint16_t>((rowCount + vectorLen - 1) / vectorLen);
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregLoop;
        uint32_t remaining = rowCount;
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(remaining);
            DataCopy(var, src + i * vectorLen);
            ComputeRstdSqrtFirstResidual<NEED_MAX>(var, rstd, pregLoop, epsilon);
            DataCopy(dst + i * vectorLen, rstd, pregLoop);
        }
    }
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> MakeZeroPadParams(int32_t copyLen, int32_t blockSize)
{
    int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(T), blockSize) / sizeof(T);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.paddingValue = static_cast<T>(0.0);
    padParams.rightPadding = copyLenAlign - copyLen;
    return padParams;
}

template <typename T>
__aicore__ inline DataCopyExtParams MakeDataCopyParams(int32_t copyLen, uint16_t blockCount = 1, uint16_t dstStride = 0,
                                                       uint16_t srcStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = srcStride;
    dataCopyParams.dstStride = dstStride;
    return dataCopyParams;
}

template <typename X1_TYPE, typename GammaQueue, typename BetaQueue>
__aicore__ inline void CopyGammaAndBetaToUBCommon(LocalTensor<X1_TYPE> gammaLocal, LocalTensor<X1_TYPE> betaLocal,
                                                  GlobalTensor<X1_TYPE>& gammaGm, GlobalTensor<X1_TYPE>& betaGm,
                                                  GammaQueue& gammaQueue, BetaQueue& betaQueue, int64_t offset,
                                                  int32_t copyLen, int32_t blockSize)
{
    DataCopyPadExtParams<X1_TYPE> padParams = MakeZeroPadParams<X1_TYPE>(copyLen, blockSize);
    DataCopyExtParams dataCopyParams = MakeDataCopyParams<X1_TYPE>(copyLen);
    DataCopyPad(betaLocal, betaGm[offset], dataCopyParams, padParams);
    betaQueue.EnQue(betaLocal);
    DataCopyPad(gammaLocal, gammaGm[offset], dataCopyParams, padParams);
    gammaQueue.EnQue(gammaLocal);
}

template <bool INIT, typename X1_TYPE, int32_t TILING_KEY>
__aicore__ inline void VFWelfordParallelUpdateCommon(LocalTensor<X1_TYPE>& x1Local, LocalTensor<X1_TYPE>& x2Local,
                                                     LocalTensor<X1_TYPE>& biasLocal, LocalTensor<X1_TYPE>& xLocal,
                                                     LocalTensor<float>& tmpMeanLocal, LocalTensor<float>& tmpVarLocal,
                                                     uint64_t calLen, uint16_t loopCount, float scale)
{
    __ubuf__ X1_TYPE* x1Addr = (__ubuf__ X1_TYPE*)x1Local[0].GetPhyAddr();
    __ubuf__ X1_TYPE* x2Addr = (__ubuf__ X1_TYPE*)x2Local[0].GetPhyAddr();
    __ubuf__ X1_TYPE* xOutAddr = (__ubuf__ X1_TYPE*)xLocal[0].GetPhyAddr();
    __ubuf__ float* tmpMeanAddr = (__ubuf__ float*)tmpMeanLocal.GetPhyAddr();
    __ubuf__ float* tmpVarAddr = (__ubuf__ float*)tmpVarLocal.GetPhyAddr();

    __ubuf__ X1_TYPE* biasAddr = nullptr;
    if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
        biasAddr = (__ubuf__ X1_TYPE*)biasLocal[0].GetPhyAddr();
    }

    asc_vf_call<VFWelfordParallelUpdateResidualFirst<INIT, X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>>(
        x1Addr, x2Addr, biasAddr, xOutAddr, tmpMeanAddr, tmpVarAddr, calLen, loopCount, scale);
}

__aicore__ inline void VFWelfordParallelFinalizeNonAlign(
    LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal, LocalTensor<float>& tmpMeanLocal,
    LocalTensor<float>& tmpVarLocal, LocalTensor<float>& dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float reduceScaleCorrection, float cnt, float eps)
{
    __ubuf__ float* meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal[0].GetPhyAddr();
    __ubuf__ float* tmpMeanAddr = (__ubuf__ float*)tmpMeanLocal[0].GetPhyAddr();
    __ubuf__ float* tmpVarAddr = (__ubuf__ float*)tmpVarLocal.GetPhyAddr();
    __ubuf__ float* dichotomyAddAddr = (__ubuf__ float*)dichotomyAddLocal.GetPhyAddr();
    AddLayerNorm::VFWelfordParallelFinalizeNonAlign(meanAddr, rstdAddr, tmpMeanAddr, tmpVarAddr, dichotomyAddAddr,
                                                    reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum,
                                                    offset, tailSize, reduceScale, reduceScaleCorrection, cnt, eps);
}

// Welford finalize aligned-scenario formulas:
// finalize_mean = sum_fun(mean) / parallel_N.
// finalize_delta = mean - finalize_mean.
// finalize_delta_square = finalize_delta * finalize_delta.
// M2_fixed = M2 + float(count) * finalize_delta_square
// finalize_std = sum_fun(M2_fixed) / float(parallel_N * count).
// Welford computes mean and variance with dichotomy accumulation.
// scale / reduceScale are exact power-of-two reciprocals; scaleCorrection / reduceScaleCorrection
// restore the true 1/colsPerLoop / 1/cols factor in one Muls after DichotomyAdd.

// Mean half of VFWelfordParallelFinalizeAlign: merge tail+full blocks into dichotomyAddAddr.
// Scope-owning phase function (VFCalcMeanPhase/VFCalcVarPhase pattern in
// quantize_add_layer_norm_regbase_stats.h): registers/masks are function-local and the driver
// exchanges data through UB pointers only. The phase boundary must not carry RegTensor or MaskReg
// references across VF launches.
__simd_vf__ inline void WelfordFinalizeAlignMeanPartVf(__ubuf__ float* tmpMeanAddr, __ubuf__ float* dichotomyAddAddr,
                                                       float scale, uint32_t dichotomyAddReminder,
                                                       uint16_t dichotomyAddReminderLoopCount,
                                                       uint16_t dichotomyAddPowerLoopCount, uint32_t dichotomyAddPower)
{
    RegTensor<float> dichotomyAddMeanL;
    RegTensor<float> dichotomyAddMeanR;
    RegTensor<float> sumMean;
    RegTensor<float> mean;
    MaskReg pregLoop;
    MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    uint32_t remaining = dichotomyAddReminder;
    for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
        pregLoop = UpdateMask<float>(remaining);
        LoadAlign(dichotomyAddMeanL, tmpMeanAddr + i * VL_FP32);
        LoadAlign(dichotomyAddMeanR, tmpMeanAddr + i * VL_FP32 + dichotomyAddPower);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
        Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale, pregLoop);
        Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
        ReduceSum(mean, sumMean, pregMain);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddAddr + i, mean, pregMerge);
    }

    for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount); i++) {
        LoadAlign(dichotomyAddMeanL, tmpMeanAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
        ReduceSum(mean, dichotomyAddMeanL, pregMain);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddAddr + dichotomyAddReminderLoopCount + i, mean, pregMerge);
    }
}

__aicore__ inline void WelfordFinalizeAlignMeanPart(__ubuf__ float* tmpMeanAddr, __ubuf__ float* dichotomyAddAddr,
                                                    float scale, uint32_t dichotomyAddReminder,
                                                    uint16_t dichotomyAddReminderLoopCount,
                                                    uint16_t dichotomyAddPowerLoopCount, uint32_t dichotomyAddPower)
{
    asc_vf_call<WelfordFinalizeAlignMeanPartVf>(tmpMeanAddr, dichotomyAddAddr, scale, dichotomyAddReminder,
                                                dichotomyAddReminderLoopCount, dichotomyAddPowerLoopCount,
                                                dichotomyAddPower);
}

// Var half of VFWelfordParallelFinalizeAlign: M2 fixed by the finalize delta, merged into
// dichotomyAddAddr. Scope-owning like WelfordFinalizeAlignMeanPart above; mean is broadcast
// back from the driver's lane-0 store via a DIST_BRC_B32 load (VFCalcVarPhase mechanism).
__simd_vf__ inline void WelfordFinalizeAlignVarPartVf(__ubuf__ float* tmpMeanAddr, __ubuf__ float* tmpVarAddr,
                                                      __ubuf__ float* dichotomyAddAddr, __ubuf__ float* meanAddr,
                                                      float cnt, float reduceScale, uint32_t dichotomyAddReminder,
                                                      uint16_t dichotomyAddReminderLoopCount,
                                                      uint16_t dichotomyAddPowerLoopCount, uint32_t dichotomyAddPower)
{
    RegTensor<float> dichotomyAddMeanL;
    RegTensor<float> dichotomyAddMeanR;
    RegTensor<float> dichotomyAddVarL;
    RegTensor<float> dichotomyAddVarR;
    RegTensor<float> sumVar;
    RegTensor<float> var;
    RegTensor<float> deltaL;
    RegTensor<float> deltaR;
    RegTensor<float> mean;
    MaskReg pregLoop;
    MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    uint32_t remaining = dichotomyAddReminder;

    LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, meanAddr);

    for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
        pregLoop = UpdateMask<float>(remaining);
        LoadAlign(dichotomyAddMeanL, tmpMeanAddr + i * VL_FP32);
        Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
        Mul(deltaL, deltaL, deltaL, pregMain);
        Muls(deltaL, deltaL, cnt, pregMain);
        LoadAlign(dichotomyAddVarL, tmpVarAddr + i * VL_FP32);
        Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
        Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

        LoadAlign(dichotomyAddMeanR, tmpMeanAddr + i * VL_FP32 + dichotomyAddPower);
        Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
        Mul(deltaR, deltaR, deltaR, pregLoop);
        Muls(deltaR, deltaR, cnt, pregLoop);
        LoadAlign(dichotomyAddVarR, tmpVarAddr + i * VL_FP32 + dichotomyAddPower);
        Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
        Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

        Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
        ReduceSum(var, sumVar, pregMain);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddAddr + i, var, pregMerge);
    }

    for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount); i++) {
        LoadAlign(dichotomyAddMeanL, tmpMeanAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
        Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
        Mul(deltaL, deltaL, deltaL, pregMain);
        Muls(deltaL, deltaL, cnt, pregMain);
        LoadAlign(dichotomyAddVarL, tmpVarAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
        Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
        Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
        ReduceSum(var, dichotomyAddVarL, pregMain);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddAddr + dichotomyAddReminderLoopCount + i, var, pregMerge);
    }
}

__aicore__ inline void WelfordFinalizeAlignVarPart(__ubuf__ float* tmpMeanAddr, __ubuf__ float* tmpVarAddr,
                                                   __ubuf__ float* dichotomyAddAddr, __ubuf__ float* meanAddr,
                                                   float cnt, float reduceScale, uint32_t dichotomyAddReminder,
                                                   uint16_t dichotomyAddReminderLoopCount,
                                                   uint16_t dichotomyAddPowerLoopCount, uint32_t dichotomyAddPower)
{
    asc_vf_call<WelfordFinalizeAlignVarPartVf>(tmpMeanAddr, tmpVarAddr, dichotomyAddAddr, meanAddr, cnt, reduceScale,
                                               dichotomyAddReminder, dichotomyAddReminderLoopCount,
                                               dichotomyAddPowerLoopCount, dichotomyAddPower);
}

// Mean reduction of VFWelfordParallelFinalizeAlign: dichotomy tree over dichotomyAddAddr, the
// power-of-two rounding correction, and the lane-0 store to meanAddr. Scope-owning phase like the
// parts above; reduction stays inside this VF scope through the residual-order helper.
__simd_vf__ inline void WelfordFinalizeAlignMeanReducePartVf(__ubuf__ float* dichotomyAddAddr, __ubuf__ float* meanAddr,
                                                             uint16_t dichotomyAddK, uint16_t innerLoopCountOrigin,
                                                             uint32_t dichotomyAddLastNum, float scaleCorrection)
{
    RegTensor<float> mean;
    MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    DichotomyAddResidual(mean, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
    Muls(mean, mean, scaleCorrection, pregMerge);
    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanAddr, mean, pregMerge);
}

__aicore__ inline void WelfordFinalizeAlignMeanReducePart(__ubuf__ float* dichotomyAddAddr, __ubuf__ float* meanAddr,
                                                          uint32_t dichotomyAddK, uint16_t innerLoopCountOrigin,
                                                          uint32_t dichotomyAddLastNum, float scaleCorrection)
{
    asc_vf_call<WelfordFinalizeAlignMeanReducePartVf>(dichotomyAddAddr, meanAddr, static_cast<uint16_t>(dichotomyAddK),
                                                      innerLoopCountOrigin, dichotomyAddLastNum, scaleCorrection);
}

// Var reduction of VFWelfordParallelFinalizeAlign: dichotomy tree, correction, Newton-Raphson
// reciprocal-sqrt and the lane-0 store to rstdAddr. Scope-owning like the mean reduce part.
template <bool USE_SQRT_FIRST>
__simd_vf__ inline void WelfordFinalizeAlignVarReducePartVf(__ubuf__ float* dichotomyAddAddr, __ubuf__ float* rstdAddr,
                                                            uint16_t dichotomyAddK, uint16_t innerLoopCountOrigin,
                                                            uint32_t dichotomyAddLastNum, float reduceScaleCorrection,
                                                            float eps)
{
    RegTensor<float> var;
    RegTensor<float> rstd;
    MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    DichotomyAddResidual(var, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
    Muls(var, var, reduceScaleCorrection, pregMerge);
    if constexpr (USE_SQRT_FIRST) {
        ComputeRstdSqrtFirstResidual<false>(var, rstd, pregMerge, eps);
    } else {
        ComputeRstdNewtonRaphsonResidual<false>(var, rstd, pregMerge, eps);
    }
    StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr, rstd, pregMerge);
}

__aicore__ inline void WelfordFinalizeAlignVarReducePart(__ubuf__ float* dichotomyAddAddr, __ubuf__ float* rstdAddr,
                                                         uint32_t dichotomyAddK, uint16_t innerLoopCountOrigin,
                                                         uint32_t dichotomyAddLastNum, float reduceScaleCorrection,
                                                         float eps)
{
    if (eps < QALN_MIN_NORMAL_FP32) {
        asc_vf_call<WelfordFinalizeAlignVarReducePartVf<true>>(
            dichotomyAddAddr, rstdAddr, static_cast<uint16_t>(dichotomyAddK), innerLoopCountOrigin, dichotomyAddLastNum,
            reduceScaleCorrection, eps);
        return;
    }
    asc_vf_call<WelfordFinalizeAlignVarReducePartVf<false>>(dichotomyAddAddr, rstdAddr,
                                                            static_cast<uint16_t>(dichotomyAddK), innerLoopCountOrigin,
                                                            dichotomyAddLastNum, reduceScaleCorrection, eps);
}

// Scalar-domain driver: each vector region is a separate VF launch, mirroring the
// VFCalcMeanVar phase topology and preserving the scalar/vector ownership boundary.
__aicore__ inline void VFWelfordParallelFinalizeAlign(LocalTensor<float>& meanLocal, LocalTensor<float>& rstdLocal,
                                                      LocalTensor<float>& tmpMeanLocal, LocalTensor<float>& tmpVarLocal,
                                                      LocalTensor<float>& dichotomyAddLocal, uint32_t reduceCount,
                                                      uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                      uint32_t dichotomyAddLastNum, uint32_t offset, float reduceScale,
                                                      float reduceScaleCorrection, float scale, float scaleCorrection,
                                                      float cnt, float eps)
{
    __ubuf__ float* meanAddr = (__ubuf__ float*)meanLocal[0].GetPhyAddr();
    __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdLocal[0].GetPhyAddr();
    __ubuf__ float* tmpMeanAddr = (__ubuf__ float*)tmpMeanLocal[0].GetPhyAddr();
    __ubuf__ float* tmpVarAddr = (__ubuf__ float*)tmpVarLocal.GetPhyAddr();
    __ubuf__ float* dichotomyAddAddr = (__ubuf__ float*)dichotomyAddLocal.GetPhyAddr();

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    // Compute the mean.
    WelfordFinalizeAlignMeanPart(tmpMeanAddr, dichotomyAddAddr, scale, dichotomyAddReminder,
                                 dichotomyAddReminderLoopCount, dichotomyAddPowerLoopCount, dichotomyAddPower);
    WelfordFinalizeAlignMeanReducePart(dichotomyAddAddr, meanAddr + offset, dichotomyAddK, innerLoopCountOrigin,
                                       dichotomyAddLastNum, scaleCorrection);

    WelfordFinalizeAlignVarPart(tmpMeanAddr, tmpVarAddr, dichotomyAddAddr, meanAddr + offset, cnt, reduceScale,
                                dichotomyAddReminder, dichotomyAddReminderLoopCount, dichotomyAddPowerLoopCount,
                                dichotomyAddPower);
    WelfordFinalizeAlignVarReducePart(dichotomyAddAddr, rstdAddr + offset, dichotomyAddK, innerLoopCountOrigin,
                                      dichotomyAddLastNum, reduceScaleCorrection, eps);
}

} // namespace NsQuantizeAddLayerNorm

#endif // QUANTIZE_ADD_LAYER_NORM_REGBASE_COMMON_H
