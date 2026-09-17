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
#include "../../add_layer_norm/arch35/add_layer_norm_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace QuantizeAddLayerNormRegbase {
using namespace AddLayerNorm;
using namespace AscendC;
using AscendC::Reg::Truncate;

// OPT_CODE bits: scales is required (always present), zero_points optional
#define OFFSET_CODE 0x1

#define IS_OFFSET_EXIST ((OPT_CODE & OFFSET_CODE) > 0)

// quant mode digit of the regbase tiling key: 0 = mul, 1 = div(per_channel), 2 = per_tensor(scalar mul)
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

// fake class members:
constexpr uint32_t vlFp32_ = GetVecLen() / sizeof(float);
constexpr uint32_t blockSize_ = GetDataBlockSizeInBytes();

constexpr AscendC::Reg::CastTrait quantCastTraitF32ToF16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr AscendC::Reg::CastTrait quantCastTraitF16ToS8 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC,
};

constexpr AscendC::Reg::DivSpecificMode divHighPrecMode = {
    AscendC::Reg::MaskMergeMode::ZEROING,
    true,
};

template <typename T>
__aicore__ inline void LoadQuantParams(__ubuf__ T* ubAddr, RegTensor<float>& dstTensor, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dstTensor, (__ubuf__ float*)ubAddr + offset);
    } else {
        RegTensor<T> dstB16;
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(dstB16, (__ubuf__ T*)ubAddr + offset);
        Cast<float, T, castTraitB162B32>(dstTensor, dstB16, preg);
    }
}

// per_tensor quant: scales/zero_points are single scalars in GM; the kernel copies only
// element 0 into UB, then broadcasts it to the whole register via a DIST_BRC load
// (same mechanism as the mean/rstd broadcast loads).
template <typename T>
__aicore__ inline void LoadScalarQuantParam(__ubuf__ T* ubAddr, RegTensor<float>& dstTensor, MaskReg& preg)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dstTensor, (__ubuf__ float*)ubAddr);
    } else {
        RegTensor<T> dstB16;
        LoadAlign<T, LoadDist::DIST_BRC_B16>(dstB16, (__ubuf__ T*)ubAddr);
        Cast<float, T, castTraitB162B32>(dstTensor, dstB16, preg);
    }
}

__aicore__ inline void Round2Int8(RegTensor<int8_t>& dstTensor, RegTensor<float>& srcTensor, MaskReg& preg)
{
    RegTensor<half> tmpFp16;
    RegTensor<float> tmpFp32;
    Truncate<float, RoundMode::CAST_RINT>(tmpFp32, srcTensor, preg);
    Cast<half, float, quantCastTraitF32ToF16>(tmpFp16, tmpFp32, preg);
    Cast<int8_t, half, quantCastTraitF16ToS8>(dstTensor, tmpFp16, preg);
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

    AddLayerNorm::VFWelfordParallelUpdateCommon<INIT, X1_TYPE, X1_TYPE, X1_TYPE, TILING_KEY>(
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

/*
  Welford Finalize aligned-scenario formulas:
  finalize_mean = sum_fun(mean) / parallel_N
  finalize_delta = mean - finalize_mean
  finalize_delta_square = finalize_delta * finalize_delta
  M2_fixed = M2 + float(count) * finalize_delta_square
  finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)

  welford computes mean and variance with dichotomy accumulation.
  scale / reduceScale are exact power-of-two reciprocals; scaleCorrection / reduceScaleCorrection
  restore the true 1/colsPerLoop / 1/cols factor in one Muls after DichotomyAdd (rounding-error fix).
*/

// Mean half of VFWelfordParallelFinalizeAlign: merge tail+full blocks into dichotomyAddAddr.
// Scope-owning phase function (VFCalcMeanPhase/VFCalcVarPhase pattern in
// quantize_add_layer_norm_regbase_stats.h): registers/masks are function-local and the driver
// exchanges data through UB pointers only. Scope-less register-reference variants of this split
// hung the device on Align shapes regardless of the __aicore__/__simd_callee__ convention
// (real-card cols=12288/16384, 2026-09-16), so it must not take RegTensor/MaskReg references.
__aicore__ inline void WelfordFinalizeAlignMeanPart(__ubuf__ float* tmpMeanAddr, __ubuf__ float* dichotomyAddAddr,
                                                    float scale, uint32_t dichotomyAddReminder,
                                                    uint16_t dichotomyAddReminderLoopCount,
                                                    uint16_t dichotomyAddPowerLoopCount, uint32_t dichotomyAddPower)
{
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t sreg0 = dichotomyAddReminder;
        // PART1: merge tail+full blocks
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL, tmpMeanAddr + i * VL_FP32);
            LoadAlign(dichotomyAddMeanR, tmpMeanAddr + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddAddr + i, mean, pregMerge);
        }

        // PART2: vcadd remaining full blocks back to UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
             i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanAddr + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddAddr + dichotomyAddReminderLoopCount + i, mean, pregMerge);
        }
    }
}

// Var half of VFWelfordParallelFinalizeAlign: M2 fixed by the finalize delta, merged into
// dichotomyAddAddr. Scope-owning like WelfordFinalizeAlignMeanPart above; mean is broadcast
// back from the driver's lane-0 store via a DIST_BRC_B32 load (VFCalcVarPhase mechanism).
__aicore__ inline void WelfordFinalizeAlignVarPart(__ubuf__ float* tmpMeanAddr, __ubuf__ float* tmpVarAddr,
                                                   __ubuf__ float* dichotomyAddAddr, __ubuf__ float* meanAddr,
                                                   float cnt, float reduceScale, uint32_t dichotomyAddReminder,
                                                   uint16_t dichotomyAddReminderLoopCount,
                                                   uint16_t dichotomyAddPowerLoopCount, uint32_t dichotomyAddPower)
{
    __VEC_SCOPE__
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
        uint32_t sreg0 = dichotomyAddReminder;

        LoadAlign<float, LoadDist::DIST_BRC_B32>(mean, meanAddr);

        // PART1: merge tail+full blocks
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
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

        // PART2: vcadd remaining full blocks back to UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
             i++) {
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
}

// Mean reduction of VFWelfordParallelFinalizeAlign: dichotomy tree over dichotomyAddAddr, the
// power-of-two rounding correction, and the lane-0 store to meanAddr. Scope-owning phase like the
// parts above; the scope-less factory helper NormCommon::DichotomyAdd is called from inside this
// scope (family convention, cf. master's monolithic driver).
__aicore__ inline void WelfordFinalizeAlignMeanReducePart(__ubuf__ float* dichotomyAddAddr, __ubuf__ float* meanAddr,
                                                          uint32_t dichotomyAddK, uint16_t innerLoopCountOrigin,
                                                          uint32_t dichotomyAddLastNum, float scaleCorrection)
{
    __VEC_SCOPE__
    {
        RegTensor<float> mean;
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();

        NormCommon::DichotomyAdd(mean, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        Muls(mean, mean, scaleCorrection, pregMerge);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanAddr, mean, pregMerge);
    }
}

// Var reduction of VFWelfordParallelFinalizeAlign: dichotomy tree, correction, Newton-Raphson
// reciprocal-sqrt and the lane-0 store to rstdAddr. Scope-owning like the mean reduce part.
__aicore__ inline void WelfordFinalizeAlignVarReducePart(__ubuf__ float* dichotomyAddAddr, __ubuf__ float* rstdAddr,
                                                         uint32_t dichotomyAddK, uint16_t innerLoopCountOrigin,
                                                         uint32_t dichotomyAddLastNum, float reduceScaleCorrection,
                                                         float eps)
{
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();

        NormCommon::DichotomyAdd(var, dichotomyAddAddr, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        Muls(var, var, reduceScaleCorrection, pregMerge);
        NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr, rstd, pregMerge);
    }
}

// Scalar-domain driver (no __VEC_SCOPE__ of its own): every vector region lives in a scope-owning
// phase function above, mirroring the VFCalcMeanVar -> VFCalcMeanPhase/VFCalcVarPhase topology in
// quantize_add_layer_norm_regbase_stats.h. A driver holding a scope that calls into other
// scope-owning helpers (nested __VEC_SCOPE__) hung the device on Align shapes (real-card
// cols=12288/16384, 2026-09-16) and has no green precedent in the NormCommon family.
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

    // compute mean
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

} // namespace QuantizeAddLayerNormRegbase

#endif // QUANTIZE_ADD_LAYER_NORM_REGBASE_COMMON_H
