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
 * \file add_layer_norm_regbase_common.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_REGBASE_COMMON_H
#define ADD_LAYER_NORM_REGBASE_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace AddLayerNorm {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr int64_t NUM_TWO = 2;
constexpr int32_t VL_FP32 = VECTOR_REG_WIDTH / sizeof(float);

constexpr AscendC::Reg::CastTrait castTraitB322B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

#define IS_BIAS_ELEWISE ((TILING_KEY % 10) == 1)
#define IS_BIAS_BROADCAST ((TILING_KEY % 10) == 2)
#define IS_BIAS_NONE ((TILING_KEY % 10) == 0)

constexpr int DOUBLE_BUFFER_NUM = 2;
constexpr int SINGLE_BUFFER_NUM = 1;

__aicore__ inline int32_t CEIL_DIV(int32_t x, int32_t y) { return (y > 0) ? (x + y - 1) / y : 0; }

__aicore__ inline uint32_t BLOCK_ALIGN(uint32_t x, uint32_t blockSize)
{
    return (blockSize > 0) ? (x + blockSize - 1) / blockSize * blockSize : 0;
}

template <typename T>
__aicore__ inline void CopyLocalToGm(GlobalTensor<T>& dstGm, LocalTensor<T> srcLocal, int64_t offset, int32_t rowsCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = rowsCount * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(dstGm[offset], srcLocal, dataCopyParams);
}

template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE>
__aicore__ inline void LoadInputsToRegWithBias(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                               __ubuf__ BIAS_TYPE* biasAddr, RegTensor<float>& dst, MaskReg& preg,
                                               uint32_t offset0, uint32_t offset1, uint32_t offset2)
{
    RegTensor<float> x1Fp32, x2Fp32, biasFp32;

    if constexpr (IsSameType<X2_TYPE, float>::value) {
        LoadAlign(x2Fp32, (__ubuf__ X2_TYPE*)x2Addr + offset1);
    } else {
        RegTensor<X2_TYPE> x2;
        LoadAlign<X2_TYPE, LoadDist::DIST_UNPACK_B16>(x2, (__ubuf__ X2_TYPE*)x2Addr + offset1);
        Cast<float, X2_TYPE, castTraitB162B32>(x2Fp32, x2, preg);
    }

    if constexpr (IsSameType<BIAS_TYPE, float>::value) {
        LoadAlign(biasFp32, (__ubuf__ BIAS_TYPE*)biasAddr + offset2);
    } else {
        RegTensor<BIAS_TYPE> bias;
        LoadAlign<BIAS_TYPE, LoadDist::DIST_UNPACK_B16>(bias, (__ubuf__ BIAS_TYPE*)biasAddr + offset2);
        Cast<float, BIAS_TYPE, castTraitB162B32>(biasFp32, bias, preg);
    }
    Add<float>(dst, x2Fp32, biasFp32, preg);

    if constexpr (IsSameType<X1_TYPE, float>::value) {
        LoadAlign(x1Fp32, (__ubuf__ X1_TYPE*)x1Addr + offset0);
    } else {
        RegTensor<X1_TYPE> x1;
        LoadAlign<X1_TYPE, LoadDist::DIST_UNPACK_B16>(x1, (__ubuf__ X1_TYPE*)x1Addr + offset0);
        Cast<float, X1_TYPE, castTraitB162B32>(x1Fp32, x1, preg);
    }
    Add<float>(dst, dst, x1Fp32, preg);
}

template <typename X1_TYPE, typename X2_TYPE>
__aicore__ inline void LoadInputsToRegWithBiasNone(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr,
                                                   RegTensor<float>& dst, MaskReg& preg, uint32_t offset0,
                                                   uint32_t offset1)
{
    RegTensor<float> x1Fp32, x2Fp32;

    if constexpr (IsSameType<X1_TYPE, float>::value) {
        LoadAlign(x1Fp32, (__ubuf__ X1_TYPE*)x1Addr + offset0);
    } else {
        RegTensor<X1_TYPE> x1;
        LoadAlign<X1_TYPE, LoadDist::DIST_UNPACK_B16>(x1, (__ubuf__ X1_TYPE*)x1Addr + offset0);
        Cast<float, X1_TYPE, castTraitB162B32>(x1Fp32, x1, preg);
    }

    if constexpr (IsSameType<X2_TYPE, float>::value) {
        LoadAlign(x2Fp32, (__ubuf__ X2_TYPE*)x2Addr + offset1);
    } else {
        RegTensor<X2_TYPE> x2;
        LoadAlign<X2_TYPE, LoadDist::DIST_UNPACK_B16>(x2, (__ubuf__ X2_TYPE*)x2Addr + offset1);
        Cast<float, X2_TYPE, castTraitB162B32>(x2Fp32, x2, preg);
    }
    Add<float>(dst, x1Fp32, x2Fp32, preg);
}

template <typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE, int TILING_KEY>
__aicore__ inline void LoadInputsToReg(__ubuf__ X1_TYPE* x1Addr, __ubuf__ X2_TYPE* x2Addr, __ubuf__ BIAS_TYPE* biasAddr,
                                       RegTensor<float>& dst, MaskReg& preg, uint32_t offset0, uint32_t offset1,
                                       uint32_t offset2)
{
    if constexpr (IS_BIAS_NONE) {
        LoadInputsToRegWithBiasNone(x1Addr, x2Addr, dst, preg, offset0, offset1);
    } else {
        LoadInputsToRegWithBias(x1Addr, x2Addr, biasAddr, dst, preg, offset0, offset1, offset2);
    }
}

template <typename T>
__aicore__ inline void LoadGammaBeta(__ubuf__ T* gammaAddr, __ubuf__ T* betaAddr, RegTensor<float>& gamma,
                                     RegTensor<float>& beta, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> gammaFp16;
        RegTensor<half> betaFp16;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(gammaFp16, (__ubuf__ half*)gammaAddr + offset);
        Cast<float, half, castTraitB162B32>(gamma, gammaFp16, preg);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(betaFp16, (__ubuf__ half*)betaAddr + offset);
        Cast<float, half, castTraitB162B32>(beta, betaFp16, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> gammaBf16;
        RegTensor<bfloat16_t> betaBf16;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(gammaBf16, (__ubuf__ bfloat16_t*)gammaAddr + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(gamma, gammaBf16, preg);
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(betaBf16, (__ubuf__ bfloat16_t*)betaAddr + offset);
        Cast<float, bfloat16_t, castTraitB162B32>(beta, betaBf16, preg);
    } else {
        LoadAlign(gamma, (__ubuf__ float*)gammaAddr + offset);
        LoadAlign(beta, (__ubuf__ float*)betaAddr + offset);
    }
}

template <bool IS_MIX, typename GAMMA_TYPE, typename BETA_TYPE, int32_t BUFFER_NUM>
__aicore__ inline void CopyGammaAndBetaToUBCommon(LocalTensor<GAMMA_TYPE> gammaLocal, LocalTensor<BETA_TYPE> betaLocal,
                                                  GlobalTensor<GAMMA_TYPE>& gammaGm, GlobalTensor<BETA_TYPE>& betaGm,
                                                  TQue<QuePosition::VECIN, BUFFER_NUM>& gammaQueue,
                                                  TQue<QuePosition::VECIN, BUFFER_NUM>& betaQueue, int64_t offset,
                                                  int32_t copyLen, uint32_t blockSize)
{
    if constexpr (IS_MIX) {
        int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(float), blockSize) / sizeof(float);
        DataCopyPadExtParams<float> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<float>(0.0);
        padParams.rightPadding = copyLenAlign - copyLen;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(betaLocal, betaGm[offset], dataCopyParams, padParams);
        betaQueue.EnQue(betaLocal);
        DataCopyPad(gammaLocal, gammaGm[offset], dataCopyParams, padParams);
        gammaQueue.EnQue(gammaLocal);
    } else {
        int32_t copyLenAlign = BLOCK_ALIGN(copyLen * sizeof(GAMMA_TYPE), blockSize) / sizeof(GAMMA_TYPE);
        DataCopyPadExtParams<GAMMA_TYPE> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<GAMMA_TYPE>(0.0);
        padParams.rightPadding = copyLenAlign - copyLen;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = copyLen * sizeof(GAMMA_TYPE);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(betaLocal, betaGm[offset], dataCopyParams, padParams);
        betaQueue.EnQue(betaLocal);
        DataCopyPad(gammaLocal, gammaGm[offset], dataCopyParams, padParams);
        gammaQueue.EnQue(gammaLocal);
    }
}

template <typename T>
__aicore__ inline void StoreRegToOutput(__ubuf__ T* dstAddr, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> dst;
        Cast<half, float, castTraitB322B16>(dst, src, preg);
        StoreAlign<half, StoreDist::DIST_PACK_B32>((__ubuf__ half*)dstAddr + offset, dst, preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> dst;
        Cast<bfloat16_t, float, castTraitB322B16>(dst, src, preg);
        StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>((__ubuf__ bfloat16_t*)dstAddr + offset, dst, preg);
    } else {
        StoreAlign((__ubuf__ float*)dstAddr + offset, src, preg);
    }
}

template <bool INIT, typename X1_TYPE, typename X2_TYPE, typename BIAS_TYPE, int TILING_KEY>
__aicore__ inline void VFWelfordParallelUpdateCommon(__ubuf__ X1_TYPE* x1Local, __ubuf__ X2_TYPE* x2Local,
                                                     __ubuf__ BIAS_TYPE* biasLocal, __ubuf__ BIAS_TYPE* xOutLocal,
                                                     __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                     uint64_t calLen, uint16_t loopCount, float scale)
{
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delat4;
        RegTensor<float> delta3;
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputsToReg<X1_TYPE, X2_TYPE, BIAS_TYPE, TILING_KEY>(x1Local, x2Local, biasLocal, x1, pregLoop,
                                                                     i * VL_FP32, i * VL_FP32, i * VL_FP32);
            StoreRegToOutput(xOutLocal, x1, pregLoop, i * VL_FP32);
            if constexpr (INIT) {
                Duplicate(tmpMean, 0.0, pregLoop);
            } else {
                LoadAlign(tmpMean, tmpMeanLocal + i * VL_FP32);
            }
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            StoreAlign(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

            if constexpr (INIT) {
                Duplicate(tmpVar, 0.0, pregLoop);
            } else {
                LoadAlign(tmpVar, tmpVarLocal + i * VL_FP32);
            }
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delat4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delat4, pregLoop);
            StoreAlign(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

/*
  // Welford Finalize非对齐场景计算公式如下:
  counts = np.ones(len(mean), dtype=np.float32)*count
  for i in range(tail_size):
      counts[i] += 1
  finalize_mean = np.sum(mean*counts) / np.sum(counts)
  finalize_delta = mean - finalize_mean
  finalize_delta_square = finalize_delta * finalize_delta
  M2_fixed = M2 + counts * finalize_delta_square
  finalize_std = np.sum(M2_fixed) / np.sum(counts)

  // Welford Finalize非对齐场景下，二分累加存在如下几种场景:
  首先,非对齐场景下存在两种尾块类型
  1. welford自身的整块和尾块，需要注意的是，welford自身的整块也可能非对齐，整块+尾块=并行度
  2. 二分累加的整块和尾块
  3.
  3.1 welford整块小于二分累加整块，这种场景下，可以进一步细分为两种场景
  3.1.1 welford整块小于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
  3.1.2 welford整块大于等于二分累加尾块向上对齐，那么welford整块处理逻辑就需要体现剩余整块回刷UB逻辑中
  3.2 welford整块大于等于二分累加整块，那么welford整块处理逻辑就需要体现在二分累加整尾块折叠逻辑中
*/

// welford整块大于等于二分累加整块
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation1(
    __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
    __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
    uint32_t dichotomyAddLastNum, uint32_t offset, uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float coeff = tailCnt / cnt;
    float tailCountScale = tailCnt * reduceScale;
    float countScale = cnt * reduceScale;

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderRealLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 -
                                              welfordDiffReminderAlign;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminderAfterSplit, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> one;
        RegTensor<float> rstd;
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t sreg0;

        // 整块使用tailCountScale,尾块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailCountScale, pregMain);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean, sumMean, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
        }

        // 处理welford第一次非对齐点, 整块使用tailCountScale,尾块部分使用tailCountScale, 部分使用countScale
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        uint32_t sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Muls(tmp, dichotomyAddMeanR, coeff, pregLoop1);
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean, sumMean, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
        }

        // 整块使用tailCountScale,尾块使用countScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL,
                      tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                             welfordDiffReminderAlign + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean, sumMean, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(mean, dichotomyAddMeanL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
        }
        NormCommon::DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
            Mul(deltaR, deltaR, deltaR, pregMain);
            Muls(deltaR, deltaR, tailCnt, pregMain);

            LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            LoadAlign(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(var, sumVar, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
        }
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            Muls(tmp, deltaR, coeff, pregLoop1);
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

            LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(var, sumVar, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL,
                      tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                             welfordDiffReminderAlign + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);

            LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign +
                                            dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(var, sumVar, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(var, dichotomyAddVarL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
        }
        NormCommon::DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

// welford整块小于二分累加整块，并且小于等于二分累加尾块向上对齐
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation2(
    __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
    __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
    uint32_t dichotomyAddLastNum, uint32_t offset, uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float countScale = cnt * reduceScale;
    float tailCnt = cnt + float(1.0);
    float tailCountScale = tailCnt * reduceScale;
    float coeff = tailCnt / cnt;

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = tailSize / VL_FP32;
    uint32_t welfordDiffReminder = tailSize - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint16_t dichotomyAddReminderRealLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 -
                                              welfordDiffReminderAlign;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminderAfterSplit, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMeanReg;
        RegTensor<float> meanReg;
        RegTensor<float> sumVarReg;
        RegTensor<float> varReg;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> oneReg;
        RegTensor<float> rstdReg;
        RegTensor<float> tmpReg;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t sreg0;
        uint32_t sreg1;

        // 整块使用tailCountScale,尾块使用countScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregMain);
            Add(sumMeanReg, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, sumMeanReg, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, meanReg,
                                                                               pregMerge);
        }

        // 处理welford第一次非对齐点, 尾块使用countScale,整块部分使用tailCountScale, 部分使用countScale
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Muls(tmpReg, dichotomyAddMeanL, coeff, pregLoop1);
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmpReg, pregLoop1);
            Add(sumMeanReg, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, sumMeanReg, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, meanReg, pregMerge);
        }

        // 整块使用countScale,尾块使用countScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL,
                      tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                             welfordDiffReminderAlign + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMeanReg, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, sumMeanReg, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, meanReg, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用countScale
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, dichotomyAddMeanL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, meanReg, pregMerge);
        }
        NormCommon::DichotomyAdd(meanReg, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, meanReg, pregMerge);

        // 计算rstd
        Duplicate(oneReg, float(1.0), pregMain);
        Duplicate(meanReg, meanReg, pregMain);
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, meanReg, pregMain);
            Mul(deltaR, deltaR, deltaR, pregMain);
            Muls(deltaR, deltaR, cnt, pregMain);

            LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            LoadAlign(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

            Add(sumVarReg, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, sumVarReg, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, varReg,
                                                                               pregMerge);
        }
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, meanReg, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            Muls(tmpReg, deltaL, coeff, pregLoop1);
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(deltaL, tmpReg, pregLoop1);

            LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVarReg, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, sumVarReg, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, varReg, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            MaskReg pregMask = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL,
                      tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 +
                                             welfordDiffReminderAlign + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, meanReg, pregMask);
            Mul(deltaR, deltaR, deltaR, pregMask);
            Muls(deltaR, deltaR, cnt, pregMask);

            LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            LoadAlign(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign +
                                            dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMask);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMask);
            Add(sumVarReg, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, sumVarReg, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, varReg, pregMerge);
        }

        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             ++i) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            LoadAlign(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, dichotomyAddVarL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, varReg, pregMerge);
        }
        NormCommon::DichotomyAdd(varReg, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        NormCommon::ComputeRstdNewtonRaphsonReg<false>(varReg, rstdReg, pregMerge, eps);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstdReg, pregMerge);
    }
}

// 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
    __ubuf__ float* meanLocal, __ubuf__ float* rstdLocal, __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
    __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
    uint32_t dichotomyAddLastNum, uint32_t offset, uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float countScale = cnt * reduceScale;
    float tailCountScale = tailCnt * reduceScale;
    float coeff = tailCnt / cnt;

    // 二分累加
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CEIL_DIV(dichotomyAddReminder, VL_FP32);
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
    uint32_t dichotomyAddReminderRoundUp = dichotomyAddReminderLoopCount * VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddReminderRoundUp;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCount = dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount -
                                                welfordDiffLoopCount - welfordReminderLoopCount;
    uint32_t dichotomyAddPowerOffset = dichotomyAddReminderRoundUp + welfordDiffLoopCount * VL_FP32 +
                                       welfordDiffReminderAlign;

    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanL;
        RegTensor<float> dichotomyAddMeanR;
        RegTensor<float> dichotomyAddVarL;
        RegTensor<float> dichotomyAddVarR;
        RegTensor<float> sumMean;
        RegTensor<float> meanReg;
        RegTensor<float> sumVar;
        RegTensor<float> varReg;
        RegTensor<float> deltaL;
        RegTensor<float> deltaR;
        RegTensor<float> one;
        RegTensor<float> rstd;
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        uint32_t sreg0 = dichotomyAddReminder;
        // 整块使用tailCountScale, 尾块使用CountScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, sumMean, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, meanReg,
                                                                               pregMerge);
        }

        // 剩余整块需要拆分成多部分
        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, dichotomyAddMeanL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, meanReg, pregMerge);
        }

        sreg0 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL,
                      tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, dichotomyAddMeanL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, meanReg, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(meanReg, dichotomyAddMeanL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
                meanReg, pregMerge);
        }

        NormCommon::DichotomyAdd(meanReg, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, meanReg, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(meanReg, meanReg, pregMain);
        // 整块使用tailCountScale, 尾块使用CountScale
        sreg0 = dichotomyAddReminder;
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

            LoadAlign(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, meanReg, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            LoadAlign(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, sumVar, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, varReg,
                                                                               pregMerge);
        }

        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, dichotomyAddVarL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, varReg, pregMerge);
        }

        sreg0 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadAlign(dichotomyAddMeanL,
                      tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            Muls(tmp, deltaL, coeff, pregLoop);
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
            LoadAlign(dichotomyAddVarL,
                      tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, dichotomyAddVarL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, varReg, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
            LoadAlign(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Sub(deltaL, dichotomyAddMeanL, meanReg, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            LoadAlign(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            Reduce<AscendC::Reg::ReduceType::SUM>(varReg, dichotomyAddVarL, pregMain);
            StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
                varReg, pregMerge);
        }

        NormCommon::DichotomyAdd(varReg, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        NormCommon::ComputeRstdNewtonRaphsonReg<false>(varReg, rstd, pregMerge, eps);
        StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

__aicore__ inline void VFWelfordParallelFinalizeNonAlign(__ubuf__ float* meanLocal, __ubuf__ float* rstdLocal,
                                                         __ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                                         __ubuf__ float* dichotomyAddLocal, uint32_t reduceCount,
                                                         uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                         uint32_t dichotomyAddLastNum, uint32_t offset,
                                                         uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint32_t dichotomyAddReminderRoundUp = CEIL_DIV(dichotomyAddReminder, VL_FP32) * VL_FP32;
    if (tailSize >= dichotomyAddPower) {
        VFWelfordParallelFinalizeNonAlignSituation1(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal,
                                                    reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum,
                                                    offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    if (tailSize <= dichotomyAddReminderRoundUp) {
        VFWelfordParallelFinalizeNonAlignSituation2(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal,
                                                    reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum,
                                                    offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    VFWelfordParallelFinalizeNonAlignSituation3(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal,
                                                reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum,
                                                offset, tailSize, reduceScale, cnt, eps);
}

} // namespace AddLayerNorm
#endif // ADD_LAYER_NORM_REGBASE_COMMON_H
