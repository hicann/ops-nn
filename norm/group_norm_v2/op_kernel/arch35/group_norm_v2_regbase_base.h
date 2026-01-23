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
 * \file group_norm_v2_regbase_base.h
 * \brief
 */
#ifndef GROUP_NORM_V2_REGBASE_BASE_H_
#define GROUP_NORM_V2_REGBASE_BASE_H_
#include "kernel_operator.h"
namespace GroupNormV2 {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
static constexpr int32_t BLOCK_SIZE = 32;
static constexpr int32_t FOUR_BUF = 4;
static constexpr int32_t FP32_ONE_REPEAT = 64;
static constexpr int32_t FLOAT_BYTE_SIZE = 4;
static constexpr int32_t FOUR_UNROLL = 4;
static constexpr int32_t HALF = 2;
static constexpr int32_t INDEX_0 = 0;
static constexpr int32_t INDEX_1 = 1;
static constexpr int32_t INDEX_2 = 2;
static constexpr int32_t INDEX_3 = 3;
static constexpr int32_t BASIC_NUM = 1024;
static constexpr int32_t GROUP_NUM = 8;
static constexpr int32_t MAX_ONCE_NUM_PER_CORE = 2048;
static constexpr int32_t DICHOTOMY_ADD_COEFF = 2;
static constexpr int32_t VL_FP32 = VECTOR_REG_WIDTH / sizeof(float);
static constexpr float SCALAR1 = -0.5;
static constexpr float SCALAR2 = 1.5;
static constexpr float SCALAR3 = 0.5;
static constexpr float POS_INF = 3.40282366920938E+38;
static constexpr float ZERO = 0.0f;
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32Even = {
    AscendC::MicroAPI::RegLayout::ZERO, // even
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32Odd = {
    AscendC::MicroAPI::RegLayout::ONE, // odd
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16Odd = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

template <typename T>
__aicore__ inline uint32_t RoundUp(uint32_t x)
{
    uint32_t elemNum = BLOCK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T>
__aicore__ inline uint16_t GetVLSize()
{
    return VECTOR_REG_WIDTH / sizeof(T);
}

template <typename T>
__aicore__ inline uint32_t RoundDown(uint32_t x)
{
    uint32_t elemNum = BLOCK_SIZE / sizeof(T);
    return (x / elemNum) * elemNum;
}

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadGammaAndBetaData(
    RegTensor<float>& gamma, RegTensor<float>& beta, __local_mem__ T* gammaLocal, __local_mem__ T* betaLocal,
    MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(gamma, gammaLocal + srcOffset);
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(beta, betaLocal + srcOffset);
    } else {
        RegTensor<T> gammaB16;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(gammaB16, gammaLocal + srcOffset);
        Cast<float, T, castTraitB162B32Even>(gamma, gammaB16, pregLoop);
        RegTensor<T> betaB16;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(betaB16, betaLocal + srcOffset);
        Cast<float, T, castTraitB162B32Even>(beta, betaB16, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOutputData(
    __local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else {
        RegTensor<T> tmpB16;
        Cast<T, float, castTraitB322B16Even>(tmpB16, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmpB16, pregLoop);
    }
}

__aicore__ inline void DichotomyAdd(
    RegTensor<float>& dstReg, __local_mem__ float* src, uint16_t outerLoop, uint16_t innerLoop, uint32_t lastNum)
{
    RegTensor<float> tmpReg1;
    RegTensor<float> tmpReg2;
    RegTensor<float> tmpReg3;
    LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t k = 0; k < outerLoop; k++) {
        innerLoop = innerLoop / DICHOTOMY_ADD_COEFF;
        for (uint16_t i = 0; i < innerLoop; i++) {
            DataCopy(tmpReg1, src + i * VL_FP32);
            DataCopy(tmpReg2, src + (i + innerLoop) * VL_FP32);
            Add(tmpReg3, tmpReg1, tmpReg2, pregMain);
            DataCopy(src + i * VL_FP32, tmpReg3, pregMain);
        }
        LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    }
    uint32_t sreg0 = lastNum;
    MaskReg pregLoop = UpdateMask<float>(sreg0);
    DataCopy(tmpReg3, src);
    ReduceSum(dstReg, tmpReg3, pregLoop);
}

__aicore__ inline void CalRstdByHighPrecision(RegTensor<float>& var, RegTensor<float>& rstd, float epsilon)
{
    RegTensor<float> r;
    RegTensor<float> y;
    RegTensor<float> s;
    RegTensor<float> t;
    RegTensor<float> e;
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> scalar2;
    RegTensor<float> scalar3;
    RegTensor<float> t1;
    RegTensor<float> t2;
    RegTensor<float> t3;
    RegTensor<float> t4;
    MaskReg cmpReg1;
    MaskReg cmpReg2;
    MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();

    Duplicate(one, float(1.0), pregMerge);
    Duplicate(scalar1, SCALAR3, pregMerge);
    Duplicate(scalar2, POS_INF, pregMerge);
    Duplicate(scalar3, ZERO, pregMerge);
    Duplicate(t1, SCALAR2, pregMerge);
    Duplicate(s, float(1.0), pregMerge);

    Adds(var, var, epsilon, pregMerge);
    Div(r, one, var, pregMerge);
    Sqrt(y, r, pregMerge);
    Muls(t, var, SCALAR1, pregMerge);
    Mul(t, t, y, pregMerge);                // -0.5 * x * y
    Mula(t1, t, y, pregMerge);              // 1.5 + (-0.5 * x * y) * y
    Mul(rstd, y, t1, pregMerge);            // y = y * (1.5 - 0.5 * x * y)
    Muls(t3, var, float(-1.0), pregMerge);  // -1 * x
    Mula(s, t3, r, pregMerge);              // 1 + (-1) * x * r
    Muls(t4, rstd, float(-1.0), pregMerge); // (-1) * y
    Mula(r, t4, rstd, pregMerge);           // r + (-1) * y * y
    Mula(s, var, r, pregMerge);             // s + x * t
    Mul(s, s, rstd, pregMerge);             // e * y
    Mula(rstd, s, scalar1, pregMerge);      // y + y * e * 0.5
    CompareScalar(cmpReg1, var, POS_INF, pregMerge);
    Select(rstd, scalar3, rstd, cmpReg1);
    CompareScalar(cmpReg2, var, ZERO, pregMerge);
    Select(rstd, scalar2, rstd, cmpReg2);
}

template <typename T>
__aicore__ inline void VFInnerWelfordParallelUpdateWithInit(
    __local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, uint64_t calLen,
    float scale)
{
    uint16_t loopCount = CeilDiv(calLen, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delta3;
        RegTensor<float> delat4;
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputData<T>(x1, x1Local, pregLoop, i * VL_FP32);
            Duplicate(tmpMean, 0.0, pregLoop);
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            DataCopy(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

            Duplicate(tmpVar, 0.0, pregLoop);
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delat4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delat4, pregLoop);
            DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

/*
  Welford update 阶段计算公式如下:
  count += 1
  delta = new_value - mean
  mean += (delta / count)
  delta2 = new_value - mean
  var += delta * delta2
  return count, mean, var
*/
template <typename T>
__aicore__ inline void VFInnerWelfordParallelUpdate(
    __local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, uint64_t calLen,
    float scale)
{
    uint16_t loopCount = CeilDiv(calLen, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delta3;
        RegTensor<float> delat4;
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputData<T>(x1, x1Local, pregLoop, i * VL_FP32);
            DataCopy(tmpMean, tmpMeanLocal + i * VL_FP32);
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            DataCopy(tmpMeanLocal + i * VL_FP32, tmpMean, pregLoop);

            DataCopy(tmpVar, tmpVarLocal + i * VL_FP32);
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delat4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delat4, pregLoop);
            DataCopy(tmpVarLocal + i * VL_FP32, tmpVar, pregLoop);
        }
    }
}

template <typename T>
__aicore__ inline void VFWelfordParallelUpdate(
    __local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, uint64_t curLoop,
    uint64_t calLen, float scale)
{
    if (curLoop == 0) {
        VFInnerWelfordParallelUpdateWithInit(x1Local, tmpMeanLocal, tmpVarLocal, calLen, scale);
    } else {
        VFInnerWelfordParallelUpdate(x1Local, tmpMeanLocal, tmpVarLocal, calLen, scale);
    }
}

/*
  Welford Finalize对齐场景计算公式如下:
  finalize_mean = sum_fun(mean) / parallel_N
  finalize_delta = mean - finalize_mean
  finalize_delta_square = finalize_delta * finalize_delta
  M2_fixed = M2 + float(count) * finalize_delta_square
  finalize_std = sum_fun(M2_fixed) / float(parallel_N * count)

  welford采用二分累加计算mean和variance, 基本逻辑为:
  先将尾块折叠到整块上，整尾块vadd之后，做一次vcadd回刷到UB上，剩余整块直接vcadd回刷到UB上，最后对UB上的结果做完全二分对折
*/
__aicore__ inline void VFWelfordParallelFinalizeAlign(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    float reduceScale, float scale, float cnt, float eps)
{
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
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
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0 = dichotomyAddReminder;
        // 计算mean
        // PART1: 整尾块合并
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, scale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, mean, pregMerge);
        }

        // PART2: 整块剩余部分vcadd回刷UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
        }

        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        sreg0 = dichotomyAddReminder;
        // PART1: 整尾块合并
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, var, pregMerge);
        }

        // PART2: 整块剩余部分vcadd回刷UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
        }

        DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
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
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float coeff = tailCnt / cnt;
    float tailCountScale = tailCnt * reduceScale;
    float countScale = cnt * reduceScale;

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderRealLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit =
        dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign;
    uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminderAfterSplit, VL_FP32);
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
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0;

        // 整块使用tailCountScale,尾块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailCountScale, pregMain);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, mean, pregMerge);
        }

        // 处理welford第一次非对齐点, 整块使用tailCountScale,尾块部分使用tailCountScale, 部分使用countScale
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        uint32_t sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Muls(tmp, dichotomyAddMeanR, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
        }

        // 整块使用tailCountScale,尾块使用countScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
        }
        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
            Mul(deltaR, deltaR, deltaR, pregMain);
            Muls(deltaR, deltaR, tailCnt, pregMain);

            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, var, pregMerge);
        }
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            Muls(tmp, deltaR, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(
                dichotomyAddVarR,
                tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
        }
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
        }
        DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

// welford整块小于二分累加整块，并且小于等于二分累加尾块向上对齐
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation2(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float coeff = tailCnt / cnt;
    float tailCountScale = tailCnt * reduceScale;
    float countScale = cnt * reduceScale;

    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = tailSize / VL_FP32;
    uint32_t welfordDiffReminder = tailSize - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint16_t dichotomyAddReminderRealLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit =
        dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 - welfordDiffReminderAlign;
    uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminderAfterSplit, VL_FP32);
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
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0;
        uint32_t sreg1;

        // 整块使用tailCountScale,尾块使用countScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregMain);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, mean, pregMerge);
        }

        // 处理welford第一次非对齐点, 尾块使用countScale,整块部分使用tailCountScale, 部分使用countScale
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Muls(tmp, dichotomyAddMeanL, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop1);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
        }

        // 整块使用countScale,尾块使用countScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用countScale
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
        }
        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregMain);
            Mul(deltaR, deltaR, deltaR, pregMain);
            Muls(deltaR, deltaR, cnt, pregMain);

            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregMain);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregMain);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, var, pregMerge);
        }
        sreg0 = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32;
        sreg1 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            pregLoop1 = UpdateMask<float>(sreg1);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            Muls(tmp, deltaL, coeff, pregLoop1);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop1);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(
                dichotomyAddMeanR,
                tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(
                dichotomyAddVarR,
                tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
        }
        DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

// 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCnt = cnt + float(1.0);
    float coeff = tailCnt / cnt;
    float tailCountScale = tailCnt * reduceScale;
    float countScale = cnt * reduceScale;

    // 二分累加
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
    uint32_t dichotomyAddReminderRoundUp = dichotomyAddReminderLoopCount * VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddReminderRoundUp;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCount =
        dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount - welfordDiffLoopCount - welfordReminderLoopCount;
    uint32_t dichotomyAddPowerOffset =
        dichotomyAddReminderRoundUp + welfordDiffLoopCount * VL_FP32 + welfordDiffReminderAlign;

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
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0 = dichotomyAddReminder;
        // 整块使用tailCountScale, 尾块使用CountScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, mean, pregMerge);
        }

        // 剩余整块需要拆分成多部分
        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
        }

        sreg0 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(
                dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, mean, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
                mean, pregMerge);
        }

        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
        Duplicate(one, float(1.0), pregMain);
        Duplicate(mean, mean, pregMain);
        // 整块使用tailCountScale, 尾块使用CountScale
        sreg0 = dichotomyAddReminder;
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);

            DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);
            DataCopy(dichotomyAddVarR, tmpVarLocal + i * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i, var, pregMerge);
        }

        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, tailCnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
        }

        sreg0 = welfordDiffReminder;
        for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(
                dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            Muls(tmp, deltaL, coeff, pregLoop);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
            DataCopy(
                dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
            DataCopy(dichotomyAddVarL, tmpVarLocal + i * VL_FP32 + dichotomyAddPowerOffset);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            ReduceSum(var, dichotomyAddVarL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
                var, pregMerge);
        }

        DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        CalRstdByHighPrecision(var, rstd, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

__aicore__ inline void VFWelfordParallelFinalizeNonAlign(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint32_t dichotomyAddReminderRoundUp = CeilDiv(dichotomyAddReminder, VL_FP32) * VL_FP32;
    if (tailSize >= dichotomyAddPower) {
        VFWelfordParallelFinalizeNonAlignSituation1(
            meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    if (tailSize <= dichotomyAddReminderRoundUp) {
        VFWelfordParallelFinalizeNonAlignSituation2(
            meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    VFWelfordParallelFinalizeNonAlignSituation3(
        meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
        dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
}

__aicore__ inline void VFWelfordParallelFinalize(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float scale, float cnt, float eps, bool welfordAlign)
{
    if (welfordAlign) {
        VFWelfordParallelFinalizeAlign(
            meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, offset, reduceScale, scale, cnt, eps);
    } else {
        cnt = cnt - 1;
        VFWelfordParallelFinalizeNonAlign(
            meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount, dichotomyAddPower,
            dichotomyAddK, dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
    }
}

template <typename T>
__aicore__ inline void CalMeanAndRstdByDichotomyAdd(
    __local_mem__ T* xLocal, __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
    __local_mem__ float* dichotomyAddLocal, uint16_t numPerCoreProcess, uint32_t dichotomyAddPower,
    uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint64_t reduceCount, float scale, float eps)
{
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
    uint32_t elemNumAlign = RoundUp<T>(reduceCount);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddL;
        RegTensor<float> dichotomyAddR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
        RegTensor<float> one;
        RegTensor<float> rstd;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0;
        for (uint16_t i = 0; i < numPerCoreProcess; i++) {
            // 计算mean
            sreg0 = dichotomyAddReminder;
            for (uint16_t j = 0; j < dichotomyAddReminderLoopCount; j++) {
                pregLoop = plt_b32(sreg0, POST_UPDATE);
                LoadInputData<T>(dichotomyAddL, xLocal, pregMain, i * elemNumAlign + j * VL_FP32);
                LoadInputData<T>(dichotomyAddR, xLocal, pregLoop, i * elemNumAlign + j * VL_FP32 + dichotomyAddPower);
                Muls(dichotomyAddL, dichotomyAddL, scale, pregMain);
                Muls(dichotomyAddR, dichotomyAddR, scale, pregLoop);
                Add(sumMean, dichotomyAddL, dichotomyAddR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + j, mean, pregMerge);
            }

            // 整块剩余部分vcadd回刷UB
            for (uint16_t j = 0; j < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 j++) {
                LoadInputData<T>(
                    dichotomyAddL, xLocal, pregMain, i * elemNumAlign + (j + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddL, dichotomyAddL, scale, pregMain);
                ReduceSum(mean, dichotomyAddL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + j, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + i, mean, pregMerge);
            // 计算rstd
            Duplicate(one, float(1.0), pregMain);
            Duplicate(mean, mean, pregMain);
            sreg0 = dichotomyAddReminder;
            for (uint16_t j = 0; j < dichotomyAddReminderLoopCount; j++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputData<T>(dichotomyAddL, xLocal, pregMain, i * elemNumAlign + j * VL_FP32);
                LoadInputData<T>(dichotomyAddR, xLocal, pregLoop, i * elemNumAlign + j * VL_FP32 + dichotomyAddPower);
                Sub(dichotomyAddL, dichotomyAddL, mean, pregMain);
                Sub(dichotomyAddR, dichotomyAddR, mean, pregLoop);
                Mul(dichotomyAddL, dichotomyAddL, dichotomyAddL, pregMain);
                Mul(dichotomyAddR, dichotomyAddR, dichotomyAddR, pregLoop);
                Muls(dichotomyAddL, dichotomyAddL, scale, pregMain);
                Muls(dichotomyAddR, dichotomyAddR, scale, pregLoop);
                Add(sumVar, dichotomyAddL, dichotomyAddR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + j, var, pregMerge);
            }

            // 整块剩余部分vcadd回刷UB
            for (uint16_t j = 0; j < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 j++) {
                LoadInputData<T>(
                    dichotomyAddL, xLocal, pregMain, i * elemNumAlign + (j + dichotomyAddReminderLoopCount) * VL_FP32);
                Sub(dichotomyAddL, dichotomyAddL, mean, pregMain);
                Mul(dichotomyAddL, dichotomyAddL, dichotomyAddL, pregMain);
                Muls(dichotomyAddL, dichotomyAddL, scale, pregMain);
                ReduceSum(var, dichotomyAddL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + j, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + i, rstd, pregMerge);
        }
    }
}

// R轴小于64
template <typename T>
__aicore__ inline void CalMeanAndRstdSpecial(
    __local_mem__ T* xLocal, __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, uint16_t numPerCoreProcess,
    uint64_t reduceCount, float scale, float eps)
{
    uint32_t elemNumAlign = RoundUp<T>(reduceCount);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xScale;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> rstd;
        RegTensor<float> one;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        Duplicate(one, float(1.0), pregMain);
        for (uint16_t i = 0; i < numPerCoreProcess; i++) {
            uint32_t sreg0 = reduceCount;
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputData<T>(x, xLocal, pregLoop, i * elemNumAlign);
            Muls(xScale, x, scale, pregLoop);
            ReduceSum(mean, xScale, pregLoop);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + i, mean, pregMerge);

            Duplicate(mean, mean, pregMain);
            Sub(x, x, mean, pregLoop);
            Mul(x, x, x, pregLoop);
            Muls(xScale, x, scale, pregLoop);
            ReduceSum(var, xScale, pregLoop);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + i, rstd, pregMerge);
        }
    }
}

template <typename T>
__aicore__ inline void CalMeanAndRstd(
    __local_mem__ T* xLocal, __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
    __local_mem__ float* dichotomyAddLocal, uint16_t numPerCoreProcess, uint32_t dichotomyAddPower,
    uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint64_t reduceCount, float scale, float eps)
{
    if (dichotomyAddPower >= VL_FP32) {
        CalMeanAndRstdByDichotomyAdd(
            xLocal, meanLocal, rstdLocal, dichotomyAddLocal, numPerCoreProcess, dichotomyAddPower, dichotomyAddK,
            dichotomyAddLastNum, reduceCount, scale, eps);
        return;
    }
    CalMeanAndRstdSpecial(xLocal, meanLocal, rstdLocal, numPerCoreProcess, reduceCount, scale, eps);
}

__aicore__ inline void VFInnerNormalize(
    RegTensor<float>& x, RegTensor<float>& mean, RegTensor<float>& rstd, RegTensor<float>& gamma,
    RegTensor<float>& beta, RegTensor<float>& y, MaskReg& preg)
{
    Sub(x, x, mean, preg);
    Mul(x, x, rstd, preg);
    Mul(x, x, gamma, preg);
    Add(y, x, beta, preg);
}

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeUnAlign(
    __local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
    __local_mem__ float* rstdLocal, __local_mem__ T1* yLocal, uint32_t rowsCount, int32_t reduceCount)
{
    uint16_t VL = GetVLSize<T1>();
    uint16_t loopCount = reduceCount / VL;
    uint16_t tailNum = reduceCount - loopCount * VL;
    uint16_t tailLoop = CeilDiv(tailNum, VL);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xOdd;
        RegTensor<float> xEven;
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> rstd;
        RegTensor<float> y;
        RegTensor<float> yOdd;
        RegTensor<float> yEven;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<T1, AscendC::MicroAPI::MaskPattern::ALL>();

        UnalignReg uSrc;
        UnalignReg uDst;
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstd, rstdLocal);
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(mean, meanLocal);
        DataCopyUnAlignPre<T1>(uSrc, xLocal);
        for (uint16_t i = 0; i < static_cast<uint16_t>(rowsCount); i++) {
            LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain, i);
            if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
                RegTensor<T1> xTmp;
                RegTensor<T1> yEvenTmp;
                RegTensor<T1> yOddTmp;
                RegTensor<T1> yTmp;
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopyUnAlign(xTmp, uSrc, xLocal, VL);
                    Cast<float, T1, castTraitB162B32Even>(xEven, xTmp, pregMain);
                    Cast<float, T1, castTraitB162B32Odd>(xOdd, xTmp, pregMain);
                    VFInnerNormalize(xEven, mean, rstd, gamma, beta, yEven, pregMain);
                    VFInnerNormalize(xOdd, mean, rstd, gamma, beta, yOdd, pregMain);
                    Cast<T1, float, castTraitB322B16Even>(yEvenTmp, yEven, pregMain);
                    Cast<T1, float, castTraitB322B16Odd>(yOddTmp, yOdd, pregMain);
                    Or((RegTensor<int16_t>&)yTmp, (RegTensor<int16_t>&)yEvenTmp, (RegTensor<int16_t>&)yOddTmp,
                       pregMain);
                    DataCopyUnAlign(yLocal, yTmp, uDst, VL);
                }
                uint32_t sreg0 = tailNum;
                for (uint16_t k = 0; k < tailLoop; k++) {
                    pregLoop = UpdateMask<half>(sreg0);
                    DataCopyUnAlign(xTmp, uSrc, xLocal, tailNum);
                    Cast<float, T1, castTraitB162B32Even>(xEven, xTmp, pregLoop);
                    Cast<float, T1, castTraitB162B32Odd>(xOdd, xTmp, pregLoop);
                    VFInnerNormalize(xEven, mean, rstd, gamma, beta, yEven, pregLoop);
                    VFInnerNormalize(xOdd, mean, rstd, gamma, beta, yOdd, pregLoop);
                    Cast<T1, float, castTraitB322B16Even>(yEvenTmp, yEven, pregLoop);
                    Cast<T1, float, castTraitB322B16Odd>(yOddTmp, yOdd, pregLoop);
                    Or((RegTensor<int16_t>&)yTmp, (RegTensor<int16_t>&)yEvenTmp, (RegTensor<int16_t>&)yOddTmp,
                       pregLoop);
                    DataCopyUnAlign(yLocal, yTmp, uDst, tailNum);
                }
                DataCopyUnAlignPost(yLocal, uDst, 0);
            } else {
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopyUnAlign(x, uSrc, xLocal, VL_FP32);
                    VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregMain);
                    DataCopyUnAlign(yLocal, y, uDst, VL_FP32);
                }
                uint32_t sreg0 = tailNum;
                for (uint16_t k = 0; k < tailLoop; k++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopyUnAlign(x, uSrc, xLocal, tailNum);
                    VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregLoop);
                    DataCopyUnAlign(yLocal, y, uDst, tailNum);
                }
                DataCopyUnAlignPost(yLocal, uDst, 0);
            }
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeAlign(
    __local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
    __local_mem__ float* rstdLocal, __local_mem__ T1* yLocal, uint16_t rowsCount, int32_t reduceCount)
{
    uint16_t loopCount = CeilDiv(reduceCount, VL_FP32);
    uint32_t reduceCountAlign = RoundUp<T1>(reduceCount);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> rstd;
        RegTensor<float> y;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstd, rstdLocal);
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(mean, meanLocal);
        for (uint16_t i = 0; i < rowsCount; i++) {
            uint32_t sreg0 = reduceCount;
            LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain, i);
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadInputData<T1>(x, xLocal, pregLoop, i * reduceCountAlign + j * VL_FP32);
                VFInnerNormalize(x, mean, rstd, gamma, beta, y, pregLoop);
                StoreOutputData<T1>(yLocal, y, pregLoop, i * reduceCountAlign + j * VL_FP32);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CopyGammaAndBeta2UB(
    const GlobalTensor<T>& gammaGm, const GlobalTensor<T>& betaGm, const LocalTensor<T>& gammaTensor,
    const LocalTensor<T>& betaTensor, const uint16_t blockCount, const uint32_t copyLen, bool hasGamma = true,
    bool hasBeta = true)
{
    int32_t copyLenAlign = RoundUp<T>(copyLen);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.paddingValue = static_cast<T>(0.0);
    padParams.rightPadding = 0;
    padParams.rightPadding = copyLenAlign - copyLen;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    if (hasGamma) {
        DataCopyPad(gammaTensor, gammaGm, dataCopyParams, padParams);
    }
    if (hasBeta) {
        DataCopyPad(betaTensor, betaGm, dataCopyParams, padParams);
    }
}

template <typename T>
__aicore__ inline void CopyX2UB(
    const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t blockCount,
    const uint32_t copyLen)
{
    int32_t copyLenAlign = RoundUp<T>(copyLen);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.paddingValue = static_cast<T>(0.0);
    padParams.rightPadding = copyLenAlign - copyLen;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyMeanAndRstd2Gm(
    const GlobalTensor<T>& meanGm, const GlobalTensor<T>& rstdGm, const LocalTensor<T>& meanTensor,
    const LocalTensor<T>& rstdTensor, const uint16_t blockCount, const uint32_t copyLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(meanGm, meanTensor, dataCopyParams);
    DataCopyPad(rstdGm, rstdTensor, dataCopyParams);
}

template <typename T1>
__aicore__ inline void ProcessMeanAndRstd(
    LocalTensor<float>& meanTensor, LocalTensor<T1>& meanOutTensor, GlobalTensor<T1>& meanGm,
    LocalTensor<float>& rstdTensor, LocalTensor<T1>& rstdOutTensor, GlobalTensor<T1>& rstdGm, uint64_t gmOffset,
    uint32_t curNumPerCore)
{
    if constexpr (IsSameType<T1, float>::value) {
        CopyMeanAndRstd2Gm<float>(meanGm[gmOffset], rstdGm[gmOffset], meanTensor, rstdTensor, 1, curNumPerCore);
    } else {
        __local_mem__ T1* meanOutLocal = (__local_mem__ T1*)meanOutTensor.GetPhyAddr();
        __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ T1* rstdOutLocal = (__local_mem__ T1*)rstdOutTensor.GetPhyAddr();
        __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor.GetPhyAddr();
        uint16_t loopCount = CeilDiv(curNumPerCore, VL_FP32);
        __VEC_SCOPE__
        {
            uint32_t sreg0 = curNumPerCore;
            MaskReg pregLoop;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<T1> meanOut;
            RegTensor<T1> rstdOut;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = UpdateMask<float>(sreg0);
                DataCopy(mean, meanLocal + i * VL_FP32);
                DataCopy(rstd, rstdLocal + i * VL_FP32);
                Cast<T1, float, castTraitB322B16Even>(meanOut, mean, pregLoop);
                Cast<T1, float, castTraitB322B16Even>(rstdOut, rstd, pregLoop);
                DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    meanOutLocal + i * VL_FP32, meanOut, pregLoop);
                DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                    rstdOutLocal + i * VL_FP32, rstdOut, pregLoop);
            }
        }
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        CopyMeanAndRstd2Gm<T1>(meanGm[gmOffset], rstdGm[gmOffset], meanOutTensor, rstdOutTensor, 1, curNumPerCore);
    }
}

template <typename T>
__aicore__ inline void CopyY2Gm(
    const GlobalTensor<T>& yGm, const LocalTensor<T>& yTensor, uint16_t blockCount, const uint32_t copyLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(yGm, yTensor, dataCopyParams);
}

} // namespace GroupNormV2

#endif