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
 * \file group_norm_regbase_base.h
 * \brief
 */
#ifndef GROUP_NORM_REGBASE_BASE_H_
#define GROUP_NORM_REGBASE_BASE_H_
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
namespace GroupNorm {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UnalignReg;
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
static constexpr int32_t VL_FP32 = VECTOR_REG_WIDTH / sizeof(float);
static constexpr float ZERO = 0.0f;
constexpr static AscendC::Reg::CastTrait castTraitB162B32Even = {
    AscendC::Reg::RegLayout::ZERO, // even
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::Reg::CastTrait castTraitB162B32Odd = {
    AscendC::Reg::RegLayout::ONE, // odd
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::Reg::CastTrait castTraitB322B16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr static AscendC::Reg::CastTrait castTraitB322B16Odd = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
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
        DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadGammaAndBetaData(RegTensor<float>& gamma, RegTensor<float>& beta,
                                            __local_mem__ T* gammaLocal, __local_mem__ T* betaLocal, MaskReg pregLoop,
                                            uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(gamma, gammaLocal + srcOffset);
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(beta, betaLocal + srcOffset);
    } else {
        RegTensor<T> gammaB16;
        DataCopy<T, AscendC::Reg::LoadDist::DIST_BRC_B16>(gammaB16, gammaLocal + srcOffset);
        Cast<float, T, castTraitB162B32Even>(gamma, gammaB16, pregLoop);
        RegTensor<T> betaB16;
        DataCopy<T, AscendC::Reg::LoadDist::DIST_BRC_B16>(betaB16, betaLocal + srcOffset);
        Cast<float, T, castTraitB162B32Even>(beta, betaB16, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOutputData(__local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop,
                                       uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else {
        RegTensor<T> tmpB16;
        Cast<T, float, castTraitB322B16Even>(tmpB16, src, pregLoop);
        DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmpB16, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreStatisticData(__local_mem__ T* dst, RegTensor<float>& src, MaskReg pregLoop,
                                          uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dst + dstOffset, src, pregLoop);
    } else {
        RegTensor<T> tmpB16;
        Cast<T, float, castTraitB322B16Even>(tmpB16, src, pregLoop);
        DataCopy<T, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B16>(dst + dstOffset, tmpB16, pregLoop);
    }
}

template <typename T>
__aicore__ inline void VFInnerWelfordParallelUpdateWithInit(__local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal,
                                                            __local_mem__ float* tmpVarLocal, uint64_t calLen,
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
__aicore__ inline void VFInnerWelfordParallelUpdate(__local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal,
                                                    __local_mem__ float* tmpVarLocal, uint64_t calLen, float scale)
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
__aicore__ inline void VFWelfordParallelUpdate(__local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal,
                                               __local_mem__ float* tmpVarLocal, uint64_t curLoop, uint64_t calLen,
                                               float scale)
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
template <typename T>
__aicore__ inline void VFWelfordParallelFinalizeAlign(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
    __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
    uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
    uint32_t offset, float reduceScale, float scale, float cnt, float eps)
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
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
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
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
        }

        // PART2: 整块剩余部分vcadd回刷UB
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, scale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
        }

        NormCommon::DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

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
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
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
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
        }

        NormCommon::DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        StoreStatisticData<T>(varianceOutLocal, var, pregMerge, offset);
        NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
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
#include "group_norm_regbase_base_part1.h"
#include "group_norm_regbase_base_part2.h"
#include "group_norm_regbase_base_part3.h"

} // namespace GroupNorm

#endif
