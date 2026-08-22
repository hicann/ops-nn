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
 * \file group_norm_silu_quant_regbase_base.h
 * \brief GroupNormSiluQuant arch35 regbase 公共设备端 helper: reg 载入/存出、Welford 单遍归约、
 *        归一化+SiLU 的 VF 计算, 以及量化输出(StoreQuantData: silu/quantScale→int8)与 CopyQuantScale2UB。
 */
#ifndef GROUP_NORM_SILU_QUANT_REGBASE_BASE_H_
#define GROUP_NORM_SILU_QUANT_REGBASE_BASE_H_
#include "kernel_operator.h"
#include "group_norm_silu_quant_tiling_data.h"
namespace GroupNormSiluQuant {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;
static constexpr int32_t BLOCK_SIZE = 32;
static constexpr int32_t FP32_ONE_REPEAT = 64;
static constexpr int32_t FLOAT_BYTE_SIZE = 4;
static constexpr int32_t INDEX_0 = 0;
static constexpr int32_t INDEX_1 = 1;
static constexpr int32_t INDEX_2 = 2;
static constexpr int32_t MAX_ONCE_NUM_PER_CORE = 2048;
static constexpr int32_t DICHOTOMY_ADD_COEFF = 2;
static constexpr int32_t VL_FP32 = VECTOR_REG_WIDTH / sizeof(float);
static constexpr int32_t GAMMA_BETA_UB_DIM = 3;
static constexpr int32_t NDDMA_SIZE = 32;
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

// GroupNormSiluQuant 量化输出用: fp32 -> fp16 -> int8, 对齐 A2 ComputeQuant(fp32->half->CastFromF16ToI8)
// 参照 norm/rms_norm_quant_v2 arch35 regbase 权威写法
constexpr static AscendC::MicroAPI::CastTrait castTraitFp322Fp16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ODD,
};

// fp32 -> fp16 奇数布局(与 castTraitFp322Fp16 的偶数布局配对), 用于 UnAlign 量化路径的 even/odd 打包。
constexpr static AscendC::MicroAPI::CastTrait castTraitFp322Fp16Odd = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ODD,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitFp162Int8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
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

template <typename T, bool hasGamma, bool hasBeta>
__aicore__ inline void LoadGammaAndBetaData(RegTensor<float>& gamma, RegTensor<float>& beta,
                                            __local_mem__ T* gammaLocal, __local_mem__ T* betaLocal, MaskReg pregLoop,
                                            uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        if constexpr (hasGamma) {
            DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(gamma, gammaLocal + srcOffset);
        }
        if constexpr (hasBeta) {
            DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(beta, betaLocal + srcOffset);
        }
    } else {
        if constexpr (hasGamma) {
            RegTensor<T> gammaB16;
            DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(gammaB16, gammaLocal + srcOffset);
            Cast<float, T, castTraitB162B32Even>(gamma, gammaB16, pregLoop);
        }
        if constexpr (hasBeta) {
            RegTensor<T> betaB16;
            DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_BRC_B16>(betaB16, betaLocal + srcOffset);
            Cast<float, T, castTraitB162B32Even>(beta, betaB16, pregLoop);
        }
    }
}

// GroupNormSiluQuant: 对 silu 结果(fp32 reg)做静态量化并存 int8。
// y_int8 = round(silu / quantScale), 走 fp32->fp16->int8, 对齐 A2 ComputeQuant(fp32->half->CastFromF16ToI8)。
// quantScaleReg 为已广播的 quantScale(per-tensor 单标量 / per-channel 逐通道), 用 Div 代替 A2 的 Muls(1/scale)。
__aicore__ inline void StoreQuantData(__local_mem__ int8_t* dst, RegTensor<float>& src, RegTensor<float>& quantScaleReg,
                                      MaskReg pregLoop, uint32_t dstOffset)
{
    RegTensor<float> yf;
    Div(yf, src, quantScaleReg, pregLoop);
    RegTensor<half> yh;
    RegTensor<int8_t> yi8;
    Cast<half, float, castTraitFp322Fp16>(yh, yf, pregLoop);
    Cast<int8_t, half, castTraitFp162Int8>(yi8, yh, pregLoop);
    DataCopy<int8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, yi8, pregLoop);
}

__aicore__ inline void DichotomyAdd(RegTensor<float>& dstReg, __local_mem__ float* src, uint16_t outerLoop,
                                    uint16_t innerLoop, uint32_t lastNum)
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
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> scalar2;
    RegTensor<float> scalar3;
    RegTensor<float> t1;
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
    uint16_t loopCountU = CeilDiv(calLen, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> x1U;
        RegTensor<float> tmpMeanU;
        RegTensor<float> tmpVarU;
        RegTensor<float> delta1U;
        RegTensor<float> delta2U;
        RegTensor<float> delta3U;
        RegTensor<float> delat4U;
        MaskReg pregLoopU;
        uint32_t sreg0U = calLen;
        for (uint16_t iU = 0; iU < loopCountU; iU++) {
            pregLoopU = UpdateMask<float>(sreg0U);
            LoadInputData<T>(x1U, x1Local, pregLoopU, iU * VL_FP32);
            DataCopy(tmpMeanU, tmpMeanLocal + iU * VL_FP32);
            Sub(delta1U, x1U, tmpMeanU, pregLoopU);
            Muls(delta2U, delta1U, scale, pregLoopU);
            Add(tmpMeanU, tmpMeanU, delta2U, pregLoopU);
            DataCopy(tmpMeanLocal + iU * VL_FP32, tmpMeanU, pregLoopU);

            DataCopy(tmpVarU, tmpVarLocal + iU * VL_FP32);
            Sub(delta3U, x1U, tmpMeanU, pregLoopU);
            Mul(delat4U, delta1U, delta3U, pregLoopU);
            Add(tmpVarU, tmpVarU, delat4U, pregLoopU);
            DataCopy(tmpVarLocal + iU * VL_FP32, tmpVarU, pregLoopU);
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
__aicore__ inline void VFWelfordParallelFinalizeAlign(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                                      __local_mem__ float* tmpMeanLocal,
                                                      __local_mem__ float* tmpVarLocal,
                                                      __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
                                                      uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                      uint32_t dichotomyAddLastNum, uint32_t offset, float reduceScale,
                                                      float scale, float cnt, float eps)
{
    uint32_t dichotomyAddReminderAl = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCountAl = CeilDiv(dichotomyAddReminderAl, VL_FP32);
    uint16_t dichotomyAddPowerLoopCountAl = dichotomyAddPower / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCountAl = dichotomyAddPowerLoopCountAl - dichotomyAddReminderLoopCountAl;
    uint32_t tmpReduceCountAl = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOriginAl = tmpReduceCountAl / VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanLAl;
        RegTensor<float> dichotomyAddMeanRAl;
        RegTensor<float> dichotomyAddVarLAl;
        RegTensor<float> dichotomyAddVarRAl;
        RegTensor<float> sumMeanAl;
        RegTensor<float> meanAl;
        RegTensor<float> sumVarAl;
        RegTensor<float> varAl;
        RegTensor<float> deltaLAl;
        RegTensor<float> deltaRAl;
        RegTensor<float> rstdAl;
        MaskReg pregLoopAl;
        MaskReg pregMainAl = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMergeAl = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0Al = dichotomyAddReminderAl;
        // 计算mean
        // PART1: 整尾块合并
        for (uint16_t iAl = 0; iAl < dichotomyAddReminderLoopCountAl; iAl++) {
            pregLoopAl = UpdateMask<float>(sreg0Al);
            DataCopy(dichotomyAddMeanLAl, tmpMeanLocal + iAl * VL_FP32);
            DataCopy(dichotomyAddMeanRAl, tmpMeanLocal + iAl * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanLAl, dichotomyAddMeanLAl, scale, pregMainAl);
            Muls(dichotomyAddMeanRAl, dichotomyAddMeanRAl, scale, pregLoopAl);
            Add(sumMeanAl, dichotomyAddMeanLAl, dichotomyAddMeanRAl, pregMainAl);
            ReduceSum(meanAl, sumMeanAl, pregMainAl);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + iAl, meanAl,
                                                                                  pregMergeAl);
        }

        // PART2: 整块剩余部分vcadd回刷UB
        for (uint16_t iAl = 0; iAl < dichotomyAddPowerRemainLoopCountAl; iAl++) {
            DataCopy(dichotomyAddMeanLAl, tmpMeanLocal + (iAl + dichotomyAddReminderLoopCountAl) * VL_FP32);
            Muls(dichotomyAddMeanLAl, dichotomyAddMeanLAl, scale, pregMainAl);
            ReduceSum(meanAl, dichotomyAddMeanLAl, pregMainAl);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountAl + iAl, meanAl, pregMergeAl);
        }

        DichotomyAdd(meanAl, dichotomyAddLocal, dichotomyAddK, innerLoopCountOriginAl, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, meanAl, pregMergeAl);

        Duplicate(meanAl, meanAl, pregMainAl);
        sreg0Al = dichotomyAddReminderAl;
        // PART1: 整尾块合并
        for (uint16_t iAl = 0; iAl < dichotomyAddReminderLoopCountAl; iAl++) {
            pregLoopAl = UpdateMask<float>(sreg0Al);
            DataCopy(dichotomyAddMeanLAl, tmpMeanLocal + iAl * VL_FP32);
            Sub(deltaLAl, dichotomyAddMeanLAl, meanAl, pregMainAl);
            Mul(deltaLAl, deltaLAl, deltaLAl, pregMainAl);
            Muls(deltaLAl, deltaLAl, cnt, pregMainAl);
            DataCopy(dichotomyAddVarLAl, tmpVarLocal + iAl * VL_FP32);
            Add(dichotomyAddVarLAl, dichotomyAddVarLAl, deltaLAl, pregMainAl);
            Muls(dichotomyAddVarLAl, dichotomyAddVarLAl, reduceScale, pregMainAl);

            DataCopy(dichotomyAddMeanRAl, tmpMeanLocal + iAl * VL_FP32 + dichotomyAddPower);
            Sub(deltaRAl, dichotomyAddMeanRAl, meanAl, pregLoopAl);
            Mul(deltaRAl, deltaRAl, deltaRAl, pregLoopAl);
            Muls(deltaRAl, deltaRAl, cnt, pregLoopAl);
            DataCopy(dichotomyAddVarRAl, tmpVarLocal + iAl * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarRAl, dichotomyAddVarRAl, deltaRAl, pregLoopAl);
            Muls(dichotomyAddVarRAl, dichotomyAddVarRAl, reduceScale, pregLoopAl);

            Add(sumVarAl, dichotomyAddVarLAl, dichotomyAddVarRAl, pregMainAl);
            ReduceSum(varAl, sumVarAl, pregMainAl);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + iAl, varAl,
                                                                                  pregMergeAl);
        }

        // PART2: 整块剩余部分vcadd回刷UB
        for (uint16_t iAl = 0; iAl < dichotomyAddPowerRemainLoopCountAl; iAl++) {
            DataCopy(dichotomyAddMeanLAl, tmpMeanLocal + (iAl + dichotomyAddReminderLoopCountAl) * VL_FP32);
            Sub(deltaLAl, dichotomyAddMeanLAl, meanAl, pregMainAl);
            Mul(deltaLAl, deltaLAl, deltaLAl, pregMainAl);
            Muls(deltaLAl, deltaLAl, cnt, pregMainAl);
            DataCopy(dichotomyAddVarLAl, tmpVarLocal + (iAl + dichotomyAddReminderLoopCountAl) * VL_FP32);
            Add(dichotomyAddVarLAl, dichotomyAddVarLAl, deltaLAl, pregMainAl);
            Muls(dichotomyAddVarLAl, dichotomyAddVarLAl, reduceScale, pregMainAl);
            ReduceSum(varAl, dichotomyAddVarLAl, pregMainAl);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountAl + iAl, varAl, pregMergeAl);
        }

        DichotomyAdd(varAl, dichotomyAddLocal, dichotomyAddK, innerLoopCountOriginAl, dichotomyAddLastNum);
        CalRstdByHighPrecision(varAl, rstdAl, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstdAl, pregMergeAl);
    }
}

/*
  // Welford Finalize非对齐场景计算公式如下:
  counts = np.ones(len(meanAl), dtype=np.float32)*count
  for iAl in range(tail_size):
      counts[iAl] += 1
  finalize_mean = np.sum(meanAl*counts) / np.sum(counts)
  finalize_delta = meanAl - finalize_mean
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
    uint16_t dichotomyAddPowerRemainLoopCount = dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;

    uint32_t welfordDiff = tailSize - dichotomyAddPower;
    uint16_t welfordDiffLoopCount = welfordDiff / VL_FP32;
    uint32_t welfordDiffReminder = welfordDiff - welfordDiffLoopCount * VL_FP32;
    uint32_t welfordDiffReminderAlign = welfordDiffReminder == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCount = welfordDiffReminderAlign / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplit = dichotomyAddReminder - welfordDiffLoopCount * VL_FP32 -
                                              welfordDiffReminderAlign;
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
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean,
                                                                                  pregMerge);
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
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign +
                                            dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
        }
        DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

        // 计算rstd
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
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var,
                                                                                  pregMerge);
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
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign +
                                            dichotomyAddPower);
            Sub(deltaR, dichotomyAddMeanR, mean, pregLoop);
            Mul(deltaR, deltaR, deltaR, pregLoop);
            Muls(deltaR, deltaR, cnt, pregLoop);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR,
                     tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);
            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, var, pregMerge);
        }
        for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
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
    float tailCntS2 = cnt + float(1.0);
    float coeffS2 = tailCntS2 / cnt;
    float tailCountScaleS2 = tailCntS2 * reduceScale;
    float countScaleS2 = cnt * reduceScale;

    uint32_t dichotomyAddReminderS2 = reduceCount - dichotomyAddPower;
    uint16_t welfordDiffLoopCountS2 = tailSize / VL_FP32;
    uint32_t welfordDiffReminderS2 = tailSize - welfordDiffLoopCountS2 * VL_FP32;
    uint32_t welfordDiffReminderAlignS2 = welfordDiffReminderS2 == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCountS2 = welfordDiffReminderAlignS2 / VL_FP32;

    uint16_t dichotomyAddReminderRealLoopCountS2 = CeilDiv(dichotomyAddReminderS2, VL_FP32);
    uint16_t dichotomyAddPowerLoopCountS2 = dichotomyAddPower / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCountS2 = dichotomyAddPowerLoopCountS2 - dichotomyAddReminderRealLoopCountS2;
    uint32_t tmpReduceCountS2 = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOriginS2 = tmpReduceCountS2 / VL_FP32;

    uint32_t dichotomyAddReminderAfterSplitS2 = dichotomyAddReminderS2 - welfordDiffLoopCountS2 * VL_FP32 -
                                                welfordDiffReminderAlignS2;
    uint16_t dichotomyAddReminderLoopCountS2 = CeilDiv(dichotomyAddReminderAfterSplitS2, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanLS2;
        RegTensor<float> dichotomyAddMeanRS2;
        RegTensor<float> dichotomyAddVarLS2;
        RegTensor<float> dichotomyAddVarRS2;
        RegTensor<float> sumMeanS2;
        RegTensor<float> meanS2;
        RegTensor<float> sumVarS2;
        RegTensor<float> varS2;
        RegTensor<float> deltaLS2;
        RegTensor<float> deltaRS2;
        RegTensor<float> rstdS2;
        RegTensor<float> tmpS2;

        MaskReg pregLoopS2;
        MaskReg pregLoop1S2;
        MaskReg pregMainS2 = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMergeS2 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0S2;
        uint32_t sreg1S2;

        // 整块使用tailCountScale,尾块使用countScale
        for (uint16_t iS2 = 0; iS2 < welfordDiffLoopCountS2; iS2++) {
            DataCopy(dichotomyAddMeanLS2, tmpMeanLocal + iS2 * VL_FP32);
            DataCopy(dichotomyAddMeanRS2, tmpMeanLocal + iS2 * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanLS2, dichotomyAddMeanLS2, tailCountScaleS2, pregMainS2);
            Muls(dichotomyAddMeanRS2, dichotomyAddMeanRS2, countScaleS2, pregMainS2);
            Add(sumMeanS2, dichotomyAddMeanLS2, dichotomyAddMeanRS2, pregMainS2);
            ReduceSum(meanS2, sumMeanS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + iS2, meanS2,
                                                                                  pregMergeS2);
        }

        // 处理welford第一次非对齐点, 尾块使用countScale,整块部分使用tailCountScale, 部分使用countScale
        sreg0S2 = dichotomyAddReminderS2 - welfordDiffLoopCountS2 * VL_FP32;
        sreg1S2 = welfordDiffReminderS2;
        for (uint16_t iS2 = 0; iS2 < welfordReminderLoopCountS2; iS2++) {
            pregLoopS2 = UpdateMask<float>(sreg0S2);
            pregLoop1S2 = UpdateMask<float>(sreg1S2);
            DataCopy(dichotomyAddMeanLS2, tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32);
            DataCopy(dichotomyAddMeanRS2, tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanLS2, dichotomyAddMeanLS2, countScaleS2, pregMainS2);
            Muls(dichotomyAddMeanRS2, dichotomyAddMeanRS2, countScaleS2, pregLoopS2);
            Muls(tmpS2, dichotomyAddMeanLS2, coeffS2, pregLoop1S2);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanLS2, tmpS2, pregLoop1S2);
            Add(sumMeanS2, dichotomyAddMeanLS2, dichotomyAddMeanRS2, pregMainS2);
            ReduceSum(meanS2, sumMeanS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + iS2 + welfordDiffLoopCountS2, meanS2, pregMergeS2);
        }

        // 整块使用countScale,尾块使用countScale
        for (uint16_t iS2 = 0; iS2 < dichotomyAddReminderLoopCountS2; iS2++) {
            pregLoopS2 = UpdateMask<float>(sreg0S2);
            DataCopy(dichotomyAddMeanLS2,
                     tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 + welfordDiffReminderAlignS2);
            DataCopy(dichotomyAddMeanRS2, tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 +
                                              welfordDiffReminderAlignS2 + dichotomyAddPower);
            Muls(dichotomyAddMeanLS2, dichotomyAddMeanLS2, countScaleS2, pregMainS2);
            Muls(dichotomyAddMeanRS2, dichotomyAddMeanRS2, countScaleS2, pregLoopS2);
            Add(sumMeanS2, dichotomyAddMeanLS2, dichotomyAddMeanRS2, pregMainS2);
            ReduceSum(meanS2, sumMeanS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + iS2 + welfordDiffLoopCountS2 + welfordReminderLoopCountS2, meanS2, pregMergeS2);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用countScale
        for (uint16_t iS2 = 0; iS2 < dichotomyAddPowerRemainLoopCountS2; iS2++) {
            DataCopy(dichotomyAddMeanLS2, tmpMeanLocal + (iS2 + dichotomyAddReminderRealLoopCountS2) * VL_FP32);
            Muls(dichotomyAddMeanLS2, dichotomyAddMeanLS2, countScaleS2, pregMainS2);
            ReduceSum(meanS2, dichotomyAddMeanLS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCountS2 + iS2, meanS2, pregMergeS2);
        }
        DichotomyAdd(meanS2, dichotomyAddLocal, dichotomyAddK, innerLoopCountOriginS2, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, meanS2, pregMergeS2);

        // 计算rstd
        Duplicate(meanS2, meanS2, pregMainS2);
        for (uint16_t iS2 = 0; iS2 < welfordDiffLoopCountS2; iS2++) {
            DataCopy(dichotomyAddMeanLS2, tmpMeanLocal + iS2 * VL_FP32);
            Sub(deltaLS2, dichotomyAddMeanLS2, meanS2, pregMainS2);
            Mul(deltaLS2, deltaLS2, deltaLS2, pregMainS2);
            Muls(deltaLS2, deltaLS2, tailCntS2, pregMainS2);
            DataCopy(dichotomyAddMeanRS2, tmpMeanLocal + iS2 * VL_FP32 + dichotomyAddPower);
            Sub(deltaRS2, dichotomyAddMeanRS2, meanS2, pregMainS2);
            Mul(deltaRS2, deltaRS2, deltaRS2, pregMainS2);
            Muls(deltaRS2, deltaRS2, cnt, pregMainS2);

            DataCopy(dichotomyAddVarLS2, tmpVarLocal + iS2 * VL_FP32);
            Add(dichotomyAddVarLS2, dichotomyAddVarLS2, deltaLS2, pregMainS2);
            Muls(dichotomyAddVarLS2, dichotomyAddVarLS2, reduceScale, pregMainS2);
            DataCopy(dichotomyAddVarRS2, tmpVarLocal + iS2 * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarRS2, dichotomyAddVarRS2, deltaRS2, pregMainS2);
            Muls(dichotomyAddVarRS2, dichotomyAddVarRS2, reduceScale, pregMainS2);

            Add(sumVarS2, dichotomyAddVarLS2, dichotomyAddVarRS2, pregMainS2);
            ReduceSum(varS2, sumVarS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + iS2, varS2,
                                                                                  pregMergeS2);
        }
        sreg0S2 = dichotomyAddReminderS2 - welfordDiffLoopCountS2 * VL_FP32;
        sreg1S2 = welfordDiffReminderS2;
        for (uint16_t iS2 = 0; iS2 < welfordReminderLoopCountS2; iS2++) {
            pregLoopS2 = UpdateMask<float>(sreg0S2);
            pregLoop1S2 = UpdateMask<float>(sreg1S2);
            DataCopy(dichotomyAddMeanLS2, tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32);
            Sub(deltaLS2, dichotomyAddMeanLS2, meanS2, pregMainS2);
            Mul(deltaLS2, deltaLS2, deltaLS2, pregMainS2);
            Muls(deltaLS2, deltaLS2, cnt, pregMainS2);
            DataCopy(dichotomyAddMeanRS2, tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 + dichotomyAddPower);
            Sub(deltaRS2, dichotomyAddMeanRS2, meanS2, pregLoopS2);
            Mul(deltaRS2, deltaRS2, deltaRS2, pregLoopS2);
            Muls(deltaRS2, deltaRS2, cnt, pregLoopS2);
            Muls(tmpS2, deltaLS2, coeffS2, pregLoop1S2);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaLS2, tmpS2, pregLoop1S2);

            DataCopy(dichotomyAddVarLS2, tmpVarLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32);
            Add(dichotomyAddVarLS2, dichotomyAddVarLS2, deltaLS2, pregMainS2);
            Muls(dichotomyAddVarLS2, dichotomyAddVarLS2, reduceScale, pregMainS2);
            DataCopy(dichotomyAddVarRS2, tmpVarLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarRS2, dichotomyAddVarRS2, deltaRS2, pregLoopS2);
            Muls(dichotomyAddVarRS2, dichotomyAddVarRS2, reduceScale, pregLoopS2);

            Add(sumVarS2, dichotomyAddVarLS2, dichotomyAddVarRS2, pregMainS2);
            ReduceSum(varS2, sumVarS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + iS2 + welfordDiffLoopCountS2, varS2, pregMergeS2);
        }

        for (uint16_t iS2 = 0; iS2 < dichotomyAddReminderLoopCountS2; iS2++) {
            pregLoopS2 = UpdateMask<float>(sreg0S2);
            DataCopy(dichotomyAddMeanLS2,
                     tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 + welfordDiffReminderAlignS2);
            Sub(deltaLS2, dichotomyAddMeanLS2, meanS2, pregMainS2);
            Mul(deltaLS2, deltaLS2, deltaLS2, pregMainS2);
            Muls(deltaLS2, deltaLS2, cnt, pregMainS2);
            DataCopy(dichotomyAddMeanRS2, tmpMeanLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 +
                                              welfordDiffReminderAlignS2 + dichotomyAddPower);
            Sub(deltaRS2, dichotomyAddMeanRS2, meanS2, pregLoopS2);
            Mul(deltaRS2, deltaRS2, deltaRS2, pregLoopS2);
            Muls(deltaRS2, deltaRS2, cnt, pregLoopS2);

            DataCopy(dichotomyAddVarLS2,
                     tmpVarLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 + welfordDiffReminderAlignS2);
            Add(dichotomyAddVarLS2, dichotomyAddVarLS2, deltaLS2, pregMainS2);
            Muls(dichotomyAddVarLS2, dichotomyAddVarLS2, reduceScale, pregMainS2);
            DataCopy(dichotomyAddVarRS2, tmpVarLocal + (iS2 + welfordDiffLoopCountS2) * VL_FP32 +
                                             welfordDiffReminderAlignS2 + dichotomyAddPower);
            Add(dichotomyAddVarRS2, dichotomyAddVarRS2, deltaRS2, pregLoopS2);
            Muls(dichotomyAddVarRS2, dichotomyAddVarRS2, reduceScale, pregLoopS2);
            Add(sumVarS2, dichotomyAddVarLS2, dichotomyAddVarRS2, pregMainS2);
            ReduceSum(varS2, sumVarS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + iS2 + welfordDiffLoopCountS2 + welfordReminderLoopCountS2, varS2, pregMergeS2);
        }

        for (uint16_t iS2 = 0; iS2 < dichotomyAddPowerRemainLoopCountS2; iS2++) {
            DataCopy(dichotomyAddMeanLS2, tmpMeanLocal + (iS2 + dichotomyAddReminderRealLoopCountS2) * VL_FP32);
            Sub(deltaLS2, dichotomyAddMeanLS2, meanS2, pregMainS2);
            Mul(deltaLS2, deltaLS2, deltaLS2, pregMainS2);
            Muls(deltaLS2, deltaLS2, cnt, pregMainS2);
            DataCopy(dichotomyAddVarLS2, tmpVarLocal + (iS2 + dichotomyAddReminderRealLoopCountS2) * VL_FP32);
            Add(dichotomyAddVarLS2, dichotomyAddVarLS2, deltaLS2, pregMainS2);
            Muls(dichotomyAddVarLS2, dichotomyAddVarLS2, reduceScale, pregMainS2);
            ReduceSum(varS2, dichotomyAddVarLS2, pregMainS2);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCountS2 + iS2, varS2, pregMergeS2);
        }
        DichotomyAdd(varS2, dichotomyAddLocal, dichotomyAddK, innerLoopCountOriginS2, dichotomyAddLastNum);
        CalRstdByHighPrecision(varS2, rstdS2, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstdS2, pregMergeS2);
    }
}

// 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ float* tmpMeanLocal,
    __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
    uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum, uint32_t offset,
    uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    float tailCntS3 = cnt + float(1.0);
    float coeffS3 = tailCntS3 / cnt;
    float tailCountScaleS3 = tailCntS3 * reduceScale;
    float countScaleS3 = cnt * reduceScale;

    // 二分累加
    uint32_t dichotomyAddReminderS3 = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCountS3 = CeilDiv(dichotomyAddReminderS3, VL_FP32);
    uint16_t dichotomyAddPowerLoopCountS3 = dichotomyAddPower / VL_FP32;
    uint32_t tmpReduceCountS3 = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOriginS3 = tmpReduceCountS3 / VL_FP32;
    uint32_t dichotomyAddReminderRoundUp = dichotomyAddReminderLoopCountS3 * VL_FP32;

    uint32_t welfordDiffS3 = tailSize - dichotomyAddReminderRoundUp;
    uint16_t welfordDiffLoopCountS3 = welfordDiffS3 / VL_FP32;
    uint32_t welfordDiffReminderS3 = welfordDiffS3 - welfordDiffLoopCountS3 * VL_FP32;
    uint32_t welfordDiffReminderAlignS3 = welfordDiffReminderS3 == 0 ? 0 : VL_FP32;
    uint16_t welfordReminderLoopCountS3 = welfordDiffReminderAlignS3 / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCountS3 = dichotomyAddPowerLoopCountS3 - dichotomyAddReminderLoopCountS3 -
                                                  welfordDiffLoopCountS3 - welfordReminderLoopCountS3;
    uint32_t dichotomyAddPowerOffset = dichotomyAddReminderRoundUp + welfordDiffLoopCountS3 * VL_FP32 +
                                       welfordDiffReminderAlignS3;

    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddMeanLS3;
        RegTensor<float> dichotomyAddMeanRS3;
        RegTensor<float> dichotomyAddVarLS3;
        RegTensor<float> dichotomyAddVarRS3;
        RegTensor<float> sumMeanS3;
        RegTensor<float> meanS3;
        RegTensor<float> sumVarS3;
        RegTensor<float> varS3;
        RegTensor<float> deltaLS3;
        RegTensor<float> deltaRS3;
        RegTensor<float> rstdS3;
        RegTensor<float> tmpS3;

        MaskReg pregLoopS3;
        MaskReg pregMainS3 = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMergeS3 = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        uint32_t sreg0S3 = dichotomyAddReminderS3;
        // 整块使用tailCountScale, 尾块使用CountScale
        for (uint16_t iS3 = 0; iS3 < dichotomyAddReminderLoopCountS3; iS3++) {
            pregLoopS3 = UpdateMask<float>(sreg0S3);
            DataCopy(dichotomyAddMeanLS3, tmpMeanLocal + iS3 * VL_FP32);
            DataCopy(dichotomyAddMeanRS3, tmpMeanLocal + iS3 * VL_FP32 + dichotomyAddPower);
            Muls(dichotomyAddMeanLS3, dichotomyAddMeanLS3, tailCountScaleS3, pregMainS3);
            Muls(dichotomyAddMeanRS3, dichotomyAddMeanRS3, countScaleS3, pregLoopS3);
            Add(sumMeanS3, dichotomyAddMeanLS3, dichotomyAddMeanRS3, pregMainS3);
            ReduceSum(meanS3, sumMeanS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + iS3, meanS3,
                                                                                  pregMergeS3);
        }

        // 剩余整块需要拆分成多部分
        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t iS3 = 0; iS3 < welfordDiffLoopCountS3; iS3++) {
            DataCopy(dichotomyAddMeanLS3, tmpMeanLocal + iS3 * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanLS3, dichotomyAddMeanLS3, tailCountScaleS3, pregMainS3);
            ReduceSum(meanS3, dichotomyAddMeanLS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountS3 + iS3, meanS3, pregMergeS3);
        }

        sreg0S3 = welfordDiffReminderS3;
        for (uint16_t iS3 = 0; iS3 < welfordReminderLoopCountS3; iS3++) {
            pregLoopS3 = UpdateMask<float>(sreg0S3);
            DataCopy(dichotomyAddMeanLS3,
                     tmpMeanLocal + (iS3 + welfordDiffLoopCountS3) * VL_FP32 + dichotomyAddReminderRoundUp);
            Muls(dichotomyAddMeanLS3, dichotomyAddMeanLS3, countScaleS3, pregMainS3);
            Muls(tmpS3, dichotomyAddMeanLS3, coeffS3, pregLoopS3);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(dichotomyAddMeanLS3, tmpS3, pregLoopS3);
            ReduceSum(meanS3, dichotomyAddMeanLS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountS3 + welfordDiffLoopCountS3 + iS3, meanS3,
                pregMergeS3);
        }

        for (uint16_t iS3 = 0; iS3 < dichotomyAddPowerRemainLoopCountS3; iS3++) {
            DataCopy(dichotomyAddMeanLS3, tmpMeanLocal + iS3 * VL_FP32 + dichotomyAddPowerOffset);
            Muls(dichotomyAddMeanLS3, dichotomyAddMeanLS3, countScaleS3, pregMainS3);
            ReduceSum(meanS3, dichotomyAddMeanLS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountS3 + welfordDiffLoopCountS3 +
                    welfordReminderLoopCountS3 + iS3,
                meanS3, pregMergeS3);
        }

        DichotomyAdd(meanS3, dichotomyAddLocal, dichotomyAddK, innerLoopCountOriginS3, dichotomyAddLastNum);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, meanS3, pregMergeS3);

        // 计算rstd
        Duplicate(meanS3, meanS3, pregMainS3);
        // 整块使用tailCountScale, 尾块使用CountScale
        sreg0S3 = dichotomyAddReminderS3;
        for (uint16_t iS3 = 0; iS3 < dichotomyAddReminderLoopCountS3; iS3++) {
            pregLoopS3 = UpdateMask<float>(sreg0S3);
            DataCopy(dichotomyAddMeanLS3, tmpMeanLocal + iS3 * VL_FP32);
            Sub(deltaLS3, dichotomyAddMeanLS3, meanS3, pregMainS3);
            Mul(deltaLS3, deltaLS3, deltaLS3, pregMainS3);
            Muls(deltaLS3, deltaLS3, tailCntS3, pregMainS3);
            DataCopy(dichotomyAddVarLS3, tmpVarLocal + iS3 * VL_FP32);
            Add(dichotomyAddVarLS3, dichotomyAddVarLS3, deltaLS3, pregMainS3);
            Muls(dichotomyAddVarLS3, dichotomyAddVarLS3, reduceScale, pregMainS3);

            DataCopy(dichotomyAddMeanRS3, tmpMeanLocal + iS3 * VL_FP32 + dichotomyAddPower);
            Sub(deltaRS3, dichotomyAddMeanRS3, meanS3, pregLoopS3);
            Mul(deltaRS3, deltaRS3, deltaRS3, pregLoopS3);
            Muls(deltaRS3, deltaRS3, cnt, pregLoopS3);
            DataCopy(dichotomyAddVarRS3, tmpVarLocal + iS3 * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarRS3, dichotomyAddVarRS3, deltaRS3, pregLoopS3);
            Muls(dichotomyAddVarRS3, dichotomyAddVarRS3, reduceScale, pregLoopS3);

            Add(sumVarS3, dichotomyAddVarLS3, dichotomyAddVarRS3, pregMainS3);
            ReduceSum(varS3, sumVarS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + iS3, varS3,
                                                                                  pregMergeS3);
        }

        // 整块剩余部分回刷UB，整块使用tailCountScale
        for (uint16_t iS3 = 0; iS3 < welfordDiffLoopCountS3; iS3++) {
            DataCopy(dichotomyAddMeanLS3, tmpMeanLocal + iS3 * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaLS3, dichotomyAddMeanLS3, meanS3, pregMainS3);
            Mul(deltaLS3, deltaLS3, deltaLS3, pregMainS3);
            Muls(deltaLS3, deltaLS3, tailCntS3, pregMainS3);
            DataCopy(dichotomyAddVarLS3, tmpVarLocal + iS3 * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarLS3, dichotomyAddVarLS3, deltaLS3, pregMainS3);
            Muls(dichotomyAddVarLS3, dichotomyAddVarLS3, reduceScale, pregMainS3);
            ReduceSum(varS3, dichotomyAddVarLS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountS3 + iS3, varS3, pregMergeS3);
        }

        sreg0S3 = welfordDiffReminderS3;
        for (uint16_t iS3 = 0; iS3 < welfordReminderLoopCountS3; iS3++) {
            pregLoopS3 = UpdateMask<float>(sreg0S3);
            DataCopy(dichotomyAddMeanLS3,
                     tmpMeanLocal + (iS3 + welfordDiffLoopCountS3) * VL_FP32 + dichotomyAddReminderRoundUp);
            Sub(deltaLS3, dichotomyAddMeanLS3, meanS3, pregMainS3);
            Mul(deltaLS3, deltaLS3, deltaLS3, pregMainS3);
            Muls(deltaLS3, deltaLS3, cnt, pregMainS3);
            Muls(tmpS3, deltaLS3, coeffS3, pregLoopS3);
            Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(deltaLS3, tmpS3, pregLoopS3);
            DataCopy(dichotomyAddVarLS3,
                     tmpVarLocal + (iS3 + welfordDiffLoopCountS3) * VL_FP32 + dichotomyAddReminderRoundUp);
            Add(dichotomyAddVarLS3, dichotomyAddVarLS3, deltaLS3, pregMainS3);
            Muls(dichotomyAddVarLS3, dichotomyAddVarLS3, reduceScale, pregMainS3);
            ReduceSum(varS3, dichotomyAddVarLS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountS3 + welfordDiffLoopCountS3 + iS3, varS3, pregMergeS3);
        }

        for (uint16_t iS3 = 0; iS3 < dichotomyAddPowerRemainLoopCountS3; iS3++) {
            DataCopy(dichotomyAddMeanLS3, tmpMeanLocal + iS3 * VL_FP32 + dichotomyAddPowerOffset);
            Sub(deltaLS3, dichotomyAddMeanLS3, meanS3, pregMainS3);
            Mul(deltaLS3, deltaLS3, deltaLS3, pregMainS3);
            Muls(deltaLS3, deltaLS3, cnt, pregMainS3);
            DataCopy(dichotomyAddVarLS3, tmpVarLocal + iS3 * VL_FP32 + dichotomyAddPowerOffset);
            Add(dichotomyAddVarLS3, dichotomyAddVarLS3, deltaLS3, pregMainS3);
            Muls(dichotomyAddVarLS3, dichotomyAddVarLS3, reduceScale, pregMainS3);
            ReduceSum(varS3, dichotomyAddVarLS3, pregMainS3);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderLoopCountS3 + welfordDiffLoopCountS3 +
                    welfordReminderLoopCountS3 + iS3,
                varS3, pregMergeS3);
        }

        DichotomyAdd(varS3, dichotomyAddLocal, dichotomyAddK, innerLoopCountOriginS3, dichotomyAddLastNum);
        CalRstdByHighPrecision(varS3, rstdS3, eps);
        DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstdS3, pregMergeS3);
    }
}

__aicore__ inline void VFWelfordParallelFinalizeNonAlign(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                                         __local_mem__ float* tmpMeanLocal,
                                                         __local_mem__ float* tmpVarLocal,
                                                         __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
                                                         uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                         uint32_t dichotomyAddLastNum, uint32_t offset,
                                                         uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint32_t dichotomyAddReminderRoundUp = CeilDiv(dichotomyAddReminder, VL_FP32) * VL_FP32;
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

__aicore__ inline void VFWelfordParallelFinalize(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                                 __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
                                                 __local_mem__ float* dichotomyAddLocal, uint32_t reduceCount,
                                                 uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                 uint32_t dichotomyAddLastNum, uint32_t offset, uint32_t tailSize,
                                                 float reduceScale, float scale, float cnt, float eps,
                                                 bool welfordAlign)
{
    if (welfordAlign) {
        VFWelfordParallelFinalizeAlign(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal, reduceCount,
                                       dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, offset, reduceScale,
                                       scale, cnt, eps);
    } else {
        cnt = cnt - 1;
        VFWelfordParallelFinalizeNonAlign(meanLocal, rstdLocal, tmpMeanLocal, tmpVarLocal, dichotomyAddLocal,
                                          reduceCount, dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, offset,
                                          tailSize, reduceScale, cnt, eps);
    }
}

template <typename T>
__aicore__ inline void CalMeanAndRstdByDichotomyAdd(__local_mem__ T* xLocal, __local_mem__ float* meanLocal,
                                                    __local_mem__ float* rstdLocal,
                                                    __local_mem__ float* dichotomyAddLocal, uint16_t numPerCoreProcess,
                                                    uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                    uint32_t dichotomyAddLastNum, uint64_t powerOfTwoForReduce,
                                                    uint64_t reduceCount, float eps)
{
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint16_t dichotomyAddReminderLoopCount = CeilDiv(dichotomyAddReminder, VL_FP32);
    uint16_t dichotomyAddPowerLoopCount = dichotomyAddPower / VL_FP32;
    uint16_t dichotomyAddPowerRemainLoopCount = dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount;
    uint32_t tmpReduceCount = dichotomyAddPower / VL_FP32;
    uint16_t innerLoopCountOrigin = tmpReduceCount / VL_FP32;
    uint32_t elemNumAlign = RoundUp<T>(reduceCount);
    float n = static_cast<float>(1) / static_cast<float>(powerOfTwoForReduce);
    float nCorrectionFactor = static_cast<float>(powerOfTwoForReduce) / static_cast<float>(reduceCount);
    __VEC_SCOPE__
    {
        RegTensor<float> dichotomyAddL;
        RegTensor<float> dichotomyAddR;
        RegTensor<float> sumMean;
        RegTensor<float> mean;
        RegTensor<float> sumVar;
        RegTensor<float> var;
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
                Muls(dichotomyAddL, dichotomyAddL, n, pregMain);
                Muls(dichotomyAddR, dichotomyAddR, n, pregLoop);
                Add(sumMean, dichotomyAddL, dichotomyAddR, pregMain);
                ReduceSum(mean, sumMean, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + j, mean,
                                                                                      pregMerge);
            }

            // 整块剩余部分vcadd回刷UB
            for (uint16_t j = 0; j < dichotomyAddPowerRemainLoopCount; j++) {
                LoadInputData<T>(dichotomyAddL, xLocal, pregMain,
                                 i * elemNumAlign + (j + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddL, dichotomyAddL, n, pregMain);
                ReduceSum(mean, dichotomyAddL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + j, mean, pregMerge);
            }

            DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(mean, mean, nCorrectionFactor, pregMerge);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + i, mean, pregMerge);
            // 计算rstd
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
                Muls(dichotomyAddL, dichotomyAddL, n, pregMain);
                Muls(dichotomyAddR, dichotomyAddR, n, pregLoop);
                Add(sumVar, dichotomyAddL, dichotomyAddR, pregMain);
                ReduceSum(var, sumVar, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + j, var,
                                                                                      pregMerge);
            }

            // 整块剩余部分vcadd回刷UB
            for (uint16_t j = 0; j < dichotomyAddPowerRemainLoopCount; j++) {
                LoadInputData<T>(dichotomyAddL, xLocal, pregMain,
                                 i * elemNumAlign + (j + dichotomyAddReminderLoopCount) * VL_FP32);
                Sub(dichotomyAddL, dichotomyAddL, mean, pregMain);
                Mul(dichotomyAddL, dichotomyAddL, dichotomyAddL, pregMain);
                Muls(dichotomyAddL, dichotomyAddL, n, pregMain);
                ReduceSum(var, dichotomyAddL, pregMain);
                DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + j, var, pregMerge);
            }
            DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            Muls(var, var, nCorrectionFactor, pregMerge);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + i, rstd, pregMerge);
        }
    }
}

// R轴小于64
template <typename T>
__aicore__ inline void CalMeanAndRstdSpecial(__local_mem__ T* xLocal, __local_mem__ float* meanLocal,
                                             __local_mem__ float* rstdLocal, uint16_t numPerCoreProcess,
                                             uint64_t powerOfTwoForReduce, uint64_t reduceCount, float eps)
{
    uint32_t elemNumAlign = RoundUp<T>(reduceCount);
    float n = static_cast<float>(1) / static_cast<float>(powerOfTwoForReduce);
    float nCorrectionFactor = static_cast<float>(powerOfTwoForReduce) / static_cast<float>(reduceCount);
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> xScale;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::MicroAPI::MaskPattern::VL1>();
        for (uint16_t i = 0; i < numPerCoreProcess; i++) {
            uint32_t sreg0 = reduceCount;
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputData<T>(x, xLocal, pregLoop, i * elemNumAlign);
            Muls(xScale, x, n, pregLoop);
            ReduceSum(mean, xScale, pregLoop);
            Muls(mean, mean, nCorrectionFactor, pregMerge);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + i, mean, pregMerge);

            Duplicate(mean, mean, pregMain);
            Sub(x, x, mean, pregLoop);
            Mul(x, x, x, pregLoop);
            Muls(xScale, x, n, pregLoop);
            ReduceSum(var, xScale, pregLoop);
            Muls(var, var, nCorrectionFactor, pregMerge);
            CalRstdByHighPrecision(var, rstd, eps);
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + i, rstd, pregMerge);
        }
    }
}

template <typename T>
__aicore__ inline void CalMeanAndRstd(__local_mem__ T* xLocal, __local_mem__ float* meanLocal,
                                      __local_mem__ float* rstdLocal, __local_mem__ float* dichotomyAddLocal,
                                      uint16_t numPerCoreProcess, uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                      uint32_t dichotomyAddLastNum, uint64_t powerOfTwoForReduce, uint64_t reduceCount,
                                      float eps)
{
    if (dichotomyAddPower >= VL_FP32) {
        CalMeanAndRstdByDichotomyAdd(xLocal, meanLocal, rstdLocal, dichotomyAddLocal, numPerCoreProcess,
                                     dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum, powerOfTwoForReduce,
                                     reduceCount, eps);
        return;
    }
    CalMeanAndRstdSpecial(xLocal, meanLocal, rstdLocal, numPerCoreProcess, powerOfTwoForReduce, reduceCount, eps);
}

template <bool hasGamma, bool hasBeta>
__aicore__ inline void VFInnerNormalizeAndSwish(RegTensor<float>& x, RegTensor<float>& mean, RegTensor<float>& rstd,
                                                RegTensor<float>& gamma, RegTensor<float>& beta, RegTensor<float>& y,
                                                MaskReg& preg)
{
    Sub(x, x, mean, preg);
    Mul(x, x, rstd, preg);
    if constexpr (hasGamma) {
        Mul(x, x, gamma, preg);
    }
    if constexpr (hasBeta) {
        Add(x, x, beta, preg);
    }
    // swish
    Muls(y, x, float(-1.0), preg);
    Exp(y, y, preg);
    Adds(y, y, float(1.0), preg);
    Div(y, x, y, preg);
}

template <bool hasGamma, bool hasBeta>
__aicore__ inline void VFInnerNormalize(RegTensor<float>& x, RegTensor<float>& mean, RegTensor<float>& rstd,
                                        RegTensor<float>& gamma, RegTensor<float>& beta, RegTensor<float>& y,
                                        MaskReg& preg)
{
    Sub(y, x, mean, preg);
    Mul(y, y, rstd, preg);
    if constexpr (hasGamma) {
        Mul(y, y, gamma, preg);
    }
    if constexpr (hasBeta) {
        Add(y, y, beta, preg);
    }
}

// GroupNormSiluQuant 量化版: 归一化+silu 后按 quantScale 量化输出 int8。
// yLocal 为 int8_t*; quantScaleLocal 为 fp32(perChannelScale 时逐通道, 否则单标量在 offset 0)。
template <typename T1, typename T2, bool activateSilu, bool hasGamma, bool hasBeta, bool perChannelScale>
__aicore__ inline void VFInnerNormalizeAndSwishQuantAlign(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                                          __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                                          __local_mem__ float* rstdLocal,
                                                          __local_mem__ float* quantScaleLocal,
                                                          __local_mem__ int8_t* yLocal, uint16_t rowsCount,
                                                          int32_t reduceCount)
{
    uint16_t loopCount = CeilDiv(reduceCount, VL_FP32);
    // 行 stride 按读取粒度 VL_FP32 对齐(每块消费 VL_FP32 个 fp16):loopCount*VL_FP32 == reduceCountAlign,
    // 保证最后一块读不出行。按 block(RoundUp<T1>)对齐时 hwNum 非 VL_FP32 倍数会过读→越界(VEC_ERROR)。
    // 打包侧(host processSize / kernel hwNumAlign)已同步按 VL_FP32 对齐,二者必须一致。
    uint32_t reduceCountAlign = CeilDiv(reduceCount, VL_FP32) * VL_FP32;
    __VEC_SCOPE__
    {
        RegTensor<float> x;
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> rstd;
        RegTensor<float> y;
        RegTensor<float> quantScale;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstd, rstdLocal);
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(mean, meanLocal);
        if constexpr (!perChannelScale) {
            DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(quantScale, quantScaleLocal);
        }
        for (uint16_t i = 0; i < rowsCount; i++) {
            uint32_t sreg0 = reduceCount;
            LoadGammaAndBetaData<T2, hasGamma, hasBeta>(gamma, beta, gammaLocal, betaLocal, pregMain, i);
            if constexpr (perChannelScale) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(quantScale, quantScaleLocal + i);
            }
            for (uint16_t j = 0; j < loopCount; j++) {
                pregLoop = UpdateMask<float>(sreg0);
                // x 读、y 存都按 reduceCountAlign(VL_FP32 对齐):防 x 向量读越界,且 int8 PACK4_B32 存需 32B 对齐偏移。
                // 拷回连续 GM 由 CopySilu2Gm 逐行完成(每行从对齐的 UB 偏移拷 hwNum 到连续 GM)。
                LoadInputData<T1>(x, xLocal, pregLoop, i * reduceCountAlign + j * VL_FP32);
                if constexpr (activateSilu) {
                    VFInnerNormalizeAndSwish<hasGamma, hasBeta>(x, mean, rstd, gamma, beta, y, pregLoop);
                } else {
                    VFInnerNormalize<hasGamma, hasBeta>(x, mean, rstd, gamma, beta, y, pregLoop);
                }
                StoreQuantData(yLocal, y, quantScale, pregLoop, i * reduceCountAlign + j * VL_FP32);
            }
        }
    }
}

// GroupNormSiluQuant 量化版 UnAlign(reduce 轴按 hwNum 非 block 对齐, 逐通道非对齐读写)。
// 镜像 VFInnerNormalizeAndSwishUnAlign 的 even/odd 结构, 仅在 fp16 打包前插 /quantScale, 末尾多一步 fp16->int8。
template <typename T1, typename T2, bool activateSilu, bool hasGamma, bool hasBeta, bool perChannelScale>
__aicore__ inline void VFInnerNormalizeAndSwishQuantUnAlign(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                                            __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                                            __local_mem__ float* rstdLocal,
                                                            __local_mem__ float* quantScaleLocal,
                                                            __local_mem__ int8_t* yLocal, uint16_t rowsCount,
                                                            int32_t reduceCount)
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
        RegTensor<float> quantScale;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<T1, AscendC::MicroAPI::MaskPattern::ALL>();

        UnalignReg uSrc;
        UnalignReg uDst;
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstd, rstdLocal);
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(mean, meanLocal);
        if constexpr (!perChannelScale) {
            DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(quantScale, quantScaleLocal);
        }
        DataCopyUnAlignPre<T1>(uSrc, xLocal);
        for (uint16_t i = 0; i < rowsCount; i++) {
            LoadGammaAndBetaData<T2, hasGamma, hasBeta>(gamma, beta, gammaLocal, betaLocal, pregMain, i);
            if constexpr (perChannelScale) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(quantScale, quantScaleLocal + i);
            }
            if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
                RegTensor<T1> xTmp;
                RegTensor<half> yhEven;
                RegTensor<half> yhOdd;
                RegTensor<half> yh;
                RegTensor<int8_t> yi8;
                RegTensor<int8_t> yi8Pack;
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopyUnAlign(xTmp, uSrc, xLocal, VL);
                    Cast<float, T1, castTraitB162B32Even>(xEven, xTmp, pregMain);
                    Cast<float, T1, castTraitB162B32Odd>(xOdd, xTmp, pregMain);
                    if constexpr (activateSilu) {
                        VFInnerNormalizeAndSwish<hasGamma, hasBeta>(xEven, mean, rstd, gamma, beta, yEven, pregMain);
                        VFInnerNormalizeAndSwish<hasGamma, hasBeta>(xOdd, mean, rstd, gamma, beta, yOdd, pregMain);
                    } else {
                        VFInnerNormalize<hasGamma, hasBeta>(xEven, mean, rstd, gamma, beta, yEven, pregMain);
                        VFInnerNormalize<hasGamma, hasBeta>(xOdd, mean, rstd, gamma, beta, yOdd, pregMain);
                    }
                    Div(yEven, yEven, quantScale, pregMain);
                    Div(yOdd, yOdd, quantScale, pregMain);
                    Cast<half, float, castTraitFp322Fp16>(yhEven, yEven, pregMain);
                    Cast<half, float, castTraitFp322Fp16Odd>(yhOdd, yOdd, pregMain);
                    Or((RegTensor<int16_t>&)yh, (RegTensor<int16_t>&)yhEven, (RegTensor<int16_t>&)yhOdd, pregMain);
                    Cast<int8_t, half, castTraitFp162Int8>(yi8, yh, pregMain);
                    // fp16->int8 窄化后 int8 落在偶字节 lane(uint16 步长), 须 Pack 压实成连续再非对齐写(对齐 CANN
                    // StoreUnAlignOneTensor idiom)
                    Pack((RegTensor<uint8_t>&)yi8Pack, (RegTensor<uint16_t>&)yi8);
                    DataCopyUnAlign(yLocal, yi8Pack, uDst, VL);
                }
                uint32_t sreg0 = tailNum;
                for (uint16_t k = 0; k < tailLoop; k++) {
                    pregLoop = UpdateMask<half>(sreg0);
                    DataCopyUnAlign(xTmp, uSrc, xLocal, tailNum);
                    Cast<float, T1, castTraitB162B32Even>(xEven, xTmp, pregLoop);
                    Cast<float, T1, castTraitB162B32Odd>(xOdd, xTmp, pregLoop);
                    if constexpr (activateSilu) {
                        VFInnerNormalizeAndSwish<hasGamma, hasBeta>(xEven, mean, rstd, gamma, beta, yEven, pregLoop);
                        VFInnerNormalizeAndSwish<hasGamma, hasBeta>(xOdd, mean, rstd, gamma, beta, yOdd, pregLoop);
                    } else {
                        VFInnerNormalize<hasGamma, hasBeta>(xEven, mean, rstd, gamma, beta, yEven, pregLoop);
                        VFInnerNormalize<hasGamma, hasBeta>(xOdd, mean, rstd, gamma, beta, yOdd, pregLoop);
                    }
                    Div(yEven, yEven, quantScale, pregLoop);
                    Div(yOdd, yOdd, quantScale, pregLoop);
                    Cast<half, float, castTraitFp322Fp16>(yhEven, yEven, pregLoop);
                    Cast<half, float, castTraitFp322Fp16Odd>(yhOdd, yOdd, pregLoop);
                    Or((RegTensor<int16_t>&)yh, (RegTensor<int16_t>&)yhEven, (RegTensor<int16_t>&)yhOdd, pregLoop);
                    Cast<int8_t, half, castTraitFp162Int8>(yi8, yh, pregLoop);
                    Pack((RegTensor<uint8_t>&)yi8Pack, (RegTensor<uint16_t>&)yi8);
                    DataCopyUnAlign(yLocal, yi8Pack, uDst, tailNum);
                }
                DataCopyUnAlignPost(yLocal, uDst, 0);
            } else {
                // fp32 输入路径(GNSQ 的 x 恒为 b16, 此分支为通用性保留): 逐 VL_FP32 归一化+silu+量化
                RegTensor<half> yh;
                RegTensor<int8_t> yi8;
                for (uint16_t j = 0; j < loopCount; j++) {
                    DataCopyUnAlign(x, uSrc, xLocal, VL_FP32);
                    if constexpr (activateSilu) {
                        VFInnerNormalizeAndSwish<hasGamma, hasBeta>(x, mean, rstd, gamma, beta, y, pregMain);
                    } else {
                        VFInnerNormalize<hasGamma, hasBeta>(x, mean, rstd, gamma, beta, y, pregMain);
                    }
                    Div(y, y, quantScale, pregMain);
                    Cast<half, float, castTraitFp322Fp16>(yh, y, pregMain);
                    Cast<int8_t, half, castTraitFp162Int8>(yi8, yh, pregMain);
                    DataCopyUnAlign(yLocal, yi8, uDst, VL_FP32);
                }
                uint32_t sreg0 = tailNum;
                for (uint16_t k = 0; k < tailLoop; k++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    DataCopyUnAlign(x, uSrc, xLocal, tailNum);
                    if constexpr (activateSilu) {
                        VFInnerNormalizeAndSwish<hasGamma, hasBeta>(x, mean, rstd, gamma, beta, y, pregLoop);
                    } else {
                        VFInnerNormalize<hasGamma, hasBeta>(x, mean, rstd, gamma, beta, y, pregLoop);
                    }
                    Div(y, y, quantScale, pregLoop);
                    Cast<half, float, castTraitFp322Fp16>(yh, y, pregLoop);
                    Cast<int8_t, half, castTraitFp162Int8>(yi8, yh, pregLoop);
                    DataCopyUnAlign(yLocal, yi8, uDst, tailNum);
                }
                DataCopyUnAlignPost(yLocal, uDst, 0);
            }
        }
    }
}

// GroupNormSiluQuant 量化版 Fold(小 hwNum 融合, 对齐). 镜像 VFInnerNormalizeAndSwishFold,
// StoreOutputData->StoreQuantData。 per-tensor: 广播单标量 quantScale; per-channel: 按元素 load(要求 quantScaleLocal 与
// gamma 同 fold 布局, 见 kernel 侧注)。
template <typename T1, typename T2, bool activateSilu, bool hasGamma, bool hasBeta, bool perChannelScale>
__aicore__ inline void VFInnerNormalizeAndSwishQuantFold(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                                         __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                                         __local_mem__ float* rstdLocal,
                                                         __local_mem__ float* quantScaleLocal,
                                                         __local_mem__ int8_t* yLocal, uint16_t groupNums,
                                                         uint16_t rowsCount, int32_t reduceCount)
{
    uint16_t loopCountFo = CeilDiv(reduceCount, VL_FP32);
    uint32_t reduceCountAlignFo = RoundUp<T1>(reduceCount);
    // int8 yFo 的每组行距须按 32B(int8 的 block)对齐:落盘 CopySilu2Gm 的 DataCopyPad 以 32B 块粒度读,
    // 若沿用 reduceCountAlignFo(=RoundUp<half>=16B 倍数, 非 32B 倍数)存 yFo, 小 reduceCount 时会被 32B
    // 块读跨组错位/越界。 xFo/gammaFo/betaFo/quantScaleFo 仍用 reduceCountAlignFo(half/float, 本就 32B 对齐), 仅 yFo
    // 用独立 32B 行距。
    uint32_t yRowStrideFo = RoundUp<int8_t>(reduceCount);
    __VEC_SCOPE__
    {
        RegTensor<float> xFo;
        RegTensor<float> gammaFo;
        RegTensor<float> betaFo;
        RegTensor<float> meanFo;
        RegTensor<float> rstdFo;
        RegTensor<float> yFo;
        RegTensor<float> quantScaleFo;
        MaskReg pregLoopFo;
        if constexpr (!perChannelScale) {
            DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(quantScaleFo, quantScaleLocal);
        }
        for (uint16_t iFo = 0; iFo < groupNums; iFo++) {
            for (uint16_t jFo = 0; jFo < rowsCount; jFo++) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstdFo, rstdLocal + iFo * rowsCount + jFo);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(meanFo, meanLocal + iFo * rowsCount + jFo);
                uint32_t sreg0Fo = reduceCount;
                for (uint16_t kFo = 0; kFo < loopCountFo; kFo++) {
                    pregLoopFo = UpdateMask<float>(sreg0Fo);
                    LoadInputData<T1>(xFo, xLocal, pregLoopFo,
                                      iFo * rowsCount * reduceCountAlignFo + jFo * reduceCountAlignFo + kFo * VL_FP32);
                    if constexpr (hasGamma) {
                        LoadInputData<T2>(gammaFo, gammaLocal, pregLoopFo, jFo * reduceCountAlignFo + kFo * VL_FP32);
                    }
                    if constexpr (hasBeta) {
                        LoadInputData<T2>(betaFo, betaLocal, pregLoopFo, jFo * reduceCountAlignFo + kFo * VL_FP32);
                    }
                    if constexpr (perChannelScale) {
                        LoadInputData<float>(quantScaleFo, quantScaleLocal, pregLoopFo,
                                             jFo * reduceCountAlignFo + kFo * VL_FP32);
                    }
                    if constexpr (activateSilu) {
                        VFInnerNormalizeAndSwish<hasGamma, hasBeta>(xFo, meanFo, rstdFo, gammaFo, betaFo, yFo,
                                                                    pregLoopFo);
                    } else {
                        VFInnerNormalize<hasGamma, hasBeta>(xFo, meanFo, rstdFo, gammaFo, betaFo, yFo, pregLoopFo);
                    }
                    StoreQuantData(yLocal, yFo, quantScaleFo, pregLoopFo,
                                   (iFo * rowsCount + jFo) * yRowStrideFo + kFo * VL_FP32);
                }
            }
        }
    }
}

// GroupNormSiluQuant 量化派发(align 路径): 按 perChannelScale 展开, 再展开 activateSilu/hasGamma/hasBeta。
template <typename T1, typename T2, bool perChannelScale>
__aicore__ inline void VFNormalizeAndSwishQuantAlignImpl(
    __local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
    __local_mem__ float* rstdLocal, __local_mem__ float* quantScaleLocal, __local_mem__ int8_t* yLocal,
    uint16_t rowsCount, int32_t reduceCount, bool activateSilu, bool hasGamma, bool hasBeta)
{
    if (activateSilu) {
        if (hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, true, true, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (hasGamma && !hasBeta) {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, true, true, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (!hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, true, false, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, true, false, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        }
    } else {
        if (hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, false, true, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (hasGamma && !hasBeta) {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, false, true, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (!hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, false, false, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else {
            VFInnerNormalizeAndSwishQuantAlign<T1, T2, false, false, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeAndSwishQuantAlign(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                                     __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                                     __local_mem__ float* rstdLocal,
                                                     __local_mem__ float* quantScaleLocal, __local_mem__ int8_t* yLocal,
                                                     uint16_t rowsCount, int32_t reduceCount, bool activateSilu,
                                                     bool hasGamma, bool hasBeta, bool perChannelScale)
{
    if (perChannelScale) {
        VFNormalizeAndSwishQuantAlignImpl<T1, T2, true>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                        quantScaleLocal, yLocal, rowsCount, reduceCount, activateSilu,
                                                        hasGamma, hasBeta);
    } else {
        VFNormalizeAndSwishQuantAlignImpl<T1, T2, false>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                         quantScaleLocal, yLocal, rowsCount, reduceCount, activateSilu,
                                                         hasGamma, hasBeta);
    }
}

// GroupNormSiluQuant 量化派发(UnAlign 路径): 按 perChannelScale 展开, 再展开 activateSilu/hasGamma/hasBeta。
template <typename T1, typename T2, bool perChannelScale>
__aicore__ inline void VFNormalizeAndSwishQuantUnAlignImpl(
    __local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
    __local_mem__ float* rstdLocal, __local_mem__ float* quantScaleLocal, __local_mem__ int8_t* yLocal,
    uint16_t rowsCount, int32_t reduceCount, bool activateSilu, bool hasGamma, bool hasBeta)
{
    if (activateSilu) {
        if (hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, true, true, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (hasGamma && !hasBeta) {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, true, true, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (!hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, true, false, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, true, false, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        }
    } else {
        if (hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, false, true, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (hasGamma && !hasBeta) {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, false, true, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else if (!hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, false, false, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        } else {
            VFInnerNormalizeAndSwishQuantUnAlign<T1, T2, false, false, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, rowsCount, reduceCount);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeAndSwishQuantUnAlign(
    __local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
    __local_mem__ float* rstdLocal, __local_mem__ float* quantScaleLocal, __local_mem__ int8_t* yLocal,
    uint16_t rowsCount, int32_t reduceCount, bool activateSilu, bool hasGamma, bool hasBeta, bool perChannelScale)
{
    if (perChannelScale) {
        VFNormalizeAndSwishQuantUnAlignImpl<T1, T2, true>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                          quantScaleLocal, yLocal, rowsCount, reduceCount, activateSilu,
                                                          hasGamma, hasBeta);
    } else {
        VFNormalizeAndSwishQuantUnAlignImpl<T1, T2, false>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                           quantScaleLocal, yLocal, rowsCount, reduceCount,
                                                           activateSilu, hasGamma, hasBeta);
    }
}

// GroupNormSiluQuant 量化派发(Fold 路径): 按 perChannelScale 展开, 再展开 activateSilu/hasGamma/hasBeta。
template <typename T1, typename T2, bool perChannelScale>
__aicore__ inline void VFNormalizeAndSwishQuantFoldImpl(
    __local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal, __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
    __local_mem__ float* rstdLocal, __local_mem__ float* quantScaleLocal, __local_mem__ int8_t* yLocal,
    uint16_t groupNums, uint16_t rowsCount, int32_t reduceCount, bool activateSilu, bool hasGamma, bool hasBeta)
{
    if (activateSilu) {
        if (hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, true, true, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        } else if (hasGamma && !hasBeta) {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, true, true, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        } else if (!hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, true, false, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        } else {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, true, false, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        }
    } else {
        if (hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, false, true, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        } else if (hasGamma && !hasBeta) {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, false, true, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        } else if (!hasGamma && hasBeta) {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, false, false, true, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        } else {
            VFInnerNormalizeAndSwishQuantFold<T1, T2, false, false, false, perChannelScale>(
                xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal, quantScaleLocal, yLocal, groupNums, rowsCount,
                reduceCount);
        }
    }
}

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeAndSwishQuantFold(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                                    __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                                    __local_mem__ float* rstdLocal,
                                                    __local_mem__ float* quantScaleLocal, __local_mem__ int8_t* yLocal,
                                                    uint16_t groupNums, uint16_t rowsCount, int32_t reduceCount,
                                                    bool activateSilu, bool hasGamma, bool hasBeta,
                                                    bool perChannelScale)
{
    if (perChannelScale) {
        VFNormalizeAndSwishQuantFoldImpl<T1, T2, true>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                       quantScaleLocal, yLocal, groupNums, rowsCount, reduceCount,
                                                       activateSilu, hasGamma, hasBeta);
    } else {
        VFNormalizeAndSwishQuantFoldImpl<T1, T2, false>(xLocal, gammaLocal, betaLocal, meanLocal, rstdLocal,
                                                        quantScaleLocal, yLocal, groupNums, rowsCount, reduceCount,
                                                        activateSilu, hasGamma, hasBeta);
    }
}

template <typename T>
__aicore__ inline void CopyGammaAndBeta2UB(const GlobalTensor<T>& gammaGm, const GlobalTensor<T>& betaGm,
                                           const LocalTensor<T>& gammaTensor, const LocalTensor<T>& betaTensor,
                                           const uint16_t blockCount, const uint32_t copyLen, bool hasGamma = true,
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

// GroupNormSiluQuant: 把 quantScale(fp32, 长 shapeQuantScale=1 或 shapeC)搬入 UB。
__aicore__ inline void CopyQuantScale2UB(const GlobalTensor<float>& quantScaleGm,
                                         const LocalTensor<float>& quantScaleTensor, const uint32_t copyLen)
{
    int32_t copyLenAlign = RoundUp<float>(copyLen);
    DataCopyPadExtParams<float> padParams;
    padParams.isPad = true;
    padParams.paddingValue = 0.0f;
    padParams.rightPadding = copyLenAlign - copyLen;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = copyLen * sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(quantScaleTensor, quantScaleGm, dataCopyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyGammaAndBeta2UBByNDDMA(const GlobalTensor<T>& gammaGm, const GlobalTensor<T>& betaGm,
                                                  const LocalTensor<T>& gammaTensor, const LocalTensor<T>& betaTensor,
                                                  const uint16_t numGroups, const uint32_t shapeD, const uint16_t hwNum,
                                                  const uint32_t eleNumAlign, bool hasGamma = true, bool hasBeta = true)
{
    MultiCopyLoopInfo<GAMMA_BETA_UB_DIM> loopInfo;
    loopInfo.loopSize[INDEX_0] = numGroups;
    loopInfo.loopSrcStride[INDEX_0] = shapeD;
    loopInfo.loopDstStride[INDEX_0] = eleNumAlign;

    loopInfo.loopSize[INDEX_1] = shapeD;
    loopInfo.loopSrcStride[INDEX_1] = 1;
    loopInfo.loopDstStride[INDEX_1] = hwNum;

    loopInfo.loopSize[INDEX_2] = hwNum;
    loopInfo.loopSrcStride[INDEX_2] = 0;
    loopInfo.loopDstStride[INDEX_2] = 1;

    T constValue = 0;
    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T, GAMMA_BETA_UB_DIM> paramsMain = {loopInfo, constValue};

    if (hasGamma) {
        DataCopy<T, GAMMA_BETA_UB_DIM, config>(gammaTensor, gammaGm, paramsMain);
    }
    if (hasBeta) {
        DataCopy<T, GAMMA_BETA_UB_DIM, config>(betaTensor, betaGm, paramsMain);
    }
}

// GroupNormSiluQuant per-channel + fold 专用: 把 quantScale(per-channel, shapeC=numGroups*shapeD 个 fp32)
// 按 fold 布局 NDDMA 复制成 [numGroups × eleNumAlign](每通道值沿 hwNum 重复), 与 gamma 的 fold 布局一致。
__aicore__ inline void CopyQuantScale2UBByNDDMA(const GlobalTensor<float>& quantScaleGm,
                                                const LocalTensor<float>& quantScaleTensor, const uint16_t numGroups,
                                                const uint32_t shapeD, const uint16_t hwNum, const uint32_t eleNumAlign)
{
    MultiCopyLoopInfo<GAMMA_BETA_UB_DIM> loopInfoQ;
    loopInfoQ.loopSize[INDEX_0] = numGroups;
    loopInfoQ.loopSrcStride[INDEX_0] = shapeD;
    loopInfoQ.loopDstStride[INDEX_0] = eleNumAlign;

    loopInfoQ.loopSize[INDEX_1] = shapeD;
    loopInfoQ.loopSrcStride[INDEX_1] = 1;
    loopInfoQ.loopDstStride[INDEX_1] = hwNum;

    loopInfoQ.loopSize[INDEX_2] = hwNum;
    loopInfoQ.loopSrcStride[INDEX_2] = 0;
    loopInfoQ.loopDstStride[INDEX_2] = 1;

    float constValueQ = 0;
    static constexpr MultiCopyConfig configQ = {false};
    MultiCopyParams<float, GAMMA_BETA_UB_DIM> paramsMainQ = {loopInfoQ, constValueQ};
    DataCopy<float, GAMMA_BETA_UB_DIM, configQ>(quantScaleTensor, quantScaleGm, paramsMainQ);
}

template <typename T>
__aicore__ inline void CopyX2UB(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor,
                                const uint16_t blockCount, const uint32_t copyLen, const uint32_t dstRowStride = 0)
{
    int32_t copyLenAlign = RoundUp<T>(copyLen);
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.paddingValue = static_cast<T>(0.0);
    padParams.rightPadding = copyLenAlign - copyLen;

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    // dstRowStride==0: UB 各行按 copyLenAlign 紧排(默认)。
    // 否则各行须落在 dstRowStride(如 VL_FP32 对齐的 hwNumAlign),供后续向量按 dstRowStride 步长读取,
    // 逐行拷入以保证行起始偏移严格对齐,避免 RoundUp<T> 与读取步长不一致导致的错读/越界。
    if (dstRowStride == 0) {
        dataCopyParams.blockCount = blockCount;
        DataCopyPad(inputTensor, inputGm, dataCopyParams, padParams);
    } else {
        dataCopyParams.blockCount = 1;
        for (uint16_t r = 0; r < blockCount; r++) {
            DataCopyPad(inputTensor[r * dstRowStride], inputGm[r * copyLen], dataCopyParams, padParams);
        }
    }
}

template <typename T>
__aicore__ inline void CopyMeanAndRstd2Gm(const GlobalTensor<T>& meanGm, const GlobalTensor<T>& rstdGm,
                                          const LocalTensor<T>& meanTensor, const LocalTensor<T>& rstdTensor,
                                          const uint16_t blockCount, const uint32_t copyLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(meanGm, meanTensor, dataCopyParams);
    DataCopyPad(rstdGm, rstdTensor, dataCopyParams);
}

template <typename T>
__aicore__ inline void ProcessMeanAndRstd(TQue<TPosition::VECOUT, 1>& outQue, const GlobalTensor<T>& tensorGm,
                                          const DataCopyExtParams& dataCopyParams, const uint32_t& dataLen,
                                          const T fillValue)
{
    LocalTensor<T> tensorUb = outQue.AllocTensor<T>();
    Duplicate(tensorUb, fillValue, dataLen);
    outQue.EnQue(tensorUb);
    tensorUb = outQue.DeQue<T>();
    DataCopyPad(tensorGm, tensorUb, dataCopyParams);
    outQue.FreeTensor(tensorUb);
}

template <typename T1>
__aicore__ inline void ProcessMeanAndRstd(LocalTensor<float>& meanTensor, LocalTensor<T1>& meanOutTensor,
                                          GlobalTensor<T1>& meanGm, LocalTensor<float>& rstdTensor,
                                          LocalTensor<T1>& rstdOutTensor, GlobalTensor<T1>& rstdGm, uint64_t gmOffset,
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
                DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(meanOutLocal + i * VL_FP32, meanOut,
                                                                          pregLoop);
                DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(rstdOutLocal + i * VL_FP32, rstdOut,
                                                                          pregLoop);
            }
        }
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        CopyMeanAndRstd2Gm<T1>(meanGm[gmOffset], rstdGm[gmOffset], meanOutTensor, rstdOutTensor, 1, curNumPerCore);
    }
}

template <typename T>
__aicore__ inline void CopySilu2Gm(const GlobalTensor<T>& siluGm, const LocalTensor<T>& yTensor, uint16_t blockCount,
                                   const uint32_t copyLen, const uint32_t srcRowStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.dstStride = 0;
    dataCopyParams.srcStride = 0;
    // srcRowStride==0 或 ==copyLen: UB 内各行连续,一次拷 blockCount 行。
    // 否则 UB 行按 srcRowStride 对齐存放(如 int8 量化按 hwNumAlign),其与 copyLen 的 gap 非 32B,
    // DataCopyPad 的 srcStride(按 32B 块)无法表达,故逐行从对齐偏移拷 copyLen 到连续 GM。
    if (srcRowStride == 0 || srcRowStride == copyLen) {
        dataCopyParams.blockCount = blockCount;
        DataCopyPad(siluGm, yTensor, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        for (uint16_t r = 0; r < blockCount; r++) {
            DataCopyPad(siluGm[r * copyLen], yTensor[r * srcRowStride], dataCopyParams);
        }
    }
}

// 常规版(非泛化)byChannel/byHW/two-pass 双缓冲流水公用的 4 对 ping/pong 事件。
// 仅把逐字重复的 alloc/release 样板抽出: 事件种类/数量/配对/分配与释放顺序与原各处完全一致, 零功能变化。
struct PingPongEvents {
    event_t mte2ToVPing;
    event_t mte2ToVPong;
    event_t vToMte3Ping;
    event_t vToMte3Pong;
    event_t vToMte2Ping;
    event_t vToMte2Pong;
    event_t mte3ToVPing;
    event_t mte3ToVPong;
};

__aicore__ inline PingPongEvents AllocPingPongEvents()
{
    PingPongEvents e;
    e.mte2ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    e.mte2ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    e.vToMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    e.vToMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    e.vToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    e.vToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    e.mte3ToVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    e.mte3ToVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    return e;
}

__aicore__ inline void ReleasePingPongEvents(const PingPongEvents& e)
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(e.mte2ToVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(e.mte2ToVPong);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(e.vToMte3Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(e.vToMte3Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(e.vToMte2Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(e.vToMte2Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(e.mte3ToVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(e.mte3ToVPong);
}

// CRTP 公共基类: 抽取 4 个载入模板(R partial/full × 常规/泛化)跨族共用的 GM SetGlobalBuffer(Init)
// 与 Process(mean/rstd 落盘调度)。差异逻辑 ParseTilingData/InitInnerBuffer/CalNormalize 由派生类实现,
// 经 static_cast<Derived*> 回调; 派生类 GM/成员经同一 downcast 访问(friend 授权)。功能与原各族逐字等价。
template <typename Derived, typename T1, typename T2>
class GroupNormSiluQuantLoadBase {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale, GM_ADDR silu, GM_ADDR mean,
                                GM_ADDR rstd, GM_ADDR workspace, const GroupNormSiluQuantRegbaseTilingData* tilingData)
    {
        Derived* d = static_cast<Derived*>(this);
        d->tiling = tilingData;
        d->blockIdx = GetBlockIdx();
        d->blockNum = GetBlockNum();
        d->xGm.SetGlobalBuffer((__gm__ T1*)x);
        if (gamma != nullptr) {
            d->hasGamma = true;
            d->gammaGm.SetGlobalBuffer((__gm__ T2*)gamma);
        }
        if (beta != nullptr) {
            d->hasBeta = true;
            d->betaGm.SetGlobalBuffer((__gm__ T2*)beta);
        }
        d->quantScaleGm.SetGlobalBuffer((__gm__ float*)quantScale);
        d->siluGm.SetGlobalBuffer((__gm__ int8_t*)silu);
        if constexpr (sizeof(T2) == sizeof(float)) {
            d->meanT2Gm.SetGlobalBuffer((__gm__ T2*)mean);
            d->rstdT2Gm.SetGlobalBuffer((__gm__ T2*)rstd);
        } else {
            d->meanGm.SetGlobalBuffer((__gm__ T1*)mean);
            d->rstdGm.SetGlobalBuffer((__gm__ T1*)rstd);
        }
        d->ParseTilingData();
        d->InitInnerBuffer();
    }

    __aicore__ inline void Process()
    {
        Derived* d = static_cast<Derived*>(this);
        uint32_t numPerCoreLoop = CeilDiv(d->numPerCore, d->innerNumPerCore);
        uint32_t numPerCoreTail = d->numPerCore % d->innerNumPerCore == 0 ? d->innerNumPerCore :
                                                                            d->numPerCore % d->innerNumPerCore;
        uint32_t numPerCoreOneLoop = d->innerNumPerCore;
        event_t eventIDMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        for (uint64_t i = 0; i < numPerCoreLoop; i++) {
            if (i > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
            if (i == numPerCoreLoop - 1) {
                numPerCoreOneLoop = numPerCoreTail;
            }
            d->CalNormalize(i * d->innerNumPerCore, numPerCoreOneLoop);
            uint64_t gmOffset = d->blockIdx * d->tiling->numPerCore + i * d->innerNumPerCore;
            if constexpr (sizeof(T2) == sizeof(float)) {
                LocalTensor<T2> meanOutTensorNew = d->meanOutTensor.template ReinterpretCast<T2>();
                LocalTensor<T2> rstdOutTensorNew = d->rstdOutTensor.template ReinterpretCast<T2>();
                ProcessMeanAndRstd<T2>(d->meanTensor, meanOutTensorNew, d->meanT2Gm, d->rstdTensor, rstdOutTensorNew,
                                       d->rstdT2Gm, gmOffset, numPerCoreOneLoop);
            } else {
                ProcessMeanAndRstd<T1>(d->meanTensor, d->meanOutTensor, d->meanGm, d->rstdTensor, d->rstdOutTensor,
                                       d->rstdGm, gmOffset, numPerCoreOneLoop);
            }
            if (i < numPerCoreLoop - 1) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3ToMte2);
            }
        }
    }

    // 两族 ParseTilingData 共用的 tiling 解析(numPerCore/numGroups/shape/eps/dichotomy 等), 与原逐字等价。
    // 族内专属项(welford/onceNumPerCore/isFold 等)仍由派生类 ParseTilingData 在调用本函数后补齐。
    __aicore__ inline void ParseCommonTilingData()
    {
        Derived* d = static_cast<Derived*>(this);
        if (d->blockIdx == d->blockNum - 1) {
            d->numPerCore = d->tiling->numLastCore;
        } else {
            d->numPerCore = d->tiling->numPerCore;
        }
        d->numGroups = d->tiling->numGroups;
        d->ubSize = d->tiling->ubSize;
        d->elemNum = d->tiling->elemNum;
        d->shapeC = d->tiling->shapeC;
        d->shapeD = d->tiling->shapeD;
        d->shapeQuantScale = d->tiling->shapeQuantScale;
        d->perChannelScale = (d->shapeQuantScale != 1);
        d->eps = d->tiling->epsilon;
        d->activateSilu = d->tiling->activateSilu;
        d->hwNum = d->tiling->hwNum;
        d->processSize = d->tiling->processSize;
        d->dichotomyAddPower = d->tiling->dichotomyAddPower;
        d->dichotomyAddK = d->tiling->dichotomyAddK;
        d->dichotomyAddLastNum = d->tiling->dichotomyAddLastNum;
    }
};

} // namespace GroupNormSiluQuant

#endif
