/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal continuation of group_norm_regbase_base.h. */
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
    RegTensor<float> mean;
    RegTensor<float> sumVar;
    RegTensor<float> var;
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
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
        DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
        Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
        Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
        ReduceSum(mean, sumMean, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
    }

    // 剩余整块需要拆分成多部分
    // 整块剩余部分回刷UB，整块使用tailCountScale
    for (uint16_t i = 0; i < welfordDiffLoopCount; i++) {
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddReminderRoundUp);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
        ReduceSum(mean, dichotomyAddMeanL, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + dichotomyAddReminderLoopCount + i, mean, pregMerge);
    }

    sreg0 = welfordDiffReminder;
    for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
        pregLoop = UpdateMask<float>(sreg0);
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
        Muls(tmp, dichotomyAddMeanL, coeff, pregLoop);
        Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop);
        ReduceSum(mean, dichotomyAddMeanL, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + i, mean, pregMerge);
    }

    for (uint16_t i = 0; i < dichotomyAddPowerRemainLoopCount; i++) {
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32 + dichotomyAddPowerOffset);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
        ReduceSum(mean, dichotomyAddMeanL, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
            mean, pregMerge);
    }

    NormCommon::DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
    DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

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
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
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
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + dichotomyAddReminderLoopCount + i, var, pregMerge);
    }

    sreg0 = welfordDiffReminder;
    for (uint16_t i = 0; i < welfordReminderLoopCount; i++) {
        pregLoop = UpdateMask<float>(sreg0);
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
        Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
        Mul(deltaL, deltaL, deltaL, pregMain);
        Muls(deltaL, deltaL, cnt, pregMain);
        Muls(tmp, deltaL, coeff, pregLoop);
        Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop);
        DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddReminderRoundUp);
        Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
        Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
        ReduceSum(var, dichotomyAddVarL, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
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
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + dichotomyAddReminderLoopCount + welfordDiffLoopCount + welfordReminderLoopCount + i,
            var, pregMerge);
    }

    NormCommon::DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
    StoreStatisticData<T>(varianceOutLocal, var, pregMerge, offset);
    NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
    DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
}
}

template <typename T>
__aicore__ inline void VFWelfordParallelFinalizeNonAlign(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
    __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
    uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
    uint32_t offset, uint32_t tailSize, float reduceScale, float cnt, float eps)
{
    // 非对齐Welford finalize阶段由于自身存在整尾块，二分折叠存在整尾块，会出现多种不同的场景，每个场景都有独立的VF
    uint32_t dichotomyAddReminder = reduceCount - dichotomyAddPower;
    uint32_t dichotomyAddReminderRoundUp = CeilDiv(dichotomyAddReminder, VL_FP32) * VL_FP32;
    if (tailSize >= dichotomyAddPower) {
        VFWelfordParallelFinalizeNonAlignSituation1(meanLocal, rstdLocal, varianceOutLocal, tmpMeanLocal, tmpVarLocal,
                                                    dichotomyAddLocal, reduceCount, dichotomyAddPower, dichotomyAddK,
                                                    dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    if (tailSize <= dichotomyAddReminderRoundUp) {
        VFWelfordParallelFinalizeNonAlignSituation2(meanLocal, rstdLocal, varianceOutLocal, tmpMeanLocal, tmpVarLocal,
                                                    dichotomyAddLocal, reduceCount, dichotomyAddPower, dichotomyAddK,
                                                    dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
        return;
    }
    VFWelfordParallelFinalizeNonAlignSituation3(meanLocal, rstdLocal, varianceOutLocal, tmpMeanLocal, tmpVarLocal,
                                                dichotomyAddLocal, reduceCount, dichotomyAddPower, dichotomyAddK,
                                                dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
}

template <typename T>
__aicore__ inline void VFWelfordParallelFinalize(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
    __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
    uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
    uint32_t offset, uint32_t tailSize, float reduceScale, float scale, float cnt, float eps, bool welfordAlign)
{
    if (welfordAlign) {
        VFWelfordParallelFinalizeAlign(meanLocal, rstdLocal, varianceOutLocal, tmpMeanLocal, tmpVarLocal,
                                       dichotomyAddLocal, reduceCount, dichotomyAddPower, dichotomyAddK,
                                       dichotomyAddLastNum, offset, reduceScale, scale, cnt, eps);
    } else {
        cnt = cnt - 1;
        VFWelfordParallelFinalizeNonAlign(meanLocal, rstdLocal, varianceOutLocal, tmpMeanLocal, tmpVarLocal,
                                          dichotomyAddLocal, reduceCount, dichotomyAddPower, dichotomyAddK,
                                          dichotomyAddLastNum, offset, tailSize, reduceScale, cnt, eps);
    }
}

template <typename T>
__aicore__ inline void CalMeanAndRstdByDichotomyAdd(__local_mem__ T* xLocal, __local_mem__ float* meanLocal,
                                                    __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
                                                    __local_mem__ float* dichotomyAddLocal, uint16_t numPerCoreProcess,
                                                    uint32_t dichotomyAddPower, uint32_t dichotomyAddK,
                                                    uint32_t dichotomyAddLastNum, uint64_t reduceCount, float scale,
                                                    float eps)
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
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
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
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + j, mean,
                                                                                 pregMerge);
            }

            // 整块剩余部分vcadd回刷UB
            for (uint16_t j = 0; j < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 j++) {
                LoadInputData<T>(dichotomyAddL, xLocal, pregMain,
                                 i * elemNumAlign + (j + dichotomyAddReminderLoopCount) * VL_FP32);
                Muls(dichotomyAddL, dichotomyAddL, scale, pregMain);
                ReduceSum(mean, dichotomyAddL, pregMain);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + j, mean, pregMerge);
            }

            NormCommon::DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + i, mean, pregMerge);
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
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + j, var, pregMerge);
            }

            // 整块剩余部分vcadd回刷UB
            for (uint16_t j = 0; j < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderLoopCount);
                 j++) {
                LoadInputData<T>(dichotomyAddL, xLocal, pregMain,
                                 i * elemNumAlign + (j + dichotomyAddReminderLoopCount) * VL_FP32);
                Sub(dichotomyAddL, dichotomyAddL, mean, pregMain);
                Mul(dichotomyAddL, dichotomyAddL, dichotomyAddL, pregMain);
                Muls(dichotomyAddL, dichotomyAddL, scale, pregMain);
                ReduceSum(var, dichotomyAddL, pregMain);
                DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    dichotomyAddLocal + dichotomyAddReminderLoopCount + j, var, pregMerge);
            }
            NormCommon::DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
            StoreStatisticData<T>(varianceOutLocal, var, pregMerge, i);
            NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + i, rstd, pregMerge);
        }
    }
}

// R轴小于64
template <typename T>
__aicore__ inline void CalMeanAndRstdSpecial(__local_mem__ T* xLocal, __local_mem__ float* meanLocal,
                                             __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
                                             uint16_t numPerCoreProcess, uint64_t reduceCount, float scale, float eps)
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
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
        Duplicate(one, float(1.0), pregMain);
        for (uint16_t i = 0; i < numPerCoreProcess; i++) {
            uint32_t sreg0 = reduceCount;
            pregLoop = UpdateMask<float>(sreg0);
            LoadInputData<T>(x, xLocal, pregLoop, i * elemNumAlign);
            Muls(xScale, x, scale, pregLoop);
            ReduceSum(mean, xScale, pregLoop);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + i, mean, pregMerge);

            Duplicate(mean, mean, pregMain);
            Sub(x, x, mean, pregLoop);
            Mul(x, x, x, pregLoop);
            Muls(xScale, x, scale, pregLoop);
            ReduceSum(var, xScale, pregLoop);
            StoreStatisticData<T>(varianceOutLocal, var, pregMerge, i);
            NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + i, rstd, pregMerge);
        }
    }
}

template <typename T>
__aicore__ inline void CalMeanAndRstd(__local_mem__ T* xLocal, __local_mem__ float* meanLocal,
                                      __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
                                      __local_mem__ float* dichotomyAddLocal, uint16_t numPerCoreProcess,
                                      uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
                                      uint64_t reduceCount, float scale, float eps)
{
    if (dichotomyAddPower >= VL_FP32) {
        CalMeanAndRstdByDichotomyAdd(xLocal, meanLocal, rstdLocal, varianceOutLocal, dichotomyAddLocal,
                                     numPerCoreProcess, dichotomyAddPower, dichotomyAddK, dichotomyAddLastNum,
                                     reduceCount, scale, eps);
        return;
    }
    CalMeanAndRstdSpecial(xLocal, meanLocal, rstdLocal, varianceOutLocal, numPerCoreProcess, reduceCount, scale, eps);
}

__aicore__ inline void VFInnerNormalize(RegTensor<float>& x, RegTensor<float>& mean, RegTensor<float>& rstd,
                                        RegTensor<float>& gamma, RegTensor<float>& beta, RegTensor<float>& y,
                                        MaskReg& preg)
{
    Sub(x, x, mean, preg);
    Mul(x, x, rstd, preg);
    Mul(x, x, gamma, preg);
    Add(y, x, beta, preg);
}

template <typename T1, typename T2>
__aicore__ inline void VFNormalizeUnAlign(__local_mem__ T1* xLocal, __local_mem__ T2* gammaLocal,
                                          __local_mem__ T2* betaLocal, __local_mem__ float* meanLocal,
                                          __local_mem__ float* rstdLocal, __local_mem__ T1* yLocal, uint32_t rowsCount,
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
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<T1, AscendC::Reg::MaskPattern::ALL>();

        UnalignReg uSrc;
        UnalignReg uDst;
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(rstd, rstdLocal);
        DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(mean, meanLocal);
        DataCopyUnAlignPre<T1>(uSrc, xLocal);
        for (uint16_t i = 0; i < static_cast<uint16_t>(rowsCount); i++) {
            LoadGammaAndBetaData<T2>(gamma, beta, gammaLocal, betaLocal, pregMain, i);
#include "group_norm_regbase_base_part3.h"
