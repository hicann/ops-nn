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
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + i * VL_FP32);
        DataCopy(dichotomyAddMeanR, tmpMeanLocal + i * VL_FP32 + dichotomyAddPower);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
        Muls(dichotomyAddMeanR, dichotomyAddMeanR, tailCountScale, pregMain);
        Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
        ReduceSum(mean, sumMean, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
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
        Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dichotomyAddMeanR, tmp, pregLoop1);
        Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
        ReduceSum(mean, sumMean, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount,
                                                                         mean, pregMerge);
    }

    // 整块使用tailCountScale,尾块使用countScale
    for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
        pregLoop = UpdateMask<float>(sreg0);
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
        DataCopy(dichotomyAddMeanR,
                 tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
        Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
        Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
        ReduceSum(mean, sumMean, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
    }
    // PART2: 整块剩余部分vcadd回刷UB,使用tailCountScale
    for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
         i++) {
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
        Muls(dichotomyAddMeanL, dichotomyAddMeanL, tailCountScale, pregMain);
        ReduceSum(mean, dichotomyAddMeanL, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
    }
    NormCommon::DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
    DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

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
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
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
        Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(deltaR, tmp, pregLoop1);

        DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
        Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
        Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
        DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
        Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
        Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

        Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
        ReduceSum(var, sumVar, pregMain);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i + welfordDiffLoopCount,
                                                                         var, pregMerge);
    }

    for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
        pregLoop = UpdateMask<float>(sreg0);
        DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
        Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
        Mul(deltaL, deltaL, deltaL, pregMain);
        Muls(deltaL, deltaL, tailCnt, pregMain);
        DataCopy(dichotomyAddMeanR,
                 tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign + dichotomyAddPower);
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
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
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
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
            dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
    }
    NormCommon::DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
    StoreStatisticData<T>(varianceOutLocal, var, pregMerge, offset);
    NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
    DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
}
}

// welford整块小于二分累加整块，并且小于等于二分累加尾块向上对齐
template <typename T>
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation2(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
    __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
    uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
    uint32_t offset, uint32_t tailSize, float reduceScale, float cnt, float eps)
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
        RegTensor<float> one;
        RegTensor<float> rstd;
        RegTensor<float> tmp;

        MaskReg pregLoop;
        MaskReg pregLoop1;
        MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
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
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, mean, pregMerge);
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
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(dichotomyAddMeanL, tmp, pregLoop1);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, mean, pregMerge);
        }

        // 整块使用countScale,尾块使用countScale
        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            DataCopy(dichotomyAddMeanR, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign +
                                            dichotomyAddPower);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            Muls(dichotomyAddMeanR, dichotomyAddMeanR, countScale, pregLoop);
            Add(sumMean, dichotomyAddMeanL, dichotomyAddMeanR, pregMain);
            ReduceSum(mean, sumMean, pregMain);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount + welfordReminderLoopCount, mean, pregMerge);
        }
        // PART2: 整块剩余部分vcadd回刷UB,使用countScale
        for (uint16_t i = 0; i < static_cast<uint16_t>(dichotomyAddPowerLoopCount - dichotomyAddReminderRealLoopCount);
             i++) {
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + dichotomyAddReminderRealLoopCount) * VL_FP32);
            Muls(dichotomyAddMeanL, dichotomyAddMeanL, countScale, pregMain);
            ReduceSum(mean, dichotomyAddMeanL, pregMain);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, mean, pregMerge);
        }
        NormCommon::DichotomyAdd(mean, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(meanLocal + offset, mean, pregMerge);

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
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(dichotomyAddLocal + i, var, pregMerge);
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
            Copy<float, AscendC::Reg::MaskMergeMode::MERGING>(deltaL, tmp, pregLoop1);

            DataCopy(dichotomyAddVarL, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32);
            Add(dichotomyAddVarL, dichotomyAddVarL, deltaL, pregMain);
            Muls(dichotomyAddVarL, dichotomyAddVarL, reduceScale, pregMain);
            DataCopy(dichotomyAddVarR, tmpVarLocal + (i + welfordDiffLoopCount) * VL_FP32 + dichotomyAddPower);
            Add(dichotomyAddVarR, dichotomyAddVarR, deltaR, pregLoop);
            Muls(dichotomyAddVarR, dichotomyAddVarR, reduceScale, pregLoop);

            Add(sumVar, dichotomyAddVarL, dichotomyAddVarR, pregMain);
            ReduceSum(var, sumVar, pregMain);
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + i + welfordDiffLoopCount, var, pregMerge);
        }

        for (uint16_t i = 0; i < dichotomyAddReminderLoopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            DataCopy(dichotomyAddMeanL, tmpMeanLocal + (i + welfordDiffLoopCount) * VL_FP32 + welfordDiffReminderAlign);
            Sub(deltaL, dichotomyAddMeanL, mean, pregMain);
            Mul(deltaL, deltaL, deltaL, pregMain);
            Muls(deltaL, deltaL, cnt, pregMain);
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
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
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
            DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
                dichotomyAddLocal + dichotomyAddReminderRealLoopCount + i, var, pregMerge);
        }
        NormCommon::DichotomyAdd(var, dichotomyAddLocal, dichotomyAddK, innerLoopCountOrigin, dichotomyAddLastNum);
        StoreStatisticData<T>(varianceOutLocal, var, pregMerge, offset);
        NormCommon::ComputeRstdNewtonRaphsonReg<false>(var, rstd, pregMerge, eps);
        DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdLocal + offset, rstd, pregMerge);
    }
}

// 场景3：welford整块小于二分累加整块，并且大于二分累加尾块向上对齐
template <typename T>
__aicore__ inline void VFWelfordParallelFinalizeNonAlignSituation3(
    __local_mem__ float* meanLocal, __local_mem__ float* rstdLocal, __local_mem__ T* varianceOutLocal,
    __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* dichotomyAddLocal,
    uint32_t reduceCount, uint32_t dichotomyAddPower, uint32_t dichotomyAddK, uint32_t dichotomyAddLastNum,
    uint32_t offset, uint32_t tailSize, float reduceScale, float cnt, float eps)
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
#include "group_norm_regbase_base_part2.h"
