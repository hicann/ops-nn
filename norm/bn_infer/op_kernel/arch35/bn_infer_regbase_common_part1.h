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
 * \file bn_infer_regbase_common_part1.h
 * \brief Split from bn_infer_regbase_common.h for code-check line budget.
 */
__aicore__ inline void CopyInAllMeanVarPad(LocalTensor<float>& allMeanTensor, LocalTensor<float>& allVarTensor,
                                           GlobalTensor<float>& workspaceGm, int64_t meanOffset, int64_t varOffset,
                                           uint32_t usedCoreNum, uint32_t currentAAlign, uint32_t patternAAlign)
{
    DataCopyPadExtParams<float> meanVarPadParams;
    meanVarPadParams.isPad = false;
    meanVarPadParams.leftPadding = 0;
    meanVarPadParams.rightPadding = 0;
    meanVarPadParams.paddingValue = 0;
    DataCopyExtParams copyInMeanVarParams;
    copyInMeanVarParams.blockCount = usedCoreNum;
    copyInMeanVarParams.dstStride = 0;
    copyInMeanVarParams.blockLen = currentAAlign * sizeof(float);
    copyInMeanVarParams.srcStride = (patternAAlign - currentAAlign) * sizeof(float);
    DataCopyPad(allMeanTensor, workspaceGm[meanOffset], copyInMeanVarParams, meanVarPadParams);
    DataCopyPad(allVarTensor, workspaceGm[varOffset], copyInMeanVarParams, meanVarPadParams);
}

template <bool INIT, typename T_SRC>
__aicore__ inline void WelfordParallelUpdateVF(__local_mem__ T_SRC* x1Local, __local_mem__ float* tmpMeanLocal,
                                               __local_mem__ float* tmpVarLocal, uint64_t calLen, uint16_t loopCount,
                                               float scale, uint32_t vectorLen)
{
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> tmpMean;
        RegTensor<float> tmpVar;
        RegTensor<float> delta1;
        RegTensor<float> delta2;
        RegTensor<float> delta3;
        RegTensor<float> delta4;
        MaskReg pregLoop;
        uint32_t sreg0 = calLen;
        for (uint16_t i = 0; i < loopCount; i++) {
            pregLoop = UpdateMask<float>(sreg0);
            uint32_t offset = i * vectorLen;
            LoadOneTensorForDtypeT(x1Local, x1, pregLoop, offset);
            if constexpr (INIT) {
                Duplicate(tmpMean, 0.0, pregLoop);
            } else {
                DataCopy(tmpMean, tmpMeanLocal + offset);
            }
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            DataCopy(tmpMeanLocal + offset, tmpMean, pregLoop);

            if constexpr (INIT) {
                Duplicate(tmpVar, 0.0, pregLoop);
            } else {
                DataCopy(tmpVar, tmpVarLocal + offset);
            }
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delta4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delta4, pregLoop);
            DataCopy(tmpVarLocal + offset, tmpVar, pregLoop);
        }
    }
}

__aicore__ inline void BNInferMeanM2TensorInit(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                               uint32_t len, uint16_t loopCount, uint32_t vectorLen)
{
    __local_mem__ float* meanTensorAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
    __local_mem__ float* m2TensorAddr = (__local_mem__ float*)m2Tensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<float> tmpMean;
        RegTensor<float> tmpM2;
        MaskReg mask0 = CreateMask<float, MaskPattern::ALL>();
        Duplicate(tmpMean, 0.0, mask0);
        Duplicate(tmpM2, 0.0, mask0);
        MaskReg mask1;
        uint32_t sreg0 = len;
        for (uint16_t i = 0; i < loopCount; i++) {
            mask1 = UpdateMask<float>(sreg0);
            uint32_t offset = i * vectorLen;
            DataCopy(meanTensorAddr + offset, tmpMean, mask1);
            DataCopy(m2TensorAddr + offset, tmpM2, mask1);
        }
    }
}

__aicore__ inline void BinaryAddVF(__local_mem__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint16_t binaryAddKLoop,
                                   uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop,
                                   uint32_t offset, RegTensor<float>& x1, RegTensor<float>& x2, RegTensor<float>& x3,
                                   RegTensor<float>& x4)
{
    uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddInnerLoop = curBinaryAddInnerLoop / BN_INFER_ROW_FOUR_OFFSET;
        for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
            DataCopy(x1,
                     ((__local_mem__ float*)binaryAddTmpAddr + (j * BN_INFER_ROW_FOUR_OFFSET) * rLoopStride + offset));
            DataCopy(x2, ((__local_mem__ float*)binaryAddTmpAddr + (j * BN_INFER_ROW_FOUR_OFFSET + 1) * rLoopStride +
                          offset));
            Add(x1, x1, x2, pregLoop);
            DataCopy(x3, ((__local_mem__ float*)binaryAddTmpAddr +
                          (j * BN_INFER_ROW_FOUR_OFFSET + BN_INFER_ROW_TWO_OFFSET) * rLoopStride + offset));
            DataCopy(x4, ((__local_mem__ float*)binaryAddTmpAddr +
                          (j * BN_INFER_ROW_FOUR_OFFSET + BN_INFER_ROW_THREE_OFFSET) * rLoopStride + offset));
            Add(x3, x3, x4, pregLoop);
            Add(x1, x1, x3, pregLoop);
            DataCopy(((__local_mem__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
        DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + offset));
        DataCopy(x2, ((__local_mem__ float*)binaryAddTmpAddr + rLoopStride + offset));
        Add(x1, x1, x2, pregLoop);
        DataCopy(((__local_mem__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

__aicore__ inline void FillCountBlock(__local_mem__ float* dst, RegTensor<float>& tmpCount, MaskReg& pregMain,
                                      MaskReg& pregLoop, float addCount, uint32_t num, uint16_t loopCount, uint32_t vl)
{
    uint32_t sreg = num;
    Duplicate(tmpCount, addCount, pregMain);
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg);
        DataCopy(dst + i * vl, tmpCount, pregLoop);
    }
}

__aicore__ inline void TwoRowAddMeanWithTail(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
                                             uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                             RegTensor<float>& rem, RegTensor<float>& nextRow,
                                             RegTensor<float>& remNextRow, float n)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
    Muls(dst, dst, n, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset3)));
    DataCopy(remNextRow, ((__local_mem__ float*)(input) + (offset4)));
    Muls(nextRow, nextRow, n, preg);
    Muls(remNextRow, remNextRow, n, preg);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddMean(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1,
                                     uint32_t offset2, RegTensor<float>& nextRow, float n)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
    Muls(dst, dst, n, preg);
    Muls(nextRow, nextRow, n, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddVarWithTail(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
                                            uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                            RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& nextRow,
                                            RegTensor<float>& remNextRow, float n)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
    Sub(dst, dst, mean, preg);
    Sub(rem, rem, mean, preg);
    Mul(dst, dst, dst, preg);
    Mul(rem, rem, rem, preg);
    Muls(dst, dst, n, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset3)));
    DataCopy(remNextRow, ((__local_mem__ float*)(input) + (offset4)));
    Sub(nextRow, nextRow, mean, preg);
    Sub(remNextRow, remNextRow, mean, preg);
    Mul(nextRow, nextRow, nextRow, preg);
    Mul(remNextRow, remNextRow, remNextRow, preg);
    Muls(nextRow, nextRow, n, preg);
    Muls(remNextRow, remNextRow, n, preg);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddVar(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1,
                                    uint32_t offset2, RegTensor<float>& mean, RegTensor<float>& nextRow, float n)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
    Sub(dst, dst, mean, preg);
    Sub(nextRow, nextRow, mean, preg);
    Mul(dst, dst, dst, preg);
    Mul(nextRow, nextRow, nextRow, preg);
    Muls(dst, dst, n, preg);
    Muls(nextRow, nextRow, n, preg);
    Add(dst, dst, nextRow, preg);
}

template <bool CALC_VAR, uint32_t SCALE_COEF>
__aicore__ inline void CalculateRLessThanVF(__local_mem__ float* xInUb, __local_mem__ float* batchMeanInUbAddr,
                                            __local_mem__ float* batchVarOutUbAddr, int64_t currentA,
                                            uint32_t currentANumAlign, uint32_t r1, uint32_t vectorLen, float n,
                                            float nCorrection)
{
    RLessThanParams params = GetRLessThanParams(SCALE_COEF, currentANumAlign, r1);
    uint16_t aLoopCount = (currentA + vectorLen - 1) / vectorLen;
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> nextRow;
        RegTensor<float> rem;
        RegTensor<float> remNextRow;
        RegTensor<float> mean;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> zero;
        Duplicate(zero, 0.0, pregMain);

        MaskReg pregLoop;
        uint32_t sreg0 = currentA;
        for (uint16_t k = 0; k < aLoopCount; k++) {
            pregLoop = UpdateMask<float>(sreg0);
            uint32_t aLoopOffset = k * vectorLen;
            if constexpr (CALC_VAR) {
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)xInUb + params.validNumInXUb + aLoopOffset), mean, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)xInUb + params.validNumInXUb + aLoopOffset), zero, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (CALC_VAR) {
                TwoRowAddVarWithTail(x1, xInUb, pregLoop, aLoopOffset, params.remainderTailOffset0 + aLoopOffset,
                                     params.aLength + aLoopOffset, params.remainderTailOffset1 + aLoopOffset, mean, rem,
                                     nextRow, remNextRow, n);
            } else {
                TwoRowAddMeanWithTail(x1, xInUb, pregLoop, aLoopOffset, params.remainderTailOffset0 + aLoopOffset,
                                      params.aLength + aLoopOffset, params.remainderTailOffset1 + aLoopOffset, rem,
                                      nextRow, remNextRow, n);
            }
            if constexpr (SCALE_COEF == BN_INFER_ROW_FOUR_OFFSET) {
                if constexpr (CALC_VAR) {
                    TwoRowAddVarWithTail(x2, xInUb, pregLoop, BN_INFER_ROW_TWO_OFFSET * params.aLength + aLoopOffset,
                                         params.remainderTailOffset2 + aLoopOffset,
                                         BN_INFER_ROW_THREE_OFFSET * params.aLength + aLoopOffset,
                                         params.remainderTailOffset3 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                } else {
                    TwoRowAddMeanWithTail(x2, xInUb, pregLoop, BN_INFER_ROW_TWO_OFFSET * params.aLength + aLoopOffset,
                                          params.remainderTailOffset2 + aLoopOffset,
                                          BN_INFER_ROW_THREE_OFFSET * params.aLength + aLoopOffset,
                                          params.remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow, n);
                }
                Add(x1, x1, x2, pregLoop);
            }
            Muls(x1, x1, nCorrection, pregLoop);
            if constexpr (CALC_VAR) {
                DataCopy(((__local_mem__ float*)batchVarOutUbAddr + aLoopOffset), x1, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }
}

__aicore__ inline void TwoRowAddPartialMeanWithTail(RegTensor<float>& dst, __local_mem__ float* input,
                                                    __local_mem__ float* tCount, MaskReg& preg, uint32_t offset1,
                                                    uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                                    uint32_t offset5, uint32_t offset6, uint32_t offset7,
                                                    uint32_t offset8, RegTensor<float>& rem, RegTensor<float>& nextRow,
                                                    RegTensor<float>& remNextRow, RegTensor<float>& dstCount,
                                                    RegTensor<float>& remCount, RegTensor<float>& nextRowCount,
                                                    RegTensor<float>& remNextRowCount, float n)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(dstCount, ((__local_mem__ float*)(tCount) + (offset5)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(remCount, ((__local_mem__ float*)(tCount) + (offset6)));
    Mul(dst, dst, dstCount, preg);
    Mul(rem, rem, remCount, preg);
    Muls(dst, dst, n, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset3)));
    DataCopy(remNextRow, ((__local_mem__ float*)(input) + (offset4)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(nextRowCount, ((__local_mem__ float*)(tCount) + (offset7)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(remNextRowCount, ((__local_mem__ float*)(tCount) + (offset8)));
    Mul(nextRow, nextRow, nextRowCount, preg);
    Mul(remNextRow, remNextRow, remNextRowCount, preg);
    Muls(nextRow, nextRow, n, preg);
    Muls(remNextRow, remNextRow, n, preg);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddPartialMean(RegTensor<float>& dst, __local_mem__ float* input,
                                            __local_mem__ float* tCount, MaskReg& preg, uint32_t offset1,
                                            uint32_t offset2, uint32_t offset5, uint32_t offset6, RegTensor<float>& rem,
                                            RegTensor<float>& dstCount, RegTensor<float>& remCount, float n)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(dstCount, ((__local_mem__ float*)(tCount) + (offset5)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(remCount, ((__local_mem__ float*)(tCount) + (offset6)));
    Mul(dst, dst, dstCount, preg);
    Mul(rem, rem, remCount, preg);
    Muls(dst, dst, n, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
}

__aicore__ inline void TwoRowAddPartialVar(RegTensor<float>& dst, __local_mem__ float* tmpMean,
                                           __local_mem__ float* tmpM2, __local_mem__ float* tCount, MaskReg& preg,
                                           uint32_t offset1, uint32_t offset2, uint32_t offset5, uint32_t offset6,
                                           RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& dstCount,
                                           RegTensor<float>& remCount, RegTensor<float>& dstM2, RegTensor<float>& remM2,
                                           float n)
{
    DataCopy(dst, ((__local_mem__ float*)(tmpMean) + (offset1)));
    DataCopy(rem, ((__local_mem__ float*)(tmpMean) + (offset2)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(dstCount, ((__local_mem__ float*)(tCount) + (offset5)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(remCount, ((__local_mem__ float*)(tCount) + (offset6)));
    Sub(dst, dst, mean, preg);
    Mul(dst, dst, dst, preg);
    Sub(rem, rem, mean, preg);
    Mul(rem, rem, rem, preg);
    Mul(dst, dst, dstCount, preg);
    Mul(rem, rem, remCount, preg);
    DataCopy(dstM2, ((__local_mem__ float*)(tmpM2) + (offset1)));
    DataCopy(remM2, ((__local_mem__ float*)(tmpM2) + (offset2)));
    Add(dst, dstM2, dst, preg);
    Muls(dst, dst, n, preg);
    Add(rem, remM2, rem, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
}

__aicore__ inline void TwoRowAddPartialVarWithTail(
    RegTensor<float>& dst, __local_mem__ float* tmpMean, __local_mem__ float* tmpM2, __local_mem__ float* tCount,
    MaskReg& preg, uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4, uint32_t offset5,
    uint32_t offset6, uint32_t offset7, uint32_t offset8, RegTensor<float>& mean, RegTensor<float>& rem,
    RegTensor<float>& nextRow, RegTensor<float>& remNextRow, RegTensor<float>& dstCount, RegTensor<float>& remCount,
    RegTensor<float>& nextRowCount, RegTensor<float>& remNextRowCount, RegTensor<float>& dstM2, RegTensor<float>& remM2,
    RegTensor<float>& nextRowM2, RegTensor<float>& remNextRowM2, float n)
{
    TwoRowAddPartialVar(dst, tmpMean, tmpM2, tCount, preg, offset1, offset2, offset5, offset6, mean, rem, dstCount,
                        remCount, dstM2, remM2, n);
    TwoRowAddPartialVar(nextRow, tmpMean, tmpM2, tCount, preg, offset3, offset4, offset7, offset8, mean, remNextRow,
                        nextRowCount, remNextRowCount, nextRowM2, remNextRowM2, n);
    Add(dst, dst, nextRow, preg);
}
