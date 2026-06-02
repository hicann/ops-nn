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
 * \file batch_norm_v3_regbase_common.h
 * \brief batch_norm_v3 regbase common helper
 */
#ifndef BATCH_NORM_V3_REGBASE_COMMON_H
#define BATCH_NORM_V3_REGBASE_COMMON_H

#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNormV3Ops {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::UpdateMask;

constexpr static uint32_t BATCH_NORM_V3_ROW_TWO_OFFSET = 2;
constexpr static uint32_t BATCH_NORM_V3_ROW_THREE_OFFSET = 3;
constexpr static uint32_t BATCH_NORM_V3_ROW_FOUR_OFFSET = 4;
constexpr static uint32_t BATCH_NORM_V3_ROW_ZERO = 0;
constexpr static uint32_t BATCH_NORM_V3_ROW_ONE = 1;
constexpr static uint32_t BATCH_NORM_V3_INDEX_ONE = 1;
constexpr static uint32_t BATCH_NORM_V3_INDEX_TWO = 2;
constexpr static uint32_t BATCH_NORM_V3_INDEX_FOUR = 4;
constexpr static uint32_t BATCH_NORM_V3_INDEX_EIGHT = 8;
constexpr static uint32_t BATCH_NORM_V3_INDEX_SIXTEEN = 16;
constexpr static float BATCH_NORM_V3_POS_INF = 3.40282366920938E+38;

struct RLessThanParams {
    uint32_t remainderOffset;
    uint32_t aLength;
    uint32_t validNumInXUb;
    uint16_t remainderTailCount;
    uint32_t remainderTailOffset0;
    uint32_t remainderTailOffset1;
    uint32_t remainderTailOffset2;
    uint32_t remainderTailOffset3;
};

__aicore__ inline RLessThanParams GetRLessThanParams(uint32_t scaleCoef, uint32_t currentANumAlign, uint32_t r1)
{
    RLessThanParams params;
    params.remainderOffset = scaleCoef * currentANumAlign;
    params.aLength = currentANumAlign;
    params.validNumInXUb = r1 * currentANumAlign;
    params.remainderTailCount = r1 - scaleCoef;
    params.remainderTailOffset0 = (BATCH_NORM_V3_ROW_ZERO > params.remainderTailCount) ? params.validNumInXUb :
        params.remainderOffset;
    params.remainderTailOffset1 = (BATCH_NORM_V3_ROW_ONE > params.remainderTailCount) ? params.validNumInXUb :
        params.remainderOffset + params.aLength;
    params.remainderTailOffset2 = (BATCH_NORM_V3_ROW_TWO_OFFSET > params.remainderTailCount) ?
        params.validNumInXUb : params.remainderOffset + BATCH_NORM_V3_ROW_TWO_OFFSET * params.aLength;
    params.remainderTailOffset3 = (BATCH_NORM_V3_ROW_THREE_OFFSET > params.remainderTailCount) ?
        params.validNumInXUb : params.remainderOffset + BATCH_NORM_V3_ROW_THREE_OFFSET * params.aLength;
    return params;
}

template <typename T_SRC>
__aicore__ inline void LoadOneTensorForDtypeT(
    __local_mem__ T_SRC* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T_SRC, half>::value) {
        RegTensor<half> xFp16;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + offset));
        Cast<float, half, NormCommon::castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + offset));
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst, xBf16, preg);
    } else {
        DataCopy(dst, ((__local_mem__ float*)(input) + offset));
    }
}

__aicore__ inline uint32_t BatchNormV3FindCofFactor(uint32_t n)
{
    if (n == 0) {
        return 0;
    }
    if ((n & (n - 1)) == 0) {
        return n;
    }
    uint32_t temp = n - 1;
    temp |= temp >> BATCH_NORM_V3_INDEX_ONE;
    temp |= temp >> BATCH_NORM_V3_INDEX_TWO;
    temp |= temp >> BATCH_NORM_V3_INDEX_FOUR;
    temp |= temp >> BATCH_NORM_V3_INDEX_EIGHT;
    temp |= temp >> BATCH_NORM_V3_INDEX_SIXTEEN;
    return (temp + 1);
}

template <typename T_SRC>
__aicore__ inline void LoadTwoTensorForDtypeT(
    __local_mem__ T_SRC* src1, __local_mem__ T_SRC* src2, RegTensor<float>& dst1, RegTensor<float>& dst2,
    MaskReg& dst1Preg, MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T_SRC, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half*)(src1) + src1Offset));
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half*)(src2) + src2Offset));
        Cast<float, half, NormCommon::castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, NormCommon::castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16Q;
        RegTensor<bfloat16_t> xBf16R;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16Q, ((__local_mem__ bfloat16_t*)(src1) + src1Offset));
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16R, ((__local_mem__ bfloat16_t*)(src2) + src2Offset));
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst1, xBf16Q, dst1Preg);
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst2, xBf16R, dst2Preg);
    } else {
        DataCopy(dst1, ((__local_mem__ float*)(src1) + src1Offset));
        DataCopy(dst2, ((__local_mem__ float*)(src2) + src2Offset));
    }
}

template <typename T_SRC>
__aicore__ inline void LoadTwoTensorForDtypeTBrc(
    __local_mem__ T_SRC* src1, __local_mem__ T_SRC* src2, RegTensor<float>& dst1, RegTensor<float>& dst2,
    MaskReg& dst1Preg, MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T_SRC, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16Q, ((__local_mem__ half*)(src1) + src1Offset));
        DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16R, ((__local_mem__ half*)(src2) + src2Offset));
        Cast<float, half, NormCommon::castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, NormCommon::castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16Q;
        RegTensor<bfloat16_t> xBf16R;
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xBf16Q, ((__local_mem__ bfloat16_t*)(src1) + src1Offset));
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xBf16R, ((__local_mem__ bfloat16_t*)(src2) + src2Offset));
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst1, xBf16Q, dst1Preg);
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst2, xBf16R, dst2Preg);
    } else {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst1, ((__local_mem__ float*)(src1) + src1Offset));
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst2, ((__local_mem__ float*)(src2) + src2Offset));
    }
}

template <typename T_GAMMA>
__aicore__ inline void CopyInGammaBetaPad(LocalTensor<T_GAMMA>& gammaInUb, LocalTensor<T_GAMMA>& betaInUb,
    GlobalTensor<T_GAMMA>& gammaGm, GlobalTensor<T_GAMMA>& betaGm, int64_t gmOffset, uint32_t currentA)
{
    DataCopyPadExtParams<T_GAMMA> dataCopyPadExtParamsT;
    dataCopyPadExtParamsT.isPad = false;
    dataCopyPadExtParamsT.leftPadding = 0;
    dataCopyPadExtParamsT.rightPadding = 0;
    dataCopyPadExtParamsT.paddingValue = 0;
    DataCopyExtParams copyInParamsT;
    copyInParamsT.blockCount = 1;
    copyInParamsT.blockLen = currentA * sizeof(T_GAMMA);
    copyInParamsT.srcStride = 0;
    copyInParamsT.dstStride = 0;
    DataCopyPad(betaInUb, betaGm[gmOffset], copyInParamsT, dataCopyPadExtParamsT);
    DataCopyPad(gammaInUb, gammaGm[gmOffset], copyInParamsT, dataCopyPadExtParamsT);
}

template <typename T_GAMMA, typename BetaQueue, typename GammaQueue>
__aicore__ inline void CopyInGammaBetaCommon(BetaQueue& betaQueue, GammaQueue& gammaQueue,
    GlobalTensor<T_GAMMA>& betaGm, GlobalTensor<T_GAMMA>& gammaGm, int64_t gmOffset, int64_t currentA)
{
    LocalTensor<T_GAMMA> betaInUb = betaQueue.template AllocTensor<T_GAMMA>();
    LocalTensor<T_GAMMA> gammaInUb = gammaQueue.template AllocTensor<T_GAMMA>();
    CopyInGammaBetaPad(gammaInUb, betaInUb, gammaGm, betaGm, gmOffset, static_cast<uint32_t>(currentA));
    betaQueue.EnQue(betaInUb);
    gammaQueue.EnQue(gammaInUb);
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CopyInRunningMeanVarPad(
    LocalTensor<T_RUNNING_MEAN>& runningMeanInUb, LocalTensor<T_RUNNING_MEAN>& runningVarInUb,
    GlobalTensor<T_RUNNING_MEAN>& runningMeanGm, GlobalTensor<T_RUNNING_MEAN>& runningVarGm, int64_t gmOffset,
    uint32_t currentA)
{
    DataCopyPadExtParams<T_RUNNING_MEAN> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = currentA * sizeof(T_RUNNING_MEAN);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPad(runningMeanInUb, runningMeanGm[gmOffset], copyInParams, dataCopyPadExtParams);
    DataCopyPad(runningVarInUb, runningVarGm[gmOffset], copyInParams, dataCopyPadExtParams);
}

template <typename T_RUNNING_MEAN, typename MeanQueue, typename VarQueue>
__aicore__ inline void CopyInRunningMeanVarCommon(MeanQueue& runningMeanInQueue,
    VarQueue& runningVarInQueue, GlobalTensor<T_RUNNING_MEAN>& runningMeanGm,
    GlobalTensor<T_RUNNING_MEAN>& runningVarGm, int64_t gmOffset, int64_t currentA)
{
    LocalTensor<T_RUNNING_MEAN> runningMeanInUb = runningMeanInQueue.template AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningVarInUb = runningVarInQueue.template AllocTensor<T_RUNNING_MEAN>();
    CopyInRunningMeanVarPad(
        runningMeanInUb, runningVarInUb, runningMeanGm, runningVarGm, gmOffset, static_cast<uint32_t>(currentA));
    runningMeanInQueue.EnQue(runningMeanInUb);
    runningVarInQueue.EnQue(runningVarInUb);
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CopyOutRunningMeanVarPad(
    LocalTensor<T_RUNNING_MEAN>& runningMeanOutUb, LocalTensor<T_RUNNING_MEAN>& runningVarOutUb,
    GlobalTensor<T_RUNNING_MEAN>& runningMeanOutGm, GlobalTensor<T_RUNNING_MEAN>& runningVarOutGm, int64_t gmOffset,
    uint32_t currentA)
{
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = currentA * sizeof(T_RUNNING_MEAN);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPad(runningMeanOutGm[gmOffset], runningMeanOutUb, copyInParams);
    DataCopyPad(runningVarOutGm[gmOffset], runningVarOutUb, copyInParams);
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CalculateRunningMeanVarVF(__local_mem__ float* batchMeanInUb,
    __local_mem__ float* batchVarInUb, __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr,
    __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr, __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr,
    __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr, uint16_t currentANum, uint16_t aLoop, uint32_t vectorLen,
    float unbiasedEstimationCoeff, float momentum, float momentumReverse);

template <typename T_RUNNING_MEAN>
__aicore__ inline void UpdateRunningMeanVarCommon(LocalTensor<float>& batchMeanTensor,
    LocalTensor<float>& batchRstdTensor, TQue<QuePosition::VECIN, 1>& runningMeanInQueue,
    TQue<QuePosition::VECIN, 1>& runningVarInQueue, TQue<QuePosition::VECOUT, 1>& runningMeanOutQueue,
    TQue<QuePosition::VECOUT, 1>& runningVarOutQueue, GlobalTensor<T_RUNNING_MEAN>& runningMeanGm,
    GlobalTensor<T_RUNNING_MEAN>& runningVarGm, GlobalTensor<T_RUNNING_MEAN>& runningMeanOutGm,
    GlobalTensor<T_RUNNING_MEAN>& runningVarOutGm, int64_t gmOffset, uint32_t currentA, uint16_t aLoop,
    uint32_t vectorLen, float unbiasedEstimationCoeff, float momentum, float momentumReverse)
{
    LocalTensor<T_RUNNING_MEAN> runningMeanInTensor = runningMeanInQueue.AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningVarInTensor = runningVarInQueue.AllocTensor<T_RUNNING_MEAN>();
    CopyInRunningMeanVarPad(
        runningMeanInTensor, runningVarInTensor, runningMeanGm, runningVarGm, gmOffset, currentA);
    runningMeanInQueue.EnQue(runningMeanInTensor);
    runningVarInQueue.EnQue(runningVarInTensor);
    runningMeanInTensor = runningMeanInQueue.template DeQue<T_RUNNING_MEAN>();
    runningVarInTensor = runningVarInQueue.template DeQue<T_RUNNING_MEAN>();

    LocalTensor<T_RUNNING_MEAN> runningMeanOutTensor = runningMeanOutQueue.AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningVarOutTensor = runningVarOutQueue.AllocTensor<T_RUNNING_MEAN>();
    __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr =
        (__local_mem__ T_RUNNING_MEAN*)runningMeanInTensor.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr =
        (__local_mem__ T_RUNNING_MEAN*)runningVarInTensor.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr =
        (__local_mem__ T_RUNNING_MEAN*)runningMeanOutTensor.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr =
        (__local_mem__ T_RUNNING_MEAN*)runningVarOutTensor.GetPhyAddr();
    __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
    __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
    CalculateRunningMeanVarVF<T_RUNNING_MEAN>(batchMeanTensorAddr, batchRstdTensorAddr, runningMeanInUbAddr,
        runningVarInUbAddr, runningMeanOutUbAddr, runningVarOutUbAddr, static_cast<uint16_t>(currentA), aLoop,
        vectorLen, unbiasedEstimationCoeff, momentum, momentumReverse);

    runningMeanInQueue.FreeTensor(runningMeanInTensor);
    runningVarInQueue.FreeTensor(runningVarInTensor);
    runningMeanOutQueue.EnQue(runningMeanOutTensor);
    runningVarOutQueue.EnQue(runningVarOutTensor);
    runningMeanOutTensor = runningMeanOutQueue.template DeQue<T_RUNNING_MEAN>();
    runningVarOutTensor = runningVarOutQueue.template DeQue<T_RUNNING_MEAN>();
    CopyOutRunningMeanVarPad(
        runningMeanOutTensor, runningVarOutTensor, runningMeanOutGm, runningVarOutGm, gmOffset, currentA);
    runningMeanOutQueue.FreeTensor(runningMeanOutTensor);
    runningVarOutQueue.FreeTensor(runningVarOutTensor);
}

__aicore__ inline void CopyOutBatchMeanRstdPad(LocalTensor<float>& batchMeanInUb, LocalTensor<float>& batchRstdInUb,
    GlobalTensor<float>& batchMeanGm, GlobalTensor<float>& batchRstdGm, int64_t gmOffset, uint32_t currentA)
{
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = currentA * sizeof(float);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPad(batchMeanGm[gmOffset], batchMeanInUb, copyInParams);
    DataCopyPad(batchRstdGm[gmOffset], batchRstdInUb, copyInParams);
}

template <typename MeanQueue, typename RstdQueue>
__aicore__ inline void CopyOutBatchMeanRstdCommon(MeanQueue& batchMeanQueue,
    RstdQueue& batchRstdQueue, GlobalTensor<float>& batchMeanGm,
    GlobalTensor<float>& batchRstdGm, int64_t gmOffset, int64_t currentA)
{
    LocalTensor<float> batchMeanInUb = batchMeanQueue.template DeQue<float>();
    LocalTensor<float> batchRstdInUb = batchRstdQueue.template DeQue<float>();
    CopyOutBatchMeanRstdPad(
        batchMeanInUb, batchRstdInUb, batchMeanGm, batchRstdGm, gmOffset, static_cast<uint32_t>(currentA));
    batchMeanQueue.FreeTensor(batchMeanInUb);
    batchRstdQueue.FreeTensor(batchRstdInUb);
}

__aicore__ inline void CopyInAllMeanVarPad(LocalTensor<float>& allMeanTensor, LocalTensor<float>& allVarTensor,
    GlobalTensor<float>& workspaceGm, int64_t meanOffset, int64_t varOffset, uint32_t usedCoreNum, uint32_t currentAAlign,
    uint32_t patternAAlign)
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
    __local_mem__ float* tmpVarLocal, uint64_t calLen, uint16_t loopCount, float scale, uint32_t vectorLen)
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

__aicore__ inline void BatchNormV3MeanM2TensorInit(
    LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor, uint32_t len, uint16_t loopCount, uint32_t vectorLen)
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

__aicore__ inline void BinaryAddVF(
    __local_mem__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint16_t binaryAddKLoop,
    uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop, uint32_t offset,
    RegTensor<float>& x1, RegTensor<float>& x2, RegTensor<float>& x3, RegTensor<float>& x4)
{
    uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddInnerLoop = curBinaryAddInnerLoop / BATCH_NORM_V3_ROW_FOUR_OFFSET;
        for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
            DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr +
                          (j * BATCH_NORM_V3_ROW_FOUR_OFFSET) * rLoopStride + offset));
            DataCopy(x2, ((__local_mem__ float*)binaryAddTmpAddr +
                          (j * BATCH_NORM_V3_ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            DataCopy(x3, ((__local_mem__ float*)binaryAddTmpAddr +
                          (j * BATCH_NORM_V3_ROW_FOUR_OFFSET + BATCH_NORM_V3_ROW_TWO_OFFSET) * rLoopStride + offset));
            DataCopy(x4, ((__local_mem__ float*)binaryAddTmpAddr +
                          (j * BATCH_NORM_V3_ROW_FOUR_OFFSET + BATCH_NORM_V3_ROW_THREE_OFFSET) * rLoopStride + offset));
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

__aicore__ inline void TwoRowAddMeanWithTail(
    RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
    uint32_t offset3, uint32_t offset4, RegTensor<float>& rem, RegTensor<float>& nextRow,
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

__aicore__ inline void TwoRowAddMean(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
    uint32_t offset1, uint32_t offset2, RegTensor<float>& nextRow, float n)
{
    DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
    DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
    Muls(dst, dst, n, preg);
    Muls(nextRow, nextRow, n, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddVarWithTail(
    RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
    uint32_t offset3, uint32_t offset4, RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& nextRow,
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

__aicore__ inline void TwoRowAddVar(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
    uint32_t offset1, uint32_t offset2, RegTensor<float>& mean, RegTensor<float>& nextRow, float n)
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
    __local_mem__ float* batchVarOutUbAddr, int64_t currentA, uint32_t currentANumAlign, uint32_t r1,
    uint32_t vectorLen, float n, float nCorrection)
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
                TwoRowAddVarWithTail(x1, xInUb, pregLoop, aLoopOffset,
                    params.remainderTailOffset0 + aLoopOffset, params.aLength + aLoopOffset,
                    params.remainderTailOffset1 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
            } else {
                TwoRowAddMeanWithTail(x1, xInUb, pregLoop, aLoopOffset,
                    params.remainderTailOffset0 + aLoopOffset, params.aLength + aLoopOffset,
                    params.remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow, n);
            }
            if constexpr (SCALE_COEF == BATCH_NORM_V3_ROW_FOUR_OFFSET) {
                if constexpr (CALC_VAR) {
                    TwoRowAddVarWithTail(x2, xInUb, pregLoop,
                        BATCH_NORM_V3_ROW_TWO_OFFSET * params.aLength + aLoopOffset,
                        params.remainderTailOffset2 + aLoopOffset,
                        BATCH_NORM_V3_ROW_THREE_OFFSET * params.aLength + aLoopOffset,
                        params.remainderTailOffset3 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                } else {
                    TwoRowAddMeanWithTail(x2, xInUb, pregLoop,
                        BATCH_NORM_V3_ROW_TWO_OFFSET * params.aLength + aLoopOffset,
                        params.remainderTailOffset2 + aLoopOffset,
                        BATCH_NORM_V3_ROW_THREE_OFFSET * params.aLength + aLoopOffset,
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

__aicore__ inline void TwoRowAddPartialMeanWithTail(
    RegTensor<float>& dst, __local_mem__ float* input, __local_mem__ float* tCount, MaskReg& preg, uint32_t offset1,
    uint32_t offset2, uint32_t offset3, uint32_t offset4, uint32_t offset5, uint32_t offset6, uint32_t offset7,
    uint32_t offset8, RegTensor<float>& rem, RegTensor<float>& nextRow, RegTensor<float>& remNextRow,
    RegTensor<float>& dstCount, RegTensor<float>& remCount, RegTensor<float>& nextRowCount,
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

__aicore__ inline void TwoRowAddPartialMean(
    RegTensor<float>& dst, __local_mem__ float* input, __local_mem__ float* tCount, MaskReg& preg, uint32_t offset1,
    uint32_t offset2, uint32_t offset5, uint32_t offset6, RegTensor<float>& rem, RegTensor<float>& dstCount,
    RegTensor<float>& remCount, float n)
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

__aicore__ inline void TwoRowAddPartialVarWithTail(
    RegTensor<float>& dst, __local_mem__ float* tmpMean, __local_mem__ float* tmpM2, __local_mem__ float* tCount,
    MaskReg& preg, uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4, uint32_t offset5,
    uint32_t offset6, uint32_t offset7, uint32_t offset8, RegTensor<float>& mean, RegTensor<float>& rem,
    RegTensor<float>& nextRow, RegTensor<float>& remNextRow, RegTensor<float>& dstCount, RegTensor<float>& remCount,
    RegTensor<float>& nextRowCount, RegTensor<float>& remNextRowCount, RegTensor<float>& dstM2,
    RegTensor<float>& remM2, RegTensor<float>& nextRowM2, RegTensor<float>& remNextRowM2, float n)
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

    DataCopy(nextRow, ((__local_mem__ float*)(tmpMean) + (offset3)));
    DataCopy(remNextRow, ((__local_mem__ float*)(tmpMean) + (offset4)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(nextRowCount, ((__local_mem__ float*)(tCount) + (offset7)));
    DataCopy<float, LoadDist::DIST_BRC_B32>(remNextRowCount, ((__local_mem__ float*)(tCount) + (offset8)));
    Sub(nextRow, nextRow, mean, preg);
    Mul(nextRow, nextRow, nextRow, preg);
    Sub(remNextRow, remNextRow, mean, preg);
    Mul(remNextRow, remNextRow, remNextRow, preg);
    Mul(nextRow, nextRow, nextRowCount, preg);
    Mul(remNextRow, remNextRow, remNextRowCount, preg);
    DataCopy(nextRowM2, ((__local_mem__ float*)(tmpM2) + (offset3)));
    DataCopy(remNextRowM2, ((__local_mem__ float*)(tmpM2) + (offset4)));
    Add(nextRow, nextRowM2, nextRow, preg);
    Muls(nextRow, nextRow, n, preg);
    Add(remNextRow, remNextRowM2, remNextRow, preg);
    Muls(remNextRow, remNextRow, n, preg);
    Add(nextRow, nextRow, remNextRow, preg);

    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddPartialVar(
    RegTensor<float>& dst, __local_mem__ float* tmpMean, __local_mem__ float* tmpM2, __local_mem__ float* tCount,
    MaskReg& preg, uint32_t offset1, uint32_t offset2, uint32_t offset5, uint32_t offset6, RegTensor<float>& mean,
    RegTensor<float>& rem, RegTensor<float>& dstCount, RegTensor<float>& remCount, RegTensor<float>& dstM2,
    RegTensor<float>& remM2, float n)
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

__aicore__ inline void LastFinalizeVF(
    LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor, LocalTensor<float>& meanTensor,
    LocalTensor<float>& varTensor, LocalTensor<float>& countTensor, LocalTensor<float>& tmpTensor, uint32_t currentAAlign, uint32_t vectorLen,
    uint16_t currentA, uint16_t usedCoreNum, uint16_t lastBinaryAddQuotient, uint16_t lastBinaryAddK,
    uint16_t lastBinaryAddLast, float lastNFactor, float lastNCorrectionFactor)
{
    __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
    __local_mem__ float* tmpCountLocal = (__local_mem__ float*)countTensor.GetPhyAddr();
    __local_mem__ float* tmpVarLocal = (__local_mem__ float*)varTensor.GetPhyAddr();
    __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
    __local_mem__ float* tmpUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr();
    __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
    uint32_t rLoopStride = currentAAlign;
    vectorLen = (vectorLen == 0) ? (NormCommon::NormCommonRegbase::GetVRegSize() / sizeof(float)) : vectorLen;
    uint16_t aLoopCount = ((currentA + vectorLen - 1) / vectorLen);
    uint16_t remainderLoopCount = usedCoreNum - lastBinaryAddQuotient;
    uint16_t quotientLoopCount = lastBinaryAddQuotient - remainderLoopCount;
    uint32_t baseLineOffset = rLoopStride;
    uint32_t remainderCountOffset = lastBinaryAddQuotient;
    uint32_t remainderOffset = lastBinaryAddQuotient * rLoopStride;
    uint16_t binaryAddKLoop = lastBinaryAddK;
    uint16_t binaryAddLastLoop = lastBinaryAddLast;
    uint16_t binaryAddInnerLoop = lastBinaryAddQuotient;
    float numScale = lastNFactor;
    float scaleCorrection = lastNCorrectionFactor;
    __VEC_SCOPE__
    {
        RegTensor<float> quot;
        RegTensor<float> rem;
        RegTensor<float> remCount;
        RegTensor<float> quotCount;
        RegTensor<float> oriQuotMean;
        RegTensor<float> resMean;
        RegTensor<float> oriRemMean;
        RegTensor<float> resVar;

        uint32_t sreg0 = currentA;
        MaskReg pregLoop;
        for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
            uint32_t aLoopOffset = aIndex * vectorLen;
            pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t i = 0; i < remainderLoopCount; i++) {
                uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                uint32_t quotCountOffset = i;
                uint32_t remCountOffset = i + remainderCountOffset;
                DataCopy(quot, ((__local_mem__ float*)(tmpMeanLocal) + (quotOffset)));
                DataCopy(rem, ((__local_mem__ float*)(tmpMeanLocal) + (remOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount, ((__local_mem__ float*)(tmpCountLocal) + quotCountOffset));
                DataCopy<float, LoadDist::DIST_BRC_B32>(remCount, ((__local_mem__ float*)(tmpCountLocal) + remCountOffset));
                Mul(quot, quot, quotCount, pregLoop);
                Mul(rem, rem, remCount, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                Muls(rem, rem, numScale, pregLoop);
                Add(quot, quot, rem, pregLoop);
                DataCopy(((__local_mem__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
            }
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                uint32_t baseCountOffset = remainderLoopCount + i;
                DataCopy(quot, ((__local_mem__ float*)(tmpMeanLocal) + (baseOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount, ((__local_mem__ float*)(tmpCountLocal) + baseCountOffset));
                Mul(quot, quot, quotCount, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                DataCopy(((__local_mem__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                    pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            BinaryAddVF(
                tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop, pregLoop,
                aLoopOffset, quot, rem, quotCount, remCount);
            DataCopy(resMean, ((__local_mem__ float*)tmpUbAddr + aLoopOffset));
            Muls(resMean, resMean, scaleCorrection, pregLoop);
            DataCopy(((__local_mem__ float*)batchMeanTensorAddr + aLoopOffset), resMean, pregLoop);
            for (uint16_t i = 0; i < remainderLoopCount; i++) {
                uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                uint32_t quotCountOffset = i;
                uint32_t remCountOffset = i + remainderCountOffset;
                DataCopy(quot, ((__local_mem__ float*)(tmpVarLocal) + (quotOffset)));
                DataCopy(rem, ((__local_mem__ float*)(tmpVarLocal) + (remOffset)));
                DataCopy(oriQuotMean, ((__local_mem__ float*)(tmpMeanLocal) + (quotOffset)));
                DataCopy(oriRemMean, ((__local_mem__ float*)(tmpMeanLocal) + (remOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount, ((__local_mem__ float*)(tmpCountLocal) + quotCountOffset));
                DataCopy<float, LoadDist::DIST_BRC_B32>(remCount, ((__local_mem__ float*)(tmpCountLocal) + remCountOffset));
                Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                Sub(oriRemMean, oriRemMean, resMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                Mul(oriRemMean, oriRemMean, oriRemMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                Mul(oriRemMean, oriRemMean, remCount, pregLoop);
                Mul(quot, quot, quotCount, pregLoop);
                Mul(rem, rem, remCount, pregLoop);
                Add(quot, quot, oriQuotMean, pregLoop);
                Add(rem, rem, oriRemMean, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                Muls(rem, rem, numScale, pregLoop);
                Add(quot, quot, rem, pregLoop);
                DataCopy(((__local_mem__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
            }
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                uint32_t baseCountOffset = remainderLoopCount + i;
                DataCopy(quot, ((__local_mem__ float*)(tmpVarLocal) + (baseOffset)));
                DataCopy(oriQuotMean, ((__local_mem__ float*)(tmpMeanLocal) + (baseOffset)));
                DataCopy<float, LoadDist::DIST_BRC_B32>(quotCount, ((__local_mem__ float*)(tmpCountLocal) + baseCountOffset));
                Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                Mul(quot, quot, quotCount, pregLoop);
                Add(quot, quot, oriQuotMean, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                DataCopy(((__local_mem__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                    pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            BinaryAddVF(
                tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop, pregLoop,
                aLoopOffset, quot, rem, quotCount, remCount);
            DataCopy(resVar, ((__local_mem__ float*)tmpUbAddr + aLoopOffset));
            Muls(resVar, resVar, scaleCorrection, pregLoop);
            DataCopy(((__local_mem__ float*)batchRstdTensorAddr + aLoopOffset), resVar, pregLoop);
        }
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CalculateRunningMeanVarWithRstdVF(
    __local_mem__ float* batchMeanInUb, __local_mem__ float* batchRstdInUb,
    __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr, __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr,
    __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr, __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr,
    uint16_t currentANum, uint16_t aLoop, uint32_t vectorLen, float besselCorrection, float momentum, float oneSubMomentum, float epsilon)
{
    __VEC_SCOPE__
    {
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> one;
        RegTensor<float> runningMean;
        RegTensor<float> saveMean;
        RegTensor<float> runningVar;
        RegTensor<float> saveVar;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        RegTensor<float> r;
        RegTensor<float> y;
        RegTensor<float> s;
        RegTensor<float> t;
        RegTensor<float> scalar1;
        RegTensor<float> scalarInf;
        RegTensor<float> scalarZero;
        RegTensor<float> t1;
        RegTensor<float> t3;
        RegTensor<float> t4;
        RegTensor<float> rstd;
        MaskReg cmpRegZero;
        MaskReg cmpRegInf;
        MaskReg pregLoop;

        Duplicate(one, 1.0, pregMain);
        uint32_t sreg2 = currentANum;
        for (uint16_t k = 0; k < aLoop; k++) {
            pregLoop = UpdateMask<float>(sreg2);
            Duplicate(scalar1, float(0.5), pregLoop);
            Duplicate(scalarInf, BATCH_NORM_V3_POS_INF, pregLoop);
            Duplicate(scalarZero, float(0.0), pregLoop);
            Duplicate(t1, float(1.5), pregLoop);
            Duplicate(s, float(1.0), pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningVarTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningVarTmp, ((__local_mem__ T_RUNNING_MEAN*)runningVarInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningVar, runningVarTmp, pregLoop);
            } else {
                DataCopy(runningVar, ((__local_mem__ float*)runningVarInUbAddr + k * vectorLen));
            }
            DataCopy(var, ((__local_mem__ float*)batchRstdInUb + k * vectorLen));
            Muls(saveVar, var, besselCorrection, pregLoop);
            Muls(saveVar, saveVar, momentum, pregLoop);
            Muls(runningVar, runningVar, oneSubMomentum, pregLoop);
            Add(saveVar, saveVar, runningVar, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveVarTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveVarTmp, saveVar, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningVarOutUbAddr + k * vectorLen), saveVarTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningVarOutUbAddr + k * vectorLen), saveVar, pregLoop);
            }

            Adds(var, var, epsilon, pregLoop);
            Div(r, one, var, pregLoop);
            Sqrt(y, r, pregLoop);
            Muls(t, var, float(-0.5), pregLoop);
            Mul(t, t, y, pregLoop);
            Mula(t1, t, y, pregLoop);
            Mul(rstd, y, t1, pregLoop);
            Muls(t3, var, float(-1.0), pregLoop);
            Mula(s, t3, r, pregLoop);
            Muls(t4, rstd, float(-1.0), pregLoop);
            Mula(r, t4, rstd, pregLoop);
            Mula(s, var, r, pregLoop);
            Mul(s, s, rstd, pregLoop);
            Mula(rstd, s, scalar1, pregLoop);
            CompareScalar(cmpRegZero, var, BATCH_NORM_V3_POS_INF, pregLoop);
            Select(rstd, scalarZero, rstd, cmpRegZero);
            CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
            Select(rstd, scalarInf, rstd, cmpRegInf);
            DataCopy(((__local_mem__ float*)batchRstdInUb + k * vectorLen), rstd, pregLoop);

            DataCopy(mean, ((__local_mem__ float*)batchMeanInUb + k * vectorLen));
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningMeanTmp, ((__local_mem__ T_RUNNING_MEAN*)runningMeanInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningMean, runningMeanTmp, pregLoop);
            } else {
                DataCopy(runningMean, ((__local_mem__ float*)runningMeanInUbAddr + k * vectorLen));
            }
            Muls(saveMean, mean, momentum, pregLoop);
            Muls(runningMean, runningMean, oneSubMomentum, pregLoop);
            Add(saveMean, saveMean, runningMean, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveMeanTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveMeanTmp, saveMean, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningMeanOutUbAddr + k * vectorLen), saveMeanTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningMeanOutUbAddr + k * vectorLen), saveMean, pregLoop);
            }
        }
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CalculateRunningMeanVarVF(
    __local_mem__ float* batchMeanInUb, __local_mem__ float* batchVarInUb,
    __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr, __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr,
    __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr, __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr,
    uint16_t currentANum, uint16_t aLoop, uint32_t vectorLen, float unbiasedEstimationCoeff, float momentum, float momentumReverse)
{
    __VEC_SCOPE__
    {
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> runningMean;
        RegTensor<float> saveMean;
        RegTensor<float> runningVar;
        RegTensor<float> saveVar;
        MaskReg pregLoop;
        uint32_t sreg2 = currentANum;
        for (uint16_t k = 0; k < aLoop; k++) {
            pregLoop = UpdateMask<float>(sreg2);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningVarTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningVarTmp, ((__local_mem__ T_RUNNING_MEAN*)runningVarInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningVar, runningVarTmp, pregLoop);
            } else {
                DataCopy(runningVar, ((__local_mem__ float*)runningVarInUbAddr + k * vectorLen));
            }
            DataCopy(var, ((__local_mem__ float*)batchVarInUb + k * vectorLen));
            Muls(saveVar, var, unbiasedEstimationCoeff, pregLoop);
            Muls(saveVar, saveVar, momentum, pregLoop);
            Muls(runningVar, runningVar, momentumReverse, pregLoop);
            Add(saveVar, saveVar, runningVar, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveVarTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveVarTmp, saveVar, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningVarOutUbAddr + k * vectorLen), saveVarTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningVarOutUbAddr + k * vectorLen), saveVar, pregLoop);
            }

            DataCopy(mean, ((__local_mem__ float*)batchMeanInUb + k * vectorLen));
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                DataCopy<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningMeanTmp, ((__local_mem__ T_RUNNING_MEAN*)runningMeanInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningMean, runningMeanTmp, pregLoop);
            } else {
                DataCopy(runningMean, ((__local_mem__ float*)runningMeanInUbAddr + k * vectorLen));
            }
            Muls(saveMean, mean, momentum, pregLoop);
            Muls(runningMean, runningMean, momentumReverse, pregLoop);
            Add(saveMean, saveMean, runningMean, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveMeanTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveMeanTmp, saveMean, pregLoop);
                DataCopy<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_RUNNING_MEAN*)runningMeanOutUbAddr + k * vectorLen), saveMeanTmp, pregLoop);
            } else {
                DataCopy(((__local_mem__ float*)runningMeanOutUbAddr + k * vectorLen), saveMean, pregLoop);
            }
        }
    }
}

} // namespace BatchNormV3Ops
#endif // BATCH_NORM_V3_REGBASE_COMMON_H
