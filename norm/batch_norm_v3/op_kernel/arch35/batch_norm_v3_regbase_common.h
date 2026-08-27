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
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

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
constexpr static int64_t MERGED_QUE_NODE_NUM = 2;
constexpr static float BATCH_NORM_V3_POS_INF = 3.40282366920938E+38;

constexpr static AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

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
    params.remainderTailOffset1 = (BATCH_NORM_V3_ROW_ONE > params.remainderTailCount) ?
                                      params.validNumInXUb :
                                      params.remainderOffset + params.aLength;
    params.remainderTailOffset2 = (BATCH_NORM_V3_ROW_TWO_OFFSET > params.remainderTailCount) ?
                                      params.validNumInXUb :
                                      params.remainderOffset + BATCH_NORM_V3_ROW_TWO_OFFSET * params.aLength;
    params.remainderTailOffset3 = (BATCH_NORM_V3_ROW_THREE_OFFSET > params.remainderTailCount) ?
                                      params.validNumInXUb :
                                      params.remainderOffset + BATCH_NORM_V3_ROW_THREE_OFFSET * params.aLength;
    return params;
}

template <typename T_SRC>
__aicore__ inline void LoadOneTensorForDtypeT(__ubuf__ T_SRC* input, RegTensor<float>& dst, MaskReg& preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T_SRC, half>::value) {
        RegTensor<half> xFp16;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__ubuf__ half*)(input) + offset));
        Cast<float, half, NormCommon::castTraitB162B32>(dst, xFp16, preg);
    } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__ubuf__ bfloat16_t*)(input) + offset));
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst, xBf16, preg);
    } else {
        LoadAlign(dst, ((__ubuf__ float*)(input) + offset));
    }
}

template <typename T_SRC>
__aicore__ inline void GatherParamForDtypeT(__ubuf__ T_SRC* src, RegTensor<float>& dst,
                                            RegTensor<uint32_t>& paramOffset, MaskReg& preg, uint32_t calcLen)
{
    if constexpr (IsSameType<T_SRC, float>::value) {
        AscendC::Reg::Gather(dst, (__ubuf__ float*)src, paramOffset, preg);
    } else {
        MaskReg pregSrc = AscendC::Reg::UpdateMask<T_SRC>(calcLen);
        RegTensor<uint16_t> paramOffsetB16;
        RegTensor<T_SRC> srcB16;
        RegTensor<T_SRC> srcB16Unpack;
        AscendC::Reg::Pack(paramOffsetB16, paramOffset);
        AscendC::Reg::Gather(srcB16, ((__ubuf__ T_SRC*)src), paramOffsetB16, pregSrc);
        AscendC::Reg::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
        AscendC::Reg::Cast<float, T_SRC, castTraitB162B32>(dst, srcB16Unpack, preg);
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void GatherRunningParamForDtypeT(__ubuf__ T_RUNNING_MEAN* src, RegTensor<float>& dst,
                                                   RegTensor<uint32_t>& paramOffset, MaskReg& preg, uint32_t calcLen)
{
    if constexpr (IsSameType<T_RUNNING_MEAN, float>::value) {
        AscendC::Reg::Gather(dst, (__ubuf__ float*)src, paramOffset, preg);
    } else {
        MaskReg pregSrc = AscendC::Reg::UpdateMask<T_RUNNING_MEAN>(calcLen);
        RegTensor<uint16_t> paramOffsetB16;
        RegTensor<T_RUNNING_MEAN> srcB16;
        RegTensor<T_RUNNING_MEAN> srcB16Unpack;
        AscendC::Reg::Pack(paramOffsetB16, paramOffset);
        AscendC::Reg::Gather(srcB16, ((__ubuf__ T_RUNNING_MEAN*)src), paramOffsetB16, pregSrc);
        AscendC::Reg::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
        AscendC::Reg::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(dst, srcB16Unpack, preg);
    }
}

template <typename T_GAMMA, typename T_RUNNING_MEAN, int32_t QUEUE_DEPTH>
__aicore__ inline void CopyInBetaGammaMeanVar(bool needCopy, int64_t offset, int64_t curTileALen,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& betaQueue,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& gammaQueue,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& meanQueue,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& varQueue,
                                              GlobalTensor<T_GAMMA>& betaGm, GlobalTensor<T_GAMMA>& gammaGm,
                                              GlobalTensor<T_RUNNING_MEAN>& meanGm, GlobalTensor<T_RUNNING_MEAN>& varGm)
{
    LocalTensor<T_GAMMA> betaLocal = betaQueue.template AllocTensor<T_GAMMA>();
    LocalTensor<T_GAMMA> gammaLocal = gammaQueue.template AllocTensor<T_GAMMA>();
    LocalTensor<T_RUNNING_MEAN> meanLocal = meanQueue.template AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> varLocal = varQueue.template AllocTensor<T_RUNNING_MEAN>();

    if (needCopy) {
        DataCopyExtParams extParam;
        extParam.blockCount = 1;

        extParam.blockLen = curTileALen * sizeof(T_GAMMA);

        DataCopyPadExtParams<T_GAMMA> padExtParam;
        padExtParam.isPad = false;

        DataCopyPad(betaLocal, betaGm[offset], extParam, padExtParam);
        DataCopyPad(gammaLocal, gammaGm[offset], extParam, padExtParam);

        extParam.blockLen = curTileALen * sizeof(T_RUNNING_MEAN);

        DataCopyPadExtParams<T_RUNNING_MEAN> padExtParams1;
        padExtParams1.isPad = false;

        DataCopyPad(meanLocal, meanGm[offset], extParam, padExtParams1);
        DataCopyPad(varLocal, varGm[offset], extParam, padExtParams1);
    }

    betaQueue.EnQue(betaLocal);
    gammaQueue.EnQue(gammaLocal);
    meanQueue.EnQue(meanLocal);
    varQueue.EnQue(varLocal);
}

template <typename T_GAMMA, typename T_RUNNING_MEAN, int32_t QUEUE_DEPTH>
__aicore__ inline void CopyInBetaGammaMeanVar(int64_t offset, int64_t curTileALen,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& betaQueue,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& gammaQueue,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& meanQueue,
                                              TQue<QuePosition::VECIN, QUEUE_DEPTH>& varQueue,
                                              GlobalTensor<T_GAMMA>& betaGm, GlobalTensor<T_GAMMA>& gammaGm,
                                              GlobalTensor<T_RUNNING_MEAN>& meanGm, GlobalTensor<T_RUNNING_MEAN>& varGm)
{
    LocalTensor<T_GAMMA> betaLocal = betaQueue.template AllocTensor<T_GAMMA>();
    LocalTensor<T_GAMMA> gammaLocal = gammaQueue.template AllocTensor<T_GAMMA>();
    LocalTensor<T_RUNNING_MEAN> meanLocal = meanQueue.template AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> varLocal = varQueue.template AllocTensor<T_RUNNING_MEAN>();

    DataCopyExtParams extParam;
    extParam.blockCount = 1;

    extParam.blockLen = curTileALen * sizeof(T_GAMMA);

    DataCopyPadExtParams<T_GAMMA> padExtParam;
    padExtParam.isPad = false;

    DataCopyPad(betaLocal, betaGm[offset], extParam, padExtParam);
    DataCopyPad(gammaLocal, gammaGm[offset], extParam, padExtParam);

    DataCopyPadExtParams<T_RUNNING_MEAN> padExtParams1;
    padExtParams1.isPad = false;

    extParam.blockLen = curTileALen * sizeof(T_RUNNING_MEAN);

    DataCopyPad(meanLocal, meanGm[offset], extParam, padExtParams1);
    DataCopyPad(varLocal, varGm[offset], extParam, padExtParams1);

    betaQueue.EnQue(betaLocal);
    gammaQueue.EnQue(gammaLocal);
    meanQueue.EnQue(meanLocal);
    varQueue.EnQue(varLocal);
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
__aicore__ inline void LoadTwoTensorForDtypeT(__ubuf__ T_SRC* src1, __ubuf__ T_SRC* src2, RegTensor<float>& dst1,
                                              RegTensor<float>& dst2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                              uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T_SRC, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__ubuf__ half*)(src1) + src1Offset));
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__ubuf__ half*)(src2) + src2Offset));
        Cast<float, half, NormCommon::castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, NormCommon::castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16Q;
        RegTensor<bfloat16_t> xBf16R;
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16Q, ((__ubuf__ bfloat16_t*)(src1) + src1Offset));
        LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16R, ((__ubuf__ bfloat16_t*)(src2) + src2Offset));
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst1, xBf16Q, dst1Preg);
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst2, xBf16R, dst2Preg);
    } else {
        LoadAlign(dst1, ((__ubuf__ float*)(src1) + src1Offset));
        LoadAlign(dst2, ((__ubuf__ float*)(src2) + src2Offset));
    }
}

template <typename T_SRC>
__aicore__ inline void LoadTwoTensorForDtypeTBrc(__ubuf__ T_SRC* src1, __ubuf__ T_SRC* src2, RegTensor<float>& dst1,
                                                 RegTensor<float>& dst2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                                 uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T_SRC, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(xFp16Q, ((__ubuf__ half*)(src1) + src1Offset));
        LoadAlign<half, LoadDist::DIST_BRC_B16>(xFp16R, ((__ubuf__ half*)(src2) + src2Offset));
        Cast<float, half, NormCommon::castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, NormCommon::castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xBf16Q;
        RegTensor<bfloat16_t> xBf16R;
        LoadAlign<bfloat16_t, LoadDist::DIST_BRC_B16>(xBf16Q, ((__ubuf__ bfloat16_t*)(src1) + src1Offset));
        LoadAlign<bfloat16_t, LoadDist::DIST_BRC_B16>(xBf16R, ((__ubuf__ bfloat16_t*)(src2) + src2Offset));
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst1, xBf16Q, dst1Preg);
        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst2, xBf16R, dst2Preg);
    } else {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst1, ((__ubuf__ float*)(src1) + src1Offset));
        LoadAlign<float, LoadDist::DIST_BRC_B32>(dst2, ((__ubuf__ float*)(src2) + src2Offset));
    }
}

template <typename T_GAMMA>
__aicore__ inline void CopyInGammaBetaPad(LocalTensor<T_GAMMA>& gammaInUb, LocalTensor<T_GAMMA>& betaInUb,
                                          GlobalTensor<T_GAMMA>& gammaGm, GlobalTensor<T_GAMMA>& betaGm,
                                          int64_t gmOffset, uint32_t currentA)
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
                                             GlobalTensor<T_GAMMA>& betaGm, GlobalTensor<T_GAMMA>& gammaGm,
                                             int64_t gmOffset, int64_t currentA)
{
    LocalTensor<T_GAMMA> betaInUb = betaQueue.template AllocTensor<T_GAMMA>();
    LocalTensor<T_GAMMA> gammaInUb = gammaQueue.template AllocTensor<T_GAMMA>();
    CopyInGammaBetaPad(gammaInUb, betaInUb, gammaGm, betaGm, gmOffset, static_cast<uint32_t>(currentA));
    betaQueue.EnQue(betaInUb);
    gammaQueue.EnQue(gammaInUb);
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CopyInRunningMeanVarPad(LocalTensor<T_RUNNING_MEAN>& runningMeanInUb,
                                               LocalTensor<T_RUNNING_MEAN>& runningVarInUb,
                                               GlobalTensor<T_RUNNING_MEAN>& runningMeanGm,
                                               GlobalTensor<T_RUNNING_MEAN>& runningVarGm, int64_t gmOffset,
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
__aicore__ inline void CopyInRunningMeanVarCommon(MeanQueue& runningMeanInQueue, VarQueue& runningVarInQueue,
                                                  GlobalTensor<T_RUNNING_MEAN>& runningMeanGm,
                                                  GlobalTensor<T_RUNNING_MEAN>& runningVarGm, int64_t gmOffset,
                                                  int64_t currentA)
{
    LocalTensor<T_RUNNING_MEAN> runningMeanInUb = runningMeanInQueue.template AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningVarInUb = runningVarInQueue.template AllocTensor<T_RUNNING_MEAN>();
    CopyInRunningMeanVarPad(runningMeanInUb, runningVarInUb, runningMeanGm, runningVarGm, gmOffset,
                            static_cast<uint32_t>(currentA));
    runningMeanInQueue.EnQue(runningMeanInUb);
    runningVarInQueue.EnQue(runningVarInUb);
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CopyOutRunningMeanVarPad(LocalTensor<T_RUNNING_MEAN>& runningMeanOutUb,
                                                LocalTensor<T_RUNNING_MEAN>& runningVarOutUb,
                                                GlobalTensor<T_RUNNING_MEAN>& runningMeanOutGm,
                                                GlobalTensor<T_RUNNING_MEAN>& runningVarOutGm, int64_t gmOffset,
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
__aicore__ inline void CalculateRunningMeanVarVF(__ubuf__ float* batchMeanInUb, __ubuf__ float* batchVarInUb,
                                                 __ubuf__ T_RUNNING_MEAN* runningMeanInUbAddr,
                                                 __ubuf__ T_RUNNING_MEAN* runningVarInUbAddr,
                                                 __ubuf__ T_RUNNING_MEAN* runningMeanOutUbAddr,
                                                 __ubuf__ T_RUNNING_MEAN* runningVarOutUbAddr, uint16_t currentANum,
                                                 uint16_t aLoop, uint32_t vectorLen, float unbiasedEstimationCoeff,
                                                 float momentum, float momentumReverse);

template <typename T_RUNNING_MEAN>
__aicore__ inline void UpdateRunningMeanVarCommon(
    LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
    TQue<QuePosition::VECIN, 1>& runningMeanVarInQueue, TQue<QuePosition::VECOUT, 1>& runningMeanVarOutQueue,
    GlobalTensor<T_RUNNING_MEAN>& runningMeanGm, GlobalTensor<T_RUNNING_MEAN>& runningVarGm,
    GlobalTensor<T_RUNNING_MEAN>& runningMeanOutGm, GlobalTensor<T_RUNNING_MEAN>& runningVarOutGm, int64_t gmOffset,
    uint32_t currentA, uint16_t aLoop, uint32_t vectorLen, float unbiasedEstimationCoeff, float momentum,
    float momentumReverse, int64_t runningHalf)
{
    // running mean/var 的 in 对、out 对各自同生命周期,分别合并为一个 que:一次 alloc,var 在 mean 之上按对齐偏移
    LocalTensor<T_RUNNING_MEAN> runningMeanVarInTensor = runningMeanVarInQueue.AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningMeanInTensor = runningMeanVarInTensor;
    LocalTensor<T_RUNNING_MEAN> runningVarInTensor = runningMeanVarInTensor[runningHalf];
    CopyInRunningMeanVarPad(runningMeanInTensor, runningVarInTensor, runningMeanGm, runningVarGm, gmOffset, currentA);
    runningMeanVarInQueue.EnQue(runningMeanVarInTensor);
    runningMeanVarInTensor = runningMeanVarInQueue.template DeQue<T_RUNNING_MEAN>();
    runningMeanInTensor = runningMeanVarInTensor;
    runningVarInTensor = runningMeanVarInTensor[runningHalf];

    LocalTensor<T_RUNNING_MEAN> runningMeanVarOutTensor = runningMeanVarOutQueue.AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningMeanOutTensor = runningMeanVarOutTensor;
    LocalTensor<T_RUNNING_MEAN> runningVarOutTensor = runningMeanVarOutTensor[runningHalf];
    __ubuf__ T_RUNNING_MEAN* runningMeanInUbAddr = (__ubuf__ T_RUNNING_MEAN*)runningMeanInTensor.GetPhyAddr();
    __ubuf__ T_RUNNING_MEAN* runningVarInUbAddr = (__ubuf__ T_RUNNING_MEAN*)runningVarInTensor.GetPhyAddr();
    __ubuf__ T_RUNNING_MEAN* runningMeanOutUbAddr = (__ubuf__ T_RUNNING_MEAN*)runningMeanOutTensor.GetPhyAddr();
    __ubuf__ T_RUNNING_MEAN* runningVarOutUbAddr = (__ubuf__ T_RUNNING_MEAN*)runningVarOutTensor.GetPhyAddr();
    __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
    __ubuf__ float* batchRstdTensorAddr = (__ubuf__ float*)batchRstdTensor.GetPhyAddr();
    CalculateRunningMeanVarVF<T_RUNNING_MEAN>(batchMeanTensorAddr, batchRstdTensorAddr, runningMeanInUbAddr,
                                              runningVarInUbAddr, runningMeanOutUbAddr, runningVarOutUbAddr,
                                              static_cast<uint16_t>(currentA), aLoop, vectorLen,
                                              unbiasedEstimationCoeff, momentum, momentumReverse);

    runningMeanVarInQueue.FreeTensor(runningMeanVarInTensor);
    runningMeanVarOutQueue.EnQue(runningMeanVarOutTensor);
    runningMeanVarOutTensor = runningMeanVarOutQueue.template DeQue<T_RUNNING_MEAN>();
    runningMeanOutTensor = runningMeanVarOutTensor;
    runningVarOutTensor = runningMeanVarOutTensor[runningHalf];
    CopyOutRunningMeanVarPad(runningMeanOutTensor, runningVarOutTensor, runningMeanOutGm, runningVarOutGm, gmOffset,
                             currentA);
    runningMeanVarOutQueue.FreeTensor(runningMeanVarOutTensor);
}

__aicore__ inline void CopyOutBatchMeanRstdPad(LocalTensor<float>& batchMeanInUb, LocalTensor<float>& batchRstdInUb,
                                               GlobalTensor<float>& batchMeanGm, GlobalTensor<float>& batchRstdGm,
                                               int64_t gmOffset, uint32_t currentA)
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
__aicore__ inline void CopyOutBatchMeanRstdCommon(MeanQueue& batchMeanQueue, RstdQueue& batchRstdQueue,
                                                  GlobalTensor<float>& batchMeanGm, GlobalTensor<float>& batchRstdGm,
                                                  int64_t gmOffset, int64_t currentA)
{
    LocalTensor<float> batchMeanInUb = batchMeanQueue.template DeQue<float>();
    LocalTensor<float> batchRstdInUb = batchRstdQueue.template DeQue<float>();
    CopyOutBatchMeanRstdPad(batchMeanInUb, batchRstdInUb, batchMeanGm, batchRstdGm, gmOffset,
                            static_cast<uint32_t>(currentA));
    batchMeanQueue.FreeTensor(batchMeanInUb);
    batchRstdQueue.FreeTensor(batchRstdInUb);
}

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
__aicore__ inline void WelfordParallelUpdateVF(__ubuf__ T_SRC* x1Local, __ubuf__ float* tmpMeanLocal,
                                               __ubuf__ float* tmpVarLocal, uint64_t calLen, uint16_t loopCount,
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
                LoadAlign(tmpMean, tmpMeanLocal + offset);
            }
            Sub(delta1, x1, tmpMean, pregLoop);
            Muls(delta2, delta1, scale, pregLoop);
            Add(tmpMean, tmpMean, delta2, pregLoop);
            StoreAlign(tmpMeanLocal + offset, tmpMean, pregLoop);

            if constexpr (INIT) {
                Duplicate(tmpVar, 0.0, pregLoop);
            } else {
                LoadAlign(tmpVar, tmpVarLocal + offset);
            }
            Sub(delta3, x1, tmpMean, pregLoop);
            Mul(delta4, delta1, delta3, pregLoop);
            Add(tmpVar, tmpVar, delta4, pregLoop);
            StoreAlign(tmpVarLocal + offset, tmpVar, pregLoop);
        }
    }
}

__aicore__ inline void BatchNormV3MeanM2TensorInit(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                                   uint32_t len, uint16_t loopCount, uint32_t vectorLen)
{
    __ubuf__ float* meanTensorAddr = (__ubuf__ float*)meanTensor.GetPhyAddr();
    __ubuf__ float* m2TensorAddr = (__ubuf__ float*)m2Tensor.GetPhyAddr();
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
            StoreAlign(meanTensorAddr + offset, tmpMean, mask1);
            StoreAlign(m2TensorAddr + offset, tmpM2, mask1);
        }
    }
}

// USE_ADDR_REG=false(默认): 原始仿射寻址,ra_welford / ra_full_reduce / LastFinalize 等调用点逐字节不变。
// USE_ADDR_REG=true : 仅 BlockSplitR 启用。本函数被调用者的 aIndex 循环包住,内层再有 i/j 两层 ⇒ 嵌套深度 3,
//                     故 CreateAddrReg 必须给三个 stride;每个 k 轮都从同一 base 重新起算,i 维 stride 取 0。
//                     启用时需由调用方把 aIndex 与其步长传进来(地址寄存器只能用循环索引变量,不能用算好的 offset)。
template <bool USE_ADDR_REG = false>
__aicore__ inline void BinaryAddVF(__ubuf__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint16_t binaryAddKLoop,
                                   uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop,
                                   uint32_t offset, RegTensor<float>& x1, RegTensor<float>& x2, RegTensor<float>& x3,
                                   RegTensor<float>& x4, uint16_t aIndex = 0, uint32_t aStride = 0)
{
    uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
    for (uint16_t i = 0; i < binaryAddKLoop; i++) {
        curBinaryAddInnerLoop = curBinaryAddInnerLoop / BATCH_NORM_V3_ROW_FOUR_OFFSET;
        for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
            if constexpr (USE_ADDR_REG) {
                AscendC::Reg::AddrReg rdAddr = AscendC::Reg::CreateAddrReg<float>(
                    aIndex, aStride, i, 0, j, BATCH_NORM_V3_ROW_FOUR_OFFSET * rLoopStride);
                AscendC::Reg::AddrReg wrAddr = AscendC::Reg::CreateAddrReg<float>(aIndex, aStride, i, 0, j,
                                                                                  rLoopStride);
                AscendC::Reg::LoadAlign(x1, (__ubuf__ float*)binaryAddTmpAddr, rdAddr);
                AscendC::Reg::LoadAlign(x2, (__ubuf__ float*)(binaryAddTmpAddr + rLoopStride), rdAddr);
                Add(x1, x1, x2, pregLoop);
                AscendC::Reg::LoadAlign(
                    x3, (__ubuf__ float*)(binaryAddTmpAddr + BATCH_NORM_V3_ROW_TWO_OFFSET * rLoopStride), rdAddr);
                AscendC::Reg::LoadAlign(
                    x4, (__ubuf__ float*)(binaryAddTmpAddr + BATCH_NORM_V3_ROW_THREE_OFFSET * rLoopStride), rdAddr);
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                AscendC::Reg::StoreAlign((__ubuf__ float*)binaryAddTmpAddr, x1, wrAddr, pregLoop);
                continue;
            }
            LoadAlign(x1,
                      ((__ubuf__ float*)binaryAddTmpAddr + (j * BATCH_NORM_V3_ROW_FOUR_OFFSET) * rLoopStride + offset));
            LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + (j * BATCH_NORM_V3_ROW_FOUR_OFFSET + 1) * rLoopStride +
                           offset));
            Add(x1, x1, x2, pregLoop);
            LoadAlign(x3, ((__ubuf__ float*)binaryAddTmpAddr +
                           (j * BATCH_NORM_V3_ROW_FOUR_OFFSET + BATCH_NORM_V3_ROW_TWO_OFFSET) * rLoopStride + offset));
            LoadAlign(x4,
                      ((__ubuf__ float*)binaryAddTmpAddr +
                       (j * BATCH_NORM_V3_ROW_FOUR_OFFSET + BATCH_NORM_V3_ROW_THREE_OFFSET) * rLoopStride + offset));
            Add(x3, x3, x4, pregLoop);
            Add(x1, x1, x3, pregLoop);
            StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
        LoadAlign(x1, ((__ubuf__ float*)binaryAddTmpAddr + offset));
        LoadAlign(x2, ((__ubuf__ float*)binaryAddTmpAddr + rLoopStride + offset));
        Add(x1, x1, x2, pregLoop);
        StoreAlign(((__ubuf__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

__aicore__ inline void FillCountBlock(__ubuf__ float* dst, RegTensor<float>& tmpCount, MaskReg& pregMain,
                                      MaskReg& pregLoop, float addCount, uint32_t num, uint16_t loopCount, uint32_t vl)
{
    uint32_t sreg = num;
    Duplicate(tmpCount, addCount, pregMain);
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = AscendC::Reg::UpdateMask<float>(sreg);
        StoreAlign(dst + i * vl, tmpCount, pregLoop);
    }
}

__aicore__ inline void TwoRowAddMeanWithTail(RegTensor<float>& dst, __ubuf__ float* input, MaskReg& preg,
                                             uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                             RegTensor<float>& rem, RegTensor<float>& nextRow,
                                             RegTensor<float>& remNextRow, float n)
{
    LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
    LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
    Muls(dst, dst, n, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
    LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset3)));
    LoadAlign(remNextRow, ((__ubuf__ float*)(input) + (offset4)));
    Muls(nextRow, nextRow, n, preg);
    Muls(remNextRow, remNextRow, n, preg);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddMean(RegTensor<float>& dst, __ubuf__ float* input, MaskReg& preg, uint32_t offset1,
                                     uint32_t offset2, RegTensor<float>& nextRow, float n)
{
    LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
    LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset2)));
    Muls(dst, dst, n, preg);
    Muls(nextRow, nextRow, n, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddVarWithTail(RegTensor<float>& dst, __ubuf__ float* input, MaskReg& preg,
                                            uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                            RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& nextRow,
                                            RegTensor<float>& remNextRow, float n)
{
    LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
    LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
    Sub(dst, dst, mean, preg);
    Sub(rem, rem, mean, preg);
    Mul(dst, dst, dst, preg);
    Mul(rem, rem, rem, preg);
    Muls(dst, dst, n, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
    LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset3)));
    LoadAlign(remNextRow, ((__ubuf__ float*)(input) + (offset4)));
    Sub(nextRow, nextRow, mean, preg);
    Sub(remNextRow, remNextRow, mean, preg);
    Mul(nextRow, nextRow, nextRow, preg);
    Mul(remNextRow, remNextRow, remNextRow, preg);
    Muls(nextRow, nextRow, n, preg);
    Muls(remNextRow, remNextRow, n, preg);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddVar(RegTensor<float>& dst, __ubuf__ float* input, MaskReg& preg, uint32_t offset1,
                                    uint32_t offset2, RegTensor<float>& mean, RegTensor<float>& nextRow, float n)
{
    LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
    LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset2)));
    Sub(dst, dst, mean, preg);
    Sub(nextRow, nextRow, mean, preg);
    Mul(dst, dst, dst, preg);
    Mul(nextRow, nextRow, nextRow, preg);
    Muls(dst, dst, n, preg);
    Muls(nextRow, nextRow, n, preg);
    Add(dst, dst, nextRow, preg);
}

template <bool CALC_VAR, uint32_t SCALE_COEF>
__aicore__ inline void CalculateRLessThanVF(__ubuf__ float* xInUb, __ubuf__ float* batchMeanInUbAddr,
                                            __ubuf__ float* batchVarOutUbAddr, int64_t currentA,
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
                LoadAlign(mean, ((__ubuf__ float*)batchMeanInUbAddr + aLoopOffset));
                StoreAlign(((__ubuf__ float*)xInUb + params.validNumInXUb + aLoopOffset), mean, pregLoop);
            } else {
                StoreAlign(((__ubuf__ float*)xInUb + params.validNumInXUb + aLoopOffset), zero, pregLoop);
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
                StoreAlign(((__ubuf__ float*)batchVarOutUbAddr + aLoopOffset), x1, pregLoop);
            } else {
                StoreAlign(((__ubuf__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }
}

__aicore__ inline void TwoRowAddPartialMean(RegTensor<float>& dst, __ubuf__ float* input, __ubuf__ float* tCount,
                                            MaskReg& preg, uint32_t offset1, uint32_t offset2, uint32_t offset5,
                                            uint32_t offset6, RegTensor<float>& rem, RegTensor<float>& dstCount,
                                            RegTensor<float>& remCount, float n)
{
    LoadAlign(dst, ((__ubuf__ float*)(input) + (offset1)));
    LoadAlign(rem, ((__ubuf__ float*)(input) + (offset2)));
    LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
    LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
    Mul(dst, dst, dstCount, preg);
    Mul(rem, rem, remCount, preg);
    Muls(dst, dst, n, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
}

__aicore__ inline void TwoRowAddPartialMeanWithTail(RegTensor<float>& dst, __ubuf__ float* input,
                                                    __ubuf__ float* tCount, MaskReg& preg, uint32_t offset1,
                                                    uint32_t offset2, uint32_t offset3, uint32_t offset4,
                                                    uint32_t offset5, uint32_t offset6, uint32_t offset7,
                                                    uint32_t offset8, RegTensor<float>& rem, RegTensor<float>& nextRow,
                                                    RegTensor<float>& remNextRow, RegTensor<float>& dstCount,
                                                    RegTensor<float>& remCount, RegTensor<float>& nextRowCount,
                                                    RegTensor<float>& remNextRowCount, float n)
{
    TwoRowAddPartialMean(dst, input, tCount, preg, offset1, offset2, offset5, offset6, rem, dstCount, remCount, n);
    LoadAlign(nextRow, ((__ubuf__ float*)(input) + (offset3)));
    LoadAlign(remNextRow, ((__ubuf__ float*)(input) + (offset4)));
    LoadAlign<float, LoadDist::DIST_BRC_B32>(nextRowCount, ((__ubuf__ float*)(tCount) + (offset7)));
    LoadAlign<float, LoadDist::DIST_BRC_B32>(remNextRowCount, ((__ubuf__ float*)(tCount) + (offset8)));
    Mul(nextRow, nextRow, nextRowCount, preg);
    Mul(remNextRow, remNextRow, remNextRowCount, preg);
    Muls(nextRow, nextRow, n, preg);
    Muls(remNextRow, remNextRow, n, preg);
    Add(nextRow, nextRow, remNextRow, preg);
    Add(dst, dst, nextRow, preg);
}

__aicore__ inline void TwoRowAddPartialVar(RegTensor<float>& dst, __ubuf__ float* tmpMean, __ubuf__ float* tmpM2,
                                           __ubuf__ float* tCount, MaskReg& preg, uint32_t offset1, uint32_t offset2,
                                           uint32_t offset5, uint32_t offset6, RegTensor<float>& mean,
                                           RegTensor<float>& rem, RegTensor<float>& dstCount,
                                           RegTensor<float>& remCount, RegTensor<float>& dstM2, RegTensor<float>& remM2,
                                           float n)
{
    LoadAlign(dst, ((__ubuf__ float*)(tmpMean) + (offset1)));
    LoadAlign(rem, ((__ubuf__ float*)(tmpMean) + (offset2)));
    LoadAlign<float, LoadDist::DIST_BRC_B32>(dstCount, ((__ubuf__ float*)(tCount) + (offset5)));
    LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tCount) + (offset6)));
    Sub(dst, dst, mean, preg);
    Mul(dst, dst, dst, preg);
    Sub(rem, rem, mean, preg);
    Mul(rem, rem, rem, preg);
    Mul(dst, dst, dstCount, preg);
    Mul(rem, rem, remCount, preg);
    LoadAlign(dstM2, ((__ubuf__ float*)(tmpM2) + (offset1)));
    LoadAlign(remM2, ((__ubuf__ float*)(tmpM2) + (offset2)));
    Add(dst, dstM2, dst, preg);
    Muls(dst, dst, n, preg);
    Add(rem, remM2, rem, preg);
    Muls(rem, rem, n, preg);
    Add(dst, dst, rem, preg);
}

__aicore__ inline void TwoRowAddPartialVarWithTail(
    RegTensor<float>& dst, __ubuf__ float* tmpMean, __ubuf__ float* tmpM2, __ubuf__ float* tCount, MaskReg& preg,
    uint32_t offset1, uint32_t offset2, uint32_t offset3, uint32_t offset4, uint32_t offset5, uint32_t offset6,
    uint32_t offset7, uint32_t offset8, RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& nextRow,
    RegTensor<float>& remNextRow, RegTensor<float>& dstCount, RegTensor<float>& remCount,
    RegTensor<float>& nextRowCount, RegTensor<float>& remNextRowCount, RegTensor<float>& dstM2, RegTensor<float>& remM2,
    RegTensor<float>& nextRowM2, RegTensor<float>& remNextRowM2, float n)
{
    TwoRowAddPartialVar(dst, tmpMean, tmpM2, tCount, preg, offset1, offset2, offset5, offset6, mean, rem, dstCount,
                        remCount, dstM2, remM2, n);
    TwoRowAddPartialVar(nextRow, tmpMean, tmpM2, tCount, preg, offset3, offset4, offset7, offset8, mean, remNextRow,
                        nextRowCount, remNextRowCount, nextRowM2, remNextRowM2, n);
    Add(dst, dst, nextRow, preg);
}

// USE_ADDR_REG=false(默认): 原始仿射寻址,rar_block_split_r 调用点逐字节不变。
// USE_ADDR_REG=true : 仅 BlockSplitR 启用,把 SyncAll 之后这一阶段的 BinaryAddVF 也切到地址寄存器。
//                     此处 aIndex 循环 + BinaryAddVF 内层 i/j 恰好三层,满足三维 CreateAddrReg 的深度要求。
template <bool USE_ADDR_REG = false>
__aicore__ inline void LastFinalizeVF(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
                                      LocalTensor<float>& meanTensor, LocalTensor<float>& varTensor,
                                      LocalTensor<float>& countTensor, LocalTensor<float>& tmpTensor,
                                      uint32_t currentAAlign, uint32_t vectorLen, uint16_t currentA,
                                      uint16_t usedCoreNum, uint16_t lastBinaryAddQuotient, uint16_t lastBinaryAddK,
                                      uint16_t lastBinaryAddLast, float lastNFactor, float lastNCorrectionFactor)
{
    __ubuf__ float* tmpMeanLocal = (__ubuf__ float*)meanTensor.GetPhyAddr();
    __ubuf__ float* tmpCountLocal = (__ubuf__ float*)countTensor.GetPhyAddr();
    __ubuf__ float* tmpVarLocal = (__ubuf__ float*)varTensor.GetPhyAddr();
    __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
    __ubuf__ float* tmpUbAddr = (__ubuf__ float*)tmpTensor.GetPhyAddr();
    __ubuf__ float* batchRstdTensorAddr = (__ubuf__ float*)batchRstdTensor.GetPhyAddr();
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
                LoadAlign(quot, ((__ubuf__ float*)(tmpMeanLocal) + (quotOffset)));
                LoadAlign(rem, ((__ubuf__ float*)(tmpMeanLocal) + (remOffset)));
                LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                         ((__ubuf__ float*)(tmpCountLocal) + quotCountOffset));
                LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tmpCountLocal) + remCountOffset));
                Mul(quot, quot, quotCount, pregLoop);
                Mul(rem, rem, remCount, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                Muls(rem, rem, numScale, pregLoop);
                Add(quot, quot, rem, pregLoop);
                StoreAlign(((__ubuf__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
            }
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                uint32_t baseCountOffset = remainderLoopCount + i;
                LoadAlign(quot, ((__ubuf__ float*)(tmpMeanLocal) + (baseOffset)));
                LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                         ((__ubuf__ float*)(tmpCountLocal) + baseCountOffset));
                Mul(quot, quot, quotCount, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                StoreAlign(((__ubuf__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                           pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            BinaryAddVF<USE_ADDR_REG>(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                                      pregLoop, aLoopOffset, quot, rem, quotCount, remCount, aIndex, vectorLen);
            LoadAlign(resMean, ((__ubuf__ float*)tmpUbAddr + aLoopOffset));
            Muls(resMean, resMean, scaleCorrection, pregLoop);
            StoreAlign(((__ubuf__ float*)batchMeanTensorAddr + aLoopOffset), resMean, pregLoop);
            for (uint16_t i = 0; i < remainderLoopCount; i++) {
                uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                uint32_t quotCountOffset = i;
                uint32_t remCountOffset = i + remainderCountOffset;
                LoadAlign(quot, ((__ubuf__ float*)(tmpVarLocal) + (quotOffset)));
                LoadAlign(rem, ((__ubuf__ float*)(tmpVarLocal) + (remOffset)));
                LoadAlign(oriQuotMean, ((__ubuf__ float*)(tmpMeanLocal) + (quotOffset)));
                LoadAlign(oriRemMean, ((__ubuf__ float*)(tmpMeanLocal) + (remOffset)));
                LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                         ((__ubuf__ float*)(tmpCountLocal) + quotCountOffset));
                LoadAlign<float, LoadDist::DIST_BRC_B32>(remCount, ((__ubuf__ float*)(tmpCountLocal) + remCountOffset));
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
                StoreAlign(((__ubuf__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
            }
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                uint32_t baseCountOffset = remainderLoopCount + i;
                LoadAlign(quot, ((__ubuf__ float*)(tmpVarLocal) + (baseOffset)));
                LoadAlign(oriQuotMean, ((__ubuf__ float*)(tmpMeanLocal) + (baseOffset)));
                LoadAlign<float, LoadDist::DIST_BRC_B32>(quotCount,
                                                         ((__ubuf__ float*)(tmpCountLocal) + baseCountOffset));
                Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                Mul(quot, quot, quotCount, pregLoop);
                Add(quot, quot, oriQuotMean, pregLoop);
                Muls(quot, quot, numScale, pregLoop);
                StoreAlign(((__ubuf__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                           pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            BinaryAddVF<USE_ADDR_REG>(tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                                      pregLoop, aLoopOffset, quot, rem, quotCount, remCount, aIndex, vectorLen);
            LoadAlign(resVar, ((__ubuf__ float*)tmpUbAddr + aLoopOffset));
            Muls(resVar, resVar, scaleCorrection, pregLoop);
            StoreAlign(((__ubuf__ float*)batchRstdTensorAddr + aLoopOffset), resVar, pregLoop);
        }
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CalculateRunningMeanVarWithRstdVF(
    __ubuf__ float* batchMeanInUb, __ubuf__ float* batchRstdInUb, __ubuf__ T_RUNNING_MEAN* runningMeanInUbAddr,
    __ubuf__ T_RUNNING_MEAN* runningVarInUbAddr, __ubuf__ T_RUNNING_MEAN* runningMeanOutUbAddr,
    __ubuf__ T_RUNNING_MEAN* runningVarOutUbAddr, uint16_t currentANum, uint16_t aLoop, uint32_t vectorLen,
    float besselCorrection, float momentum, float oneSubMomentum, float epsilon)
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
                LoadAlign<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningVarTmp, ((__ubuf__ T_RUNNING_MEAN*)runningVarInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningVar, runningVarTmp, pregLoop);
            } else {
                LoadAlign(runningVar, ((__ubuf__ float*)runningVarInUbAddr + k * vectorLen));
            }
            LoadAlign(var, ((__ubuf__ float*)batchRstdInUb + k * vectorLen));
            Muls(saveVar, var, besselCorrection, pregLoop);
            Muls(saveVar, saveVar, momentum, pregLoop);
            Muls(runningVar, runningVar, oneSubMomentum, pregLoop);
            Add(saveVar, saveVar, runningVar, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveVarTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveVarTmp, saveVar, pregLoop);
                StoreAlign<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__ubuf__ T_RUNNING_MEAN*)runningVarOutUbAddr + k * vectorLen), saveVarTmp, pregLoop);
            } else {
                StoreAlign(((__ubuf__ float*)runningVarOutUbAddr + k * vectorLen), saveVar, pregLoop);
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
            Compares(cmpRegZero, var, BATCH_NORM_V3_POS_INF, pregLoop);
            Select(rstd, scalarZero, rstd, cmpRegZero);
            Compares(cmpRegInf, var, float(0.0), pregLoop);
            Select(rstd, scalarInf, rstd, cmpRegInf);
            StoreAlign(((__ubuf__ float*)batchRstdInUb + k * vectorLen), rstd, pregLoop);

            LoadAlign(mean, ((__ubuf__ float*)batchMeanInUb + k * vectorLen));
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                LoadAlign<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningMeanTmp, ((__ubuf__ T_RUNNING_MEAN*)runningMeanInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningMean, runningMeanTmp, pregLoop);
            } else {
                LoadAlign(runningMean, ((__ubuf__ float*)runningMeanInUbAddr + k * vectorLen));
            }
            Muls(saveMean, mean, momentum, pregLoop);
            Muls(runningMean, runningMean, oneSubMomentum, pregLoop);
            Add(saveMean, saveMean, runningMean, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveMeanTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveMeanTmp, saveMean, pregLoop);
                StoreAlign<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__ubuf__ T_RUNNING_MEAN*)runningMeanOutUbAddr + k * vectorLen), saveMeanTmp, pregLoop);
            } else {
                StoreAlign(((__ubuf__ float*)runningMeanOutUbAddr + k * vectorLen), saveMean, pregLoop);
            }
        }
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void CalculateRunningMeanVarVF(__ubuf__ float* batchMeanInUb, __ubuf__ float* batchVarInUb,
                                                 __ubuf__ T_RUNNING_MEAN* runningMeanInUbAddr,
                                                 __ubuf__ T_RUNNING_MEAN* runningVarInUbAddr,
                                                 __ubuf__ T_RUNNING_MEAN* runningMeanOutUbAddr,
                                                 __ubuf__ T_RUNNING_MEAN* runningVarOutUbAddr, uint16_t currentANum,
                                                 uint16_t aLoop, uint32_t vectorLen, float unbiasedEstimationCoeff,
                                                 float momentum, float momentumReverse)
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
                LoadAlign<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningVarTmp, ((__ubuf__ T_RUNNING_MEAN*)runningVarInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningVar, runningVarTmp, pregLoop);
            } else {
                LoadAlign(runningVar, ((__ubuf__ float*)runningVarInUbAddr + k * vectorLen));
            }
            LoadAlign(var, ((__ubuf__ float*)batchVarInUb + k * vectorLen));
            Muls(saveVar, var, unbiasedEstimationCoeff, pregLoop);
            Muls(saveVar, saveVar, momentum, pregLoop);
            Muls(runningVar, runningVar, momentumReverse, pregLoop);
            Add(saveVar, saveVar, runningVar, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveVarTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveVarTmp, saveVar, pregLoop);
                StoreAlign<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__ubuf__ T_RUNNING_MEAN*)runningVarOutUbAddr + k * vectorLen), saveVarTmp, pregLoop);
            } else {
                StoreAlign(((__ubuf__ float*)runningVarOutUbAddr + k * vectorLen), saveVar, pregLoop);
            }

            LoadAlign(mean, ((__ubuf__ float*)batchMeanInUb + k * vectorLen));
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                LoadAlign<T_RUNNING_MEAN, LoadDist::DIST_UNPACK_B16>(
                    runningMeanTmp, ((__ubuf__ T_RUNNING_MEAN*)runningMeanInUbAddr + k * vectorLen));
                Cast<float, T_RUNNING_MEAN, NormCommon::castTraitB162B32>(runningMean, runningMeanTmp, pregLoop);
            } else {
                LoadAlign(runningMean, ((__ubuf__ float*)runningMeanInUbAddr + k * vectorLen));
            }
            Muls(saveMean, mean, momentum, pregLoop);
            Muls(runningMean, runningMean, momentumReverse, pregLoop);
            Add(saveMean, saveMean, runningMean, pregLoop);
            if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                RegTensor<T_RUNNING_MEAN> saveMeanTmp;
                Cast<T_RUNNING_MEAN, float, NormCommon::castTraitB322B16>(saveMeanTmp, saveMean, pregLoop);
                StoreAlign<T_RUNNING_MEAN, StoreDist::DIST_PACK_B32>(
                    ((__ubuf__ T_RUNNING_MEAN*)runningMeanOutUbAddr + k * vectorLen), saveMeanTmp, pregLoop);
            } else {
                StoreAlign(((__ubuf__ float*)runningMeanOutUbAddr + k * vectorLen), saveMean, pregLoop);
            }
        }
    }
}

template <typename T_GAMMA, typename T_RUNNING_MEAN>
__aicore__ inline void VFPrepareParamCache(__ubuf__ T_GAMMA* gammaLocal, __ubuf__ T_GAMMA* betaLocal,
                                           __ubuf__ T_RUNNING_MEAN* meanLocal, __ubuf__ T_RUNNING_MEAN* varLocal,
                                           __ubuf__ uint32_t* offsetLocal, __ubuf__ float* gammaFp32Local,
                                           __ubuf__ float* betaFp32Local, __ubuf__ float* meanFp32Local,
                                           __ubuf__ float* rstdFp32Local, uint32_t paramCacheElemLen, float epsilon)
{
    __VEC_SCOPE__
    {
        RegTensor<float> gamma;
        RegTensor<float> beta;
        RegTensor<float> mean;
        RegTensor<float> var;
        RegTensor<float> rstd;
        RegTensor<uint32_t> paramOffset;
        uint32_t maskLen = paramCacheElemLen;
        MaskReg pregMask = AscendC::Reg::UpdateMask<float>(maskLen);

        AscendC::Reg::LoadAlign<uint32_t, LoadDist::DIST_NORM>(paramOffset, offsetLocal);
        GatherParamForDtypeT(gammaLocal, gamma, paramOffset, pregMask, paramCacheElemLen);
        GatherParamForDtypeT(betaLocal, beta, paramOffset, pregMask, paramCacheElemLen);
        GatherRunningParamForDtypeT(varLocal, var, paramOffset, pregMask, paramCacheElemLen);
        NormCommon::ComputeRstdNewtonRaphsonReg(var, rstd, pregMask, epsilon);
        GatherRunningParamForDtypeT(meanLocal, mean, paramOffset, pregMask, paramCacheElemLen);

        StoreAlign(gammaFp32Local, gamma, pregMask);
        StoreAlign(betaFp32Local, beta, pregMask);
        StoreAlign(meanFp32Local, mean, pregMask);
        StoreAlign(rstdFp32Local, rstd, pregMask);
    }
}

template <typename T_GAMMA, typename T_RUNNING_MEAN, typename QueT, typename BufT>
__aicore__ inline void PrepareParamCache(QueT& betaQueue, QueT& gammaQueue, QueT& meanQueue, QueT& varQueue,
                                         BufT& offsetBuf, BufT& betaFp32Buf, BufT& gammaFp32Buf, BufT& meanFp32Buf,
                                         BufT& rstdFp32Buf, uint32_t paramCacheElemLen, float epsilon)
{
    LocalTensor<T_GAMMA> beta = betaQueue.template DeQue<T_GAMMA>();
    LocalTensor<T_GAMMA> gamma = gammaQueue.template DeQue<T_GAMMA>();
    LocalTensor<T_RUNNING_MEAN> mean = meanQueue.template DeQue<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> var = varQueue.template DeQue<T_RUNNING_MEAN>();
    LocalTensor<uint32_t> offset = offsetBuf.template Get<uint32_t>();
    LocalTensor<float> betaFp32 = betaFp32Buf.template Get<float>();
    LocalTensor<float> gammaFp32 = gammaFp32Buf.template Get<float>();
    LocalTensor<float> meanFp32 = meanFp32Buf.template Get<float>();
    LocalTensor<float> rstdFp32 = rstdFp32Buf.template Get<float>();

    __ubuf__ T_GAMMA* betaLocal = (__ubuf__ T_GAMMA*)beta.GetPhyAddr();
    __ubuf__ T_GAMMA* gammaLocal = (__ubuf__ T_GAMMA*)gamma.GetPhyAddr();
    __ubuf__ T_RUNNING_MEAN* meanLocal = (__ubuf__ T_RUNNING_MEAN*)mean.GetPhyAddr();
    __ubuf__ T_RUNNING_MEAN* varLocal = (__ubuf__ T_RUNNING_MEAN*)var.GetPhyAddr();
    __ubuf__ uint32_t* offsetLocal = (__ubuf__ uint32_t*)offset.GetPhyAddr();
    __ubuf__ float* betaFp32Local = (__ubuf__ float*)betaFp32.GetPhyAddr();
    __ubuf__ float* gammaFp32Local = (__ubuf__ float*)gammaFp32.GetPhyAddr();
    __ubuf__ float* meanFp32Local = (__ubuf__ float*)meanFp32.GetPhyAddr();
    __ubuf__ float* rstdFp32Local = (__ubuf__ float*)rstdFp32.GetPhyAddr();

    VFPrepareParamCache<T_GAMMA, T_RUNNING_MEAN>(gammaLocal, betaLocal, meanLocal, varLocal, offsetLocal,
                                                 gammaFp32Local, betaFp32Local, meanFp32Local, rstdFp32Local,
                                                 paramCacheElemLen, epsilon);

    betaQueue.template FreeTensor<T_GAMMA>(beta);
    gammaQueue.template FreeTensor<T_GAMMA>(gamma);
    meanQueue.template FreeTensor<T_RUNNING_MEAN>(mean);
    varQueue.template FreeTensor<T_RUNNING_MEAN>(var);
}

} // namespace BatchNormV3Ops
#endif // BATCH_NORM_V3_REGBASE_COMMON_H
