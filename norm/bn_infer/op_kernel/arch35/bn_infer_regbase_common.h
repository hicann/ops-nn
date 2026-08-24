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
 * \file bn_infer_regbase_common.h
 * \brief bn_infer regbase common helper
 */
#ifndef BN_INFER_REGBASE_COMMON_H
#define BN_INFER_REGBASE_COMMON_H

#include "../../norm_common/reduce_common_regbase.h"

namespace BNInferOps {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using AscendC::Reg::LoadAlign;
using AscendC::Reg::StoreAlign;

constexpr static uint32_t BN_INFER_ROW_TWO_OFFSET = 2;
constexpr static uint32_t BN_INFER_ROW_THREE_OFFSET = 3;
constexpr static uint32_t BN_INFER_ROW_FOUR_OFFSET = 4;
constexpr static uint32_t BN_INFER_ROW_ZERO = 0;
constexpr static uint32_t BN_INFER_ROW_ONE = 1;
constexpr static uint32_t BN_INFER_INDEX_ONE = 1;
constexpr static uint32_t BN_INFER_INDEX_TWO = 2;
constexpr static uint32_t BN_INFER_INDEX_FOUR = 4;
constexpr static uint32_t BN_INFER_INDEX_EIGHT = 8;
constexpr static uint32_t BN_INFER_INDEX_SIXTEEN = 16;
constexpr static float BN_INFER_POS_INF = 3.40282366920938E+38;

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
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
    params.remainderTailOffset0 = (BN_INFER_ROW_ZERO > params.remainderTailCount) ? params.validNumInXUb :
                                                                                    params.remainderOffset;
    params.remainderTailOffset1 = (BN_INFER_ROW_ONE > params.remainderTailCount) ?
                                      params.validNumInXUb :
                                      params.remainderOffset + params.aLength;
    params.remainderTailOffset2 = (BN_INFER_ROW_TWO_OFFSET > params.remainderTailCount) ?
                                      params.validNumInXUb :
                                      params.remainderOffset + BN_INFER_ROW_TWO_OFFSET * params.aLength;
    params.remainderTailOffset3 = (BN_INFER_ROW_THREE_OFFSET > params.remainderTailCount) ?
                                      params.validNumInXUb :
                                      params.remainderOffset + BN_INFER_ROW_THREE_OFFSET * params.aLength;
    return params;
}

template <typename T_SRC>
__aicore__ inline void LoadOneTensorForDtypeT(__local_mem__ T_SRC* input, RegTensor<float>& dst, MaskReg& preg,
                                              uint32_t offset)
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

template <typename T_SRC>
__aicore__ inline void GatherParamForDtypeT(__ubuf__ T_SRC* src, RegTensor<float>& dst,
                                            RegTensor<uint32_t>& paramOffset, MaskReg& preg, uint32_t calcLen)
{
    if constexpr (IsSameType<T_SRC, float>::value) {
        AscendC::MicroAPI::DataCopyGather(dst, (__local_mem__ float*)src, paramOffset, preg);
    } else {
        MaskReg pregSrc = AscendC::MicroAPI::UpdateMask<T_SRC>(calcLen);
        RegTensor<uint16_t> paramOffsetB16;
        RegTensor<T_SRC> srcB16;
        RegTensor<T_SRC> srcB16Unpack;
        AscendC::MicroAPI::Pack(paramOffsetB16, paramOffset);
        AscendC::MicroAPI::DataCopyGather(srcB16, ((__local_mem__ T_SRC*)src), paramOffsetB16, pregSrc);
        AscendC::MicroAPI::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
        AscendC::MicroAPI::Cast<float, T_SRC, castTraitB162B32>(dst, srcB16Unpack, preg);
    }
}

template <typename T_RUNNING_MEAN>
__aicore__ inline void GatherRunningParamForDtypeT(__ubuf__ T_RUNNING_MEAN* src, RegTensor<float>& dst,
                                                   RegTensor<uint32_t>& paramOffset, MaskReg& preg, uint32_t calcLen)
{
    if constexpr (IsSameType<T_RUNNING_MEAN, float>::value) {
        AscendC::MicroAPI::DataCopyGather(dst, (__local_mem__ float*)src, paramOffset, preg);
    } else {
        MaskReg pregSrc = AscendC::MicroAPI::UpdateMask<T_RUNNING_MEAN>(calcLen);
        RegTensor<uint16_t> paramOffsetB16;
        RegTensor<T_RUNNING_MEAN> srcB16;
        RegTensor<T_RUNNING_MEAN> srcB16Unpack;
        AscendC::MicroAPI::Pack(paramOffsetB16, paramOffset);
        AscendC::MicroAPI::DataCopyGather(srcB16, ((__local_mem__ T_RUNNING_MEAN*)src), paramOffsetB16, pregSrc);
        AscendC::MicroAPI::UnPack((RegTensor<uint32_t>&)srcB16Unpack, (RegTensor<uint16_t>&)srcB16);
        AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(dst, srcB16Unpack, preg);
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

__aicore__ inline uint32_t BNInferFindCofFactor(uint32_t n)
{
    if (n == 0) {
        return 0;
    }
    if ((n & (n - 1)) == 0) {
        return n;
    }
    uint32_t temp = n - 1;
    temp |= temp >> BN_INFER_INDEX_ONE;
    temp |= temp >> BN_INFER_INDEX_TWO;
    temp |= temp >> BN_INFER_INDEX_FOUR;
    temp |= temp >> BN_INFER_INDEX_EIGHT;
    temp |= temp >> BN_INFER_INDEX_SIXTEEN;
    return (temp + 1);
}

template <typename T_SRC>
__aicore__ inline void LoadTwoTensorForDtypeT(__local_mem__ T_SRC* src1, __local_mem__ T_SRC* src2,
                                              RegTensor<float>& dst1, RegTensor<float>& dst2, MaskReg& dst1Preg,
                                              MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
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
__aicore__ inline void LoadTwoTensorForDtypeTBrc(__local_mem__ T_SRC* src1, __local_mem__ T_SRC* src2,
                                                 RegTensor<float>& dst1, RegTensor<float>& dst2, MaskReg& dst1Preg,
                                                 MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
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
__aicore__ inline void CalculateRunningMeanVarVF(__local_mem__ float* batchMeanInUb, __local_mem__ float* batchVarInUb,
                                                 __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr,
                                                 __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr,
                                                 __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr,
                                                 __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr,
                                                 uint16_t currentANum, uint16_t aLoop, uint32_t vectorLen,
                                                 float unbiasedEstimationCoeff, float momentum, float momentumReverse);

template <typename T_RUNNING_MEAN>
__aicore__ inline void UpdateRunningMeanVarCommon(
    LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
    TQue<QuePosition::VECIN, 1>& runningMeanInQueue, TQue<QuePosition::VECIN, 1>& runningVarInQueue,
    TQue<QuePosition::VECOUT, 1>& runningMeanOutQueue, TQue<QuePosition::VECOUT, 1>& runningVarOutQueue,
    GlobalTensor<T_RUNNING_MEAN>& runningMeanGm, GlobalTensor<T_RUNNING_MEAN>& runningVarGm,
    GlobalTensor<T_RUNNING_MEAN>& runningMeanOutGm, GlobalTensor<T_RUNNING_MEAN>& runningVarOutGm, int64_t gmOffset,
    uint32_t currentA, uint16_t aLoop, uint32_t vectorLen, float unbiasedEstimationCoeff, float momentum,
    float momentumReverse)
{
    LocalTensor<T_RUNNING_MEAN> runningMeanInTensor = runningMeanInQueue.AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningVarInTensor = runningVarInQueue.AllocTensor<T_RUNNING_MEAN>();
    CopyInRunningMeanVarPad(runningMeanInTensor, runningVarInTensor, runningMeanGm, runningVarGm, gmOffset, currentA);
    runningMeanInQueue.EnQue(runningMeanInTensor);
    runningVarInQueue.EnQue(runningVarInTensor);
    runningMeanInTensor = runningMeanInQueue.template DeQue<T_RUNNING_MEAN>();
    runningVarInTensor = runningVarInQueue.template DeQue<T_RUNNING_MEAN>();

    LocalTensor<T_RUNNING_MEAN> runningMeanOutTensor = runningMeanOutQueue.AllocTensor<T_RUNNING_MEAN>();
    LocalTensor<T_RUNNING_MEAN> runningVarOutTensor = runningVarOutQueue.AllocTensor<T_RUNNING_MEAN>();
    __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningMeanInTensor.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarInTensor.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr = (__local_mem__ T_RUNNING_MEAN*)
                                                             runningMeanOutTensor.GetPhyAddr();
    __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarOutTensor.GetPhyAddr();
    __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
    __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
    CalculateRunningMeanVarVF<T_RUNNING_MEAN>(batchMeanTensorAddr, batchRstdTensorAddr, runningMeanInUbAddr,
                                              runningVarInUbAddr, runningMeanOutUbAddr, runningVarOutUbAddr,
                                              static_cast<uint16_t>(currentA), aLoop, vectorLen,
                                              unbiasedEstimationCoeff, momentum, momentumReverse);

    runningMeanInQueue.FreeTensor(runningMeanInTensor);
    runningVarInQueue.FreeTensor(runningVarInTensor);
    runningMeanOutQueue.EnQue(runningMeanOutTensor);
    runningVarOutQueue.EnQue(runningVarOutTensor);
    runningMeanOutTensor = runningMeanOutQueue.template DeQue<T_RUNNING_MEAN>();
    runningVarOutTensor = runningVarOutQueue.template DeQue<T_RUNNING_MEAN>();
    CopyOutRunningMeanVarPad(runningMeanOutTensor, runningVarOutTensor, runningMeanOutGm, runningVarOutGm, gmOffset,
                             currentA);
    runningMeanOutQueue.FreeTensor(runningMeanOutTensor);
    runningVarOutQueue.FreeTensor(runningVarOutTensor);
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

#include "bn_infer_regbase_common_part1.h"
#include "bn_infer_regbase_common_part2.h"
} // namespace BNInferOps
#endif // BN_INFER_REGBASE_COMMON_H
