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
 * \file batch_norm_ra_full_reduce.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_RA_FULL_REDUCE_H
#define NORM_BATCH_NORM_RA_FULL_REDUCE_H

#include "batch_norm_base.h"

namespace BatchNormOps
{
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

template <typename T1, typename T2>
class BatchNormRAFullReduce
{
public:
    __aicore__ inline BatchNormRAFullReduce(const BatchNormRAFullReduceTilingData* tilingData, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->r1 = tilingData->r1;
        this->a = tilingData->a;
        this->r1Quotient = tilingData->binaryAddQuotient;
        this->binaryAddK = tilingData->binaryAddK;
        this->binaryAddLast = tilingData->binaryAddLast;

        this->blockNum = tilingData->blockNum;
        this->aBlockFactor = tilingData->aBlockFactor;
        this->aFactor = tilingData->aFactor;
        this->powerOfTwoForR = tilingData->powerOfTwoForR;

        this->epsilon = tilingData->epsilon;
        this->momentum = tilingData->momentum;

        this->useRunningMeanVar = tilingData->useRunningMeanVar > 0 ? true : false;

        if (this->useRunningMeanVar) {
            float one = 1.0;
            this->oneSubMomentum = one - this->momentum;
        }

        this->besselCorrectionFactor = this->r1 == 1
                                           ? AscendC::NumericLimits<float>::QuietNaN()
                                           : (static_cast<float>(this->r1) / static_cast<float>(this->r1 - 1));
        this->nFactor = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        this->nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(this->r1);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_rstd)
    {
        auto blockIdx = GetBlockIdx();

        this->singleA = (blockIdx == this->blockNum - 1) ? (this->a - this->aBlockFactor * (this->blockNum - 1))
                                                         : this->aBlockFactor;
        int64_t aGmOffset = this->aBlockFactor * blockIdx;
        xGm.SetGlobalBuffer((__gm__ T1*)x + aGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T2*)beta + aGmOffset);
        gammaGm.SetGlobalBuffer((__gm__ T2*)gamma + aGmOffset);
        if (this->useRunningMeanVar) {
            runningMeanGm.SetGlobalBuffer((__gm__ T2*)mean + aGmOffset);
            runningVarGm.SetGlobalBuffer((__gm__ T2*)var + aGmOffset);
        }

        yGm.SetGlobalBuffer((__gm__ T1*)y + aGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean + aGmOffset);
        batchRstdGm.SetGlobalBuffer((__gm__ float*)batch_rstd + aGmOffset);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T2*)mean_out + aGmOffset);
        runningVarOutGm.SetGlobalBuffer((__gm__ T2*)var_out + aGmOffset);

        int64_t aFactorAlign = (((this->aFactor * sizeof(T1) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T1);
        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            pipe->InitBuffer(xQueue, DOUBLE_BUFFER, this->r1 * aFactorAlign * sizeof(T1));
            pipe->InitBuffer(castBuf, (this->r1 + 1) * aFactorAlign * sizeof(float));
        } else {
            pipe->InitBuffer(xQueue, DOUBLE_BUFFER, (this->r1 + 1) * aFactorAlign * sizeof(T1));
        }
        pipe->InitBuffer(yQueue, DOUBLE_BUFFER, this->r1 * aFactorAlign * sizeof(T1));

        int64_t aFactorAlignF32 =
            (((this->aFactor * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / FLOAT_BYTES;

        pipe->InitBuffer(betaQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(T2));
        pipe->InitBuffer(gammaQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(T2));

        pipe->InitBuffer(batchMeanQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe->InitBuffer(batchRstdQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        if (this->useRunningMeanVar) {
            pipe->InitBuffer(runningMeanInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
            pipe->InitBuffer(runningVarInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        }
        pipe->InitBuffer(runningMeanOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe->InitBuffer(runningVarOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t quotient = (this->singleA + this->aFactor - 1) / this->aFactor;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
            int64_t aOffset = ubLoopIdx * this->aFactor;
            int64_t currentA =
                (ubLoopIdx == (quotient - 1)) ? (this->singleA - (quotient - 1) * this->aFactor) : this->aFactor;
            currentANumAlign = (((currentA * sizeof(T1) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T1);
            ProcessUB(aOffset, currentA);
        }
    }

private:
    __aicore__ inline void ProcessUB(int64_t aOffset, int64_t currentANum)
    {
        CopyInX(aOffset, currentANum);
        LocalTensor<T1> xInUb = xQueue.template DeQue<T1>();
        LocalTensor<T1> yInUb = yQueue.AllocTensor<T1>();
        LocalTensor<float> batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchRstdOutUb = batchRstdQueue.AllocTensor<float>();
        __local_mem__ T1* xInUbAddr = (__local_mem__ T1*)xInUb.GetPhyAddr();
        __local_mem__ float* xFp32InUbAddr = (__local_mem__ float*)xInUbAddr;

        __local_mem__ T1* yInUbAddr = (__local_mem__ T1*)yInUb.GetPhyAddr();
        __local_mem__ float* batchMeanOutUbAddr = (__local_mem__ float*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ float* batchRstdOutUbAddr = (__local_mem__ float*)batchRstdOutUb.GetPhyAddr();

        if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
            LocalTensor<float> castInUb = castBuf.Get<float>();
            xFp32InUbAddr = (__local_mem__ float*)castInUb.GetPhyAddr();
            CastToFp32(xInUbAddr, xFp32InUbAddr, currentANum);
            CalculateMean(xFp32InUbAddr, yInUbAddr, batchMeanOutUbAddr, currentANum);
            CalculateVar(xFp32InUbAddr, yInUbAddr, batchMeanOutUbAddr, batchRstdOutUbAddr, currentANum);
        } else {
            CalculateMean(xFp32InUbAddr, yInUbAddr, batchMeanOutUbAddr, currentANum);
            CalculateVar(xFp32InUbAddr, yInUbAddr, batchMeanOutUbAddr, batchRstdOutUbAddr, currentANum);
        }
        CopyInGammaBeta(aOffset, currentANum);
        if (this->useRunningMeanVar) {
            CopyInRunningMeanVar(aOffset, currentANum);
        }

        LocalTensor<T2> betaInUb = betaQueue.template DeQue<T2>();
        LocalTensor<T2> gammaInUb = gammaQueue.template DeQue<T2>();
        LocalTensor<T2> runningMeanOutUb = runningMeanOutQueue.AllocTensor<T2>();
        LocalTensor<T2> runningVarOutUb = runningVarOutQueue.AllocTensor<T2>();
        __local_mem__ T2* betaInUbAddr = (__local_mem__ T2*)betaInUb.GetPhyAddr();
        __local_mem__ T2* gammaInUbAddr = (__local_mem__ T2*)gammaInUb.GetPhyAddr();
        __local_mem__ T2* runningMeanOutUbAddr = (__local_mem__ T2*)runningMeanOutUb.GetPhyAddr();
        __local_mem__ T2* runningVarOutUbAddr = (__local_mem__ T2*)runningVarOutUb.GetPhyAddr();

        LocalTensor<T2> runningMeanInUb;
        LocalTensor<T2> runningVarInUb;
        __local_mem__ T2* runningMeanInUbAddr = nullptr;
        __local_mem__ T2* runningVarInUbAddr = nullptr;

        if (this->useRunningMeanVar) {
            runningMeanInUb = runningMeanInQueue.template DeQue<T2>();
            runningVarInUb = runningVarInQueue.template DeQue<T2>();
            runningMeanInUbAddr = (__local_mem__ T2*)runningMeanInUb.GetPhyAddr();
            runningVarInUbAddr = (__local_mem__ T2*)runningVarInUb.GetPhyAddr();
        }

        CalculateRuningMeanVarVF(batchMeanOutUbAddr, batchRstdOutUbAddr, runningMeanInUbAddr, runningVarInUbAddr,
                                 runningMeanOutUbAddr, runningVarOutUbAddr, currentANum);
        QueueOperateAfterMeanVar(runningMeanInUb, runningVarInUb, batchMeanOutUb, batchRstdOutUb, runningMeanOutUb,
                                 runningVarOutUb);

        CopyOutRunningMeanVar(aOffset, currentANum);
        CalculateNormalizeVF(xFp32InUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanOutUbAddr,
                             batchRstdOutUbAddr, currentANum);
        QueueOperateAfterNormalize(xInUb, betaInUb, gammaInUb, yInUb);
        CopyOutY(aOffset, currentANum);
        CopyOutBatchMeanRstd(aOffset, currentANum);
    }

    __aicore__ inline void QueueOperateAfterMeanVar(LocalTensor<T2>& runningMeanInUb, LocalTensor<T2>& runningVarInUb,
                                                    LocalTensor<float>& batchMeanOutUb,
                                                    LocalTensor<float>& batchRstdOutUb,
                                                    LocalTensor<T2>& runningMeanOutUb, LocalTensor<T2>& runningVarOutUb)
    {
        if (this->useRunningMeanVar) {
            runningMeanInQueue.FreeTensor(runningMeanInUb);
            runningVarInQueue.FreeTensor(runningVarInUb);
        }
        batchMeanQueue.EnQue(batchMeanOutUb);
        batchRstdQueue.EnQue(batchRstdOutUb);
        runningMeanOutQueue.EnQue(runningMeanOutUb);
        runningVarOutQueue.EnQue(runningVarOutUb);
    }

    __aicore__ inline void QueueOperateAfterNormalize(LocalTensor<T1>& xInUb, LocalTensor<T2>& betaInUb,
                                                      LocalTensor<T2>& gammaInUb, LocalTensor<T1>& yInUb)
    {
        xQueue.FreeTensor(xInUb);
        betaQueue.FreeTensor(betaInUb);
        gammaQueue.FreeTensor(gammaInUb);
        yQueue.EnQue(yInUb);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T1> xInUb = xQueue.AllocTensor<T1>();
        if (currentANum * sizeof(T1) <= NDDMA_THRESHOLD) {
            T1 constValue = 0;
            static constexpr MultiCopyConfig config = {false};

            MultiCopyLoopInfo<NDDMA_DIM_NUM> loopInfo;
            loopInfo.loopSize[0] = currentANum;
            loopInfo.loopSrcStride[0] = 1;
            loopInfo.loopDstStride[0] = 1;
            loopInfo.loopLpSize[0] = 0;
            loopInfo.loopRpSize[0] = 0;

            loopInfo.loopSize[1] = this->r1;
            loopInfo.loopSrcStride[1] = this->a;
            loopInfo.loopDstStride[1] = currentANumAlign;
            loopInfo.loopLpSize[1] = 0;
            loopInfo.loopRpSize[1] = 0;
            MultiCopyParams<T1, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
            DataCopy<T1, NDDMA_DIM_NUM, config>(xInUb, xGm[offset], paramsMain);
        } else {
            DataCopyPadExtParams<T1> dataCopyPadExtParams;
            dataCopyPadExtParams.isPad = (currentANum != currentANumAlign);
            dataCopyPadExtParams.leftPadding = 0;
            // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
            dataCopyPadExtParams.rightPadding = 0;
            dataCopyPadExtParams.paddingValue = 0;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = this->r1;
            copyInParams.blockLen = currentANum * sizeof(T1);
            copyInParams.srcStride = (this->a - currentANum) * sizeof(T1);
            copyInParams.dstStride = 0;
            DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
        }
        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void CopyInGammaBeta(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T2> betaInUb = betaQueue.AllocTensor<T2>();
        LocalTensor<T2> gammaInUb = gammaQueue.AllocTensor<T2>();
        DataCopyPadExtParams<T2> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(betaInUb, betaGm[offset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(gammaInUb, gammaGm[offset], copyInParams, dataCopyPadExtParams);
        betaQueue.EnQue(betaInUb);
        gammaQueue.EnQue(gammaInUb);
    }

    __aicore__ inline void CopyInRunningMeanVar(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T2> runningMeanInUb = runningMeanInQueue.AllocTensor<T2>();
        LocalTensor<T2> runningVarInUb = runningVarInQueue.AllocTensor<T2>();
        DataCopyPadExtParams<T2> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanInUb, runningMeanGm[offset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(runningVarInUb, runningVarGm[offset], copyInParams, dataCopyPadExtParams);
        runningMeanInQueue.EnQue(runningMeanInUb);
        runningVarInQueue.EnQue(runningVarInUb);
    }

    __aicore__ inline void CopyOutY(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T1> yOutUb = yQueue.template DeQue<T1>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = this->r1;
        copyInParams.blockLen = currentANum * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->a - currentANum) * sizeof(T1);
        DataCopyPad(yGm[offset], yOutUb, copyInParams);
        yQueue.FreeTensor(yOutUb);
    }

    __aicore__ inline void CopyOutBatchMeanRstd(int64_t offset, int64_t currentANum)
    {
        LocalTensor<float> batchMeanInUb = batchMeanQueue.template DeQue<float>();
        LocalTensor<float> batchRstdInUb = batchRstdQueue.template DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(batchMeanGm[offset], batchMeanInUb, copyInParams);
        DataCopyPad(batchRstdGm[offset], batchRstdInUb, copyInParams);
        batchMeanQueue.FreeTensor(batchMeanInUb);
        batchRstdQueue.FreeTensor(batchRstdInUb);
    }

    __aicore__ inline void CopyOutRunningMeanVar(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T2> runningMeanOutUb = runningMeanOutQueue.template DeQue<T2>();
        LocalTensor<T2> runningVarOutUb = runningVarOutQueue.template DeQue<T2>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T2);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanOutGm[offset], runningMeanOutUb, copyInParams);
        DataCopyPad(runningVarOutGm[offset], runningVarOutUb, copyInParams);
        runningMeanOutQueue.FreeTensor(runningMeanOutUb);
        runningVarOutQueue.FreeTensor(runningVarOutUb);
    }

    __aicore__ inline void TwoRowAddForMeanWithTail(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
                                                    uint32_t offset1, uint32_t offset2, uint32_t offset3,
                                                    uint32_t offset4, RegTensor<float>& rem, RegTensor<float>& nextRow,
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

    __aicore__ inline void TwoRowAddForMean(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
                                            uint32_t offset1, uint32_t offset2, RegTensor<float>& nextRow, float n)
    {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
        DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
        Muls(dst, dst, n, preg);
        Muls(nextRow, nextRow, n, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForVarWithTail(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
                                                   uint32_t offset1, uint32_t offset2, uint32_t offset3,
                                                   uint32_t offset4, RegTensor<float>& mean, RegTensor<float>& rem,
                                                   RegTensor<float>& nextRow, RegTensor<float>& remNextRow, float n)
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

    __aicore__ inline void TwoRowAddForVar(RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg,
                                           uint32_t offset1, uint32_t offset2, RegTensor<float>& mean,
                                           RegTensor<float>& nextRow, float n)
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

    __aicore__ inline void CastToFp32(__local_mem__ T1* xInUb, __local_mem__ float* castInUb, int64_t currentA)
    {
        int64_t calculateNum = this->r1 * currentANumAlign;
        uint16_t loopCount = ops::CeilDiv(calculateNum, static_cast<int64_t>(VL_FP32));

        __VEC_SCOPE__
        {
            RegTensor<float> tmp;
            MaskReg pregLoop;
            uint32_t sreg0 = calculateNum;
            for (uint16_t k = 0; k < loopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                LoadTensorForDtypeT<T1>(tmp, xInUb, pregLoop, k * VL_FP32);
                DataCopy(((__local_mem__ float*)castInUb + k * VL_FP32), tmp, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateMean(__local_mem__ float* xInUb, __local_mem__ T1* yInUb,
                                         __local_mem__ float* batchMeanInUbAddr, int64_t currentA)
    {
        if (this->r1 <= SCALE_COEF_TWO) {
            CalculateMeanRLessThan2(xInUb, batchMeanInUbAddr, currentA);
        } else if (this->r1 <= SCALE_COEF_FOUR) {
            CalculateMeanRLessThan4(xInUb, batchMeanInUbAddr, currentA);
        } else if (this->r1 <= SCALE_COEF_EIGHT) {
            CalculateMeanRLessThan8(xInUb, batchMeanInUbAddr, currentA);
        } else {
            CalculateMeanRMoreThan8(xInUb, yInUb, batchMeanInUbAddr, currentA);
        }
    }

    __aicore__ inline void CalculateMeanRLessThan2(__local_mem__ float* xInUb, __local_mem__ float* batchMeanInUbAddr,
                                                   int64_t currentA)
    {
        uint32_t rStride = (((currentA * sizeof(T1) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T1);
        uint16_t rLoopCount = this->r1;
        float n = static_cast<float>(1) / static_cast<float>(this->r1);
        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));

        __VEC_SCOPE__
        {
            RegTensor<float> xld;
            RegTensor<float> xmuls;
            RegTensor<float> sum;

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(sum, 0.0, pregLoop);
                for (uint16_t i = 0; i < rLoopCount; i++) {
                    DataCopy(xld, ((__local_mem__ float*)xInUb + i * rStride + k * VL_FP32));
                    Muls(xmuls, xld, n, pregLoop);
                    Add(sum, sum, xmuls, pregLoop);
                }
                DataCopy(((__local_mem__ float*)batchMeanInUbAddr + k * VL_FP32), sum, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateMeanRLessThan4(__local_mem__ float* xInUb, __local_mem__ float* batchMeanInUbAddr,
                                                   int64_t currentA)
    {
        uint32_t remainderOffset = SCALE_COEF_TWO * currentANumAlign;
        uint32_t aLength = currentANumAlign;
        uint32_t validNumInXUb = this->r1 * currentANumAlign;

        float n = this->nFactor;
        float nCorrection = this->nCorrectionFactor;

        uint16_t remainderTailCount = this->r1 - SCALE_COEF_TWO;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;

        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForMeanWithTail(x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                         aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow,
                                         remNextRow, n);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateMeanRLessThan8(__local_mem__ float* xInUb, __local_mem__ float* batchMeanInUbAddr,
                                                   int64_t currentA)
    {
        uint32_t remainderOffset = SCALE_COEF_FOUR * currentANumAlign;
        uint32_t aLength = currentANumAlign;
        uint32_t validNumInXUb = this->r1 * currentANumAlign;

        float n = this->nFactor;
        float nCorrection = this->nCorrectionFactor;

        uint16_t remainderTailCount = this->r1 - SCALE_COEF_FOUR;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_THREE_OFFSET * aLength;

        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForMeanWithTail(x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                         aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow,
                                         remNextRow, n);
                TwoRowAddForMeanWithTail(x2, xInUb, pregLoop, ROW_TWO_OFFSET * aLength + aLoopOffset,
                                         remainderTailOffset2 + aLoopOffset, ROW_THREE_OFFSET * aLength + aLoopOffset,
                                         remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow, n);
                Add(x1, x1, x2, pregLoop);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateMeanRMoreThan8(__local_mem__ float* xInUb, __local_mem__ T1* yInUb,
                                                   __local_mem__ float* batchMeanInUbAddr, int64_t currentA)
    {
        uint16_t remainderLoopCount = (this->r1 - this->r1Quotient + SCALE_COEF_EIGHT - 1) / SCALE_COEF_EIGHT;
        uint16_t quotientLoopCount = (this->r1Quotient / SCALE_COEF_EIGHT) - remainderLoopCount;
        uint32_t remainderOffset = this->r1Quotient * currentANumAlign;

        uint32_t baseLineOffset = SCALE_COEF_EIGHT * currentANumAlign;
        uint32_t aLength = currentANumAlign;

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddInnerLoop = this->r1Quotient / SCALE_COEF_EIGHT;
        uint16_t binaryAddLastLoop = this->binaryAddLast;

        uint32_t validNumInXUb = this->r1 * currentANumAlign;

        float n = this->nFactor;
        float nCorrection = this->nCorrectionFactor;

        uint16_t remainderTailCount = (this->r1 - this->r1Quotient) - (remainderLoopCount - 1) * SCALE_COEF_EIGHT;
        uint32_t quotientTailOffset = (remainderLoopCount - 1) * baseLineOffset;
        uint32_t remainderTailOffset = quotientTailOffset + remainderOffset;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderTailOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderTailOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_THREE_OFFSET * aLength;
        uint32_t remainderTailOffset4 =
            (ROW_FOUR > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FOUR_OFFSET * aLength;
        uint32_t remainderTailOffset5 =
            (ROW_FIVE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FIVE_OFFSET * aLength;
        uint32_t remainderTailOffset6 =
            (ROW_SIX > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SIX_OFFSET * aLength;
        uint32_t remainderTailOffset7 =
            (ROW_SEVEN > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SEVEN_OFFSET * aLength;

        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 前半部分与后半部分中，都为8行的部分
                for (uint16_t i = 0; i < static_cast<uint16_t>(remainderLoopCount - 1); i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    TwoRowAddForMeanWithTail(x1, xInUb, pregLoop, quotOffset, remOffset, quotOffset + aLength,
                                             remOffset + aLength, rem, nextRow, remNextRow, n);
                    TwoRowAddForMeanWithTail(x2, xInUb, pregLoop, quotOffset + ROW_TWO_OFFSET * aLength,
                                             remOffset + ROW_TWO_OFFSET * aLength,
                                             quotOffset + ROW_THREE_OFFSET * aLength,
                                             remOffset + ROW_THREE_OFFSET * aLength, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForMeanWithTail(x3, xInUb, pregLoop, quotOffset + ROW_FOUR_OFFSET * aLength,
                                             remOffset + ROW_FOUR_OFFSET * aLength,
                                             quotOffset + ROW_FIVE_OFFSET * aLength,
                                             remOffset + ROW_FIVE_OFFSET * aLength, rem, nextRow, remNextRow, n);
                    TwoRowAddForMeanWithTail(x4, xInUb, pregLoop, quotOffset + ROW_SIX_OFFSET * aLength,
                                             remOffset + ROW_SIX_OFFSET * aLength,
                                             quotOffset + ROW_SEVEN_OFFSET * aLength,
                                             remOffset + ROW_SEVEN_OFFSET * aLength, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + i * aLength + aLoopOffset), x1, pregLoop);
                }
                // 前半部分为8行，后半部分可能不足8行
                {
                    TwoRowAddForMeanWithTail(x1, xInUb, pregLoop, quotientTailOffset + aLoopOffset,
                                             remainderTailOffset0 + aLoopOffset,
                                             quotientTailOffset + aLength + aLoopOffset,
                                             remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow, n);
                    TwoRowAddForMeanWithTail(x2, xInUb, pregLoop,
                                             quotientTailOffset + ROW_TWO_OFFSET * aLength + aLoopOffset,
                                             remainderTailOffset2 + aLoopOffset,
                                             quotientTailOffset + ROW_THREE_OFFSET * aLength + aLoopOffset,
                                             remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForMeanWithTail(x3, xInUb, pregLoop,
                                             quotientTailOffset + ROW_FOUR_OFFSET * aLength + aLoopOffset,
                                             remainderTailOffset4 + aLoopOffset,
                                             quotientTailOffset + ROW_FIVE_OFFSET * aLength + aLoopOffset,
                                             remainderTailOffset5 + aLoopOffset, rem, nextRow, remNextRow, n);
                    TwoRowAddForMeanWithTail(x4, xInUb, pregLoop,
                                             quotientTailOffset + ROW_SIX_OFFSET * aLength + aLoopOffset,
                                             remainderTailOffset6 + aLoopOffset,
                                             quotientTailOffset + ROW_SEVEN_OFFSET * aLength + aLoopOffset,
                                             remainderTailOffset7 + aLoopOffset, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + (remainderLoopCount - 1) * aLength + aLoopOffset), x1,
                             pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    TwoRowAddForMean(x1, xInUb, pregLoop, baseOffset, baseOffset + aLength, nextRow, n);
                    TwoRowAddForMean(x2, xInUb, pregLoop, baseOffset + ROW_TWO_OFFSET * aLength,
                                     baseOffset + ROW_THREE_OFFSET * aLength, nextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForMean(x3, xInUb, pregLoop, baseOffset + ROW_FOUR_OFFSET * aLength,
                                     baseOffset + ROW_FIVE_OFFSET * aLength, nextRow, n);
                    TwoRowAddForMean(x4, xInUb, pregLoop, baseOffset + ROW_SIX_OFFSET * aLength,
                                     baseOffset + ROW_SEVEN_OFFSET * aLength, nextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + (remainderLoopCount + i) * aLength + aLoopOffset), x1,
                             pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF((__local_mem__ float*)yInUb, aLength, aLoopOffset, binaryAddKLoop, binaryAddInnerLoop,
                            binaryAddLastLoop, pregLoop, x1, x2, x3, x4);
                DataCopy(x1, ((__local_mem__ float*)yInUb + aLoopOffset));
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVar(__local_mem__ float* xInUb, __local_mem__ T1* yInUb,
                                        __local_mem__ float* batchMeanInUbAddr, __local_mem__ float* batchRstdInUbAddr,
                                        int64_t currentA)
    {
        if (this->r1 <= SCALE_COEF_TWO) {
            CalculateVarRLessThan2(xInUb, batchMeanInUbAddr, batchRstdInUbAddr, currentA);
        } else if (this->r1 <= SCALE_COEF_FOUR) {
            CalculateVarRLessThan4(xInUb, batchMeanInUbAddr, batchRstdInUbAddr, currentA);
        } else if (this->r1 <= SCALE_COEF_EIGHT) {
            CalculateVarRLessThan8(xInUb, batchMeanInUbAddr, batchRstdInUbAddr, currentA);
        } else {
            CalculateVarRMoreThan8(xInUb, yInUb, batchMeanInUbAddr, batchRstdInUbAddr, currentA);
        }
    }

    __aicore__ inline void CalculateVarRLessThan2(__local_mem__ float* xInUb, __local_mem__ float* batchMeanInUbAddr,
                                                  __local_mem__ float* batchRstdOutUbAddr, int64_t currentA)
    {
        uint32_t rStride = (((currentA * sizeof(T1) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T1);
        uint16_t rLoopCount = this->r1;
        float n = static_cast<float>(1) / static_cast<float>(this->r1);
        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));

        __VEC_SCOPE__
        {
            RegTensor<float> xld;
            RegTensor<float> xsub;
            RegTensor<float> xpow;
            RegTensor<float> xmuls;
            RegTensor<float> sum;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(sum, 0.0, pregLoop);
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUbAddr + k * VL_FP32));
                for (uint16_t i = 0; i < rLoopCount; i++) {
                    DataCopy(xld, ((__local_mem__ float*)xInUb + i * rStride + k * VL_FP32));
                    Sub(xsub, xld, mean, pregLoop);
                    Mul(xpow, xsub, xsub, pregLoop);
                    Muls(xmuls, xpow, n, pregLoop);
                    Add(sum, sum, xmuls, pregLoop);
                }
                DataCopy(((__local_mem__ float*)batchRstdOutUbAddr + k * VL_FP32), sum, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVarRLessThan4(__local_mem__ float* xInUb, __local_mem__ float* batchMeanInUbAddr,
                                                  __local_mem__ float* batchRstdOutUbAddr, int64_t currentA)
    {
        uint32_t remainderOffset = SCALE_COEF_TWO * currentANumAlign;
        uint32_t aLength = currentANumAlign;
        uint32_t validNumInXUb = this->r1 * currentANumAlign;

        float n = this->nFactor;
        float nCorrection = this->nCorrectionFactor;

        uint16_t remainderTailCount = this->r1 - SCALE_COEF_TWO;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;

        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), mean, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForVarWithTail(x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                        aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, mean, rem, nextRow,
                                        remNextRow, n);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchRstdOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVarRLessThan8(__local_mem__ float* xInUb, __local_mem__ float* batchMeanInUbAddr,
                                                  __local_mem__ float* batchRstdOutUbAddr, int64_t currentA)
    {
        uint32_t remainderOffset = SCALE_COEF_FOUR * currentANumAlign;
        uint32_t aLength = currentANumAlign;
        uint32_t validNumInXUb = this->r1 * currentANumAlign;

        float n = this->nFactor;
        float nCorrection = this->nCorrectionFactor;

        uint16_t remainderTailCount = this->r1 - SCALE_COEF_FOUR;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_THREE_OFFSET * aLength;

        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), mean, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForVarWithTail(x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset,
                                        aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, mean, rem, nextRow,
                                        remNextRow, n);
                TwoRowAddForVarWithTail(x2, xInUb, pregLoop, ROW_TWO_OFFSET * aLength + aLoopOffset,
                                        remainderTailOffset2 + aLoopOffset, ROW_THREE_OFFSET * aLength + aLoopOffset,
                                        remainderTailOffset3 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                Add(x1, x1, x2, pregLoop);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchRstdOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVarRMoreThan8(__local_mem__ float* xInUb, __local_mem__ T1* yInUb,
                                                  __local_mem__ float* batchMeanInUbAddr,
                                                  __local_mem__ float* batchRstdOutUbAddr, int64_t currentA)
    {
        uint16_t remainderLoopCount = (this->r1 - this->r1Quotient + SCALE_COEF_EIGHT - 1) / SCALE_COEF_EIGHT;
        uint16_t quotientLoopCount = (this->r1Quotient / SCALE_COEF_EIGHT) - remainderLoopCount;
        uint32_t remainderOffset = this->r1Quotient * currentANumAlign;

        uint32_t baseLineOffset = SCALE_COEF_EIGHT * currentANumAlign;
        uint32_t aLength = currentANumAlign;

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddInnerLoop = this->r1Quotient / SCALE_COEF_EIGHT;
        uint16_t binaryAddLastLoop = this->binaryAddLast;

        uint32_t validNumInXUb = this->r1 * currentANumAlign;

        float n = this->nFactor;
        float nCorrection = this->nCorrectionFactor;

        uint16_t remainderTailCount = (this->r1 - this->r1Quotient) - (remainderLoopCount - 1) * SCALE_COEF_EIGHT;
        uint32_t quotientTailOffset = (remainderLoopCount - 1) * baseLineOffset;
        uint32_t remainderTailOffset = quotientTailOffset + remainderOffset;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderTailOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderTailOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_THREE_OFFSET * aLength;
        uint32_t remainderTailOffset4 =
            (ROW_FOUR > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FOUR_OFFSET * aLength;
        uint32_t remainderTailOffset5 =
            (ROW_FIVE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FIVE_OFFSET * aLength;
        uint32_t remainderTailOffset6 =
            (ROW_SIX > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SIX_OFFSET * aLength;
        uint32_t remainderTailOffset7 =
            (ROW_SEVEN > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SEVEN_OFFSET * aLength;

        uint16_t aLoopCount = ops::CeilDiv(currentA, static_cast<int64_t>(VL_FP32));
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), mean, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 前半部分与后半部分中，都为8行的部分
                for (uint16_t i = 0; i < static_cast<uint16_t>(remainderLoopCount - 1); i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    TwoRowAddForVarWithTail(x1, xInUb, pregLoop, quotOffset, remOffset, quotOffset + aLength,
                                            remOffset + aLength, mean, rem, nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(x2, xInUb, pregLoop, quotOffset + ROW_TWO_OFFSET * aLength,
                                            remOffset + ROW_TWO_OFFSET * aLength,
                                            quotOffset + ROW_THREE_OFFSET * aLength,
                                            remOffset + ROW_THREE_OFFSET * aLength, mean, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForVarWithTail(x3, xInUb, pregLoop, quotOffset + ROW_FOUR_OFFSET * aLength,
                                            remOffset + ROW_FOUR_OFFSET * aLength,
                                            quotOffset + ROW_FIVE_OFFSET * aLength,
                                            remOffset + ROW_FIVE_OFFSET * aLength, mean, rem, nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(x4, xInUb, pregLoop, quotOffset + ROW_SIX_OFFSET * aLength,
                                            remOffset + ROW_SIX_OFFSET * aLength,
                                            quotOffset + ROW_SEVEN_OFFSET * aLength,
                                            remOffset + ROW_SEVEN_OFFSET * aLength, mean, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + i * aLength + aLoopOffset), x1, pregLoop);
                }
                // 前半部分为8行，后半部分可能不足8行
                {
                    TwoRowAddForVarWithTail(x1, xInUb, pregLoop, quotientTailOffset + aLoopOffset,
                                            remainderTailOffset0 + aLoopOffset,
                                            quotientTailOffset + aLength + aLoopOffset,
                                            remainderTailOffset1 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(x2, xInUb, pregLoop,
                                            quotientTailOffset + ROW_TWO_OFFSET * aLength + aLoopOffset,
                                            remainderTailOffset2 + aLoopOffset,
                                            quotientTailOffset + ROW_THREE_OFFSET * aLength + aLoopOffset,
                                            remainderTailOffset3 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForVarWithTail(x3, xInUb, pregLoop,
                                            quotientTailOffset + ROW_FOUR_OFFSET * aLength + aLoopOffset,
                                            remainderTailOffset4 + aLoopOffset,
                                            quotientTailOffset + ROW_FIVE_OFFSET * aLength + aLoopOffset,
                                            remainderTailOffset5 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(x4, xInUb, pregLoop,
                                            quotientTailOffset + ROW_SIX_OFFSET * aLength + aLoopOffset,
                                            remainderTailOffset6 + aLoopOffset,
                                            quotientTailOffset + ROW_SEVEN_OFFSET * aLength + aLoopOffset,
                                            remainderTailOffset7 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + (remainderLoopCount - 1) * aLength + aLoopOffset), x1,
                             pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    TwoRowAddForVar(x1, xInUb, pregLoop, baseOffset, baseOffset + aLength, mean, nextRow, n);
                    TwoRowAddForVar(x2, xInUb, pregLoop, baseOffset + ROW_TWO_OFFSET * aLength,
                                    baseOffset + ROW_THREE_OFFSET * aLength, mean, nextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForVar(x3, xInUb, pregLoop, baseOffset + ROW_FOUR_OFFSET * aLength,
                                    baseOffset + ROW_FIVE_OFFSET * aLength, mean, nextRow, n);
                    TwoRowAddForVar(x4, xInUb, pregLoop, baseOffset + ROW_SIX_OFFSET * aLength,
                                    baseOffset + ROW_SEVEN_OFFSET * aLength, mean, nextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yInUb + (remainderLoopCount + i) * aLength + aLoopOffset), x1,
                             pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF((__local_mem__ float*)yInUb, aLength, aLoopOffset, binaryAddKLoop, binaryAddInnerLoop,
                            binaryAddLastLoop, pregLoop, x1, x2, x3, x4);
                DataCopy(x1, ((__local_mem__ float*)yInUb + aLoopOffset));
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchRstdOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void BinaryAddVF(__local_mem__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint32_t offset,
                                       uint16_t binaryAddKLoop, uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop,
                                       MaskReg& pregLoop, RegTensor<float>& x1, RegTensor<float>& x2,
                                       RegTensor<float>& x3, RegTensor<float>& x4)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                DataCopy(x2,
                         ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                Add(x1, x1, x2, pregLoop);
                DataCopy(x3, ((__local_mem__ float*)binaryAddTmpAddr +
                              (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride + offset));
                DataCopy(x4, ((__local_mem__ float*)binaryAddTmpAddr +
                              (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
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

    __aicore__ inline void CalculateRuningMeanVarVF(__local_mem__ float* batchMeanInUb,
                                                    __local_mem__ float* batchRstdInUb,
                                                    __local_mem__ T2* runningMeanInUbAddr,
                                                    __local_mem__ T2* runningVarInUbAddr,
                                                    __local_mem__ T2* runningMeanOutUbAddr,
                                                    __local_mem__ T2* runningVarOutUbAddr, uint16_t currentANum)
    {
        uint16_t aLoop = ops::CeilDiv(currentANum, VL_FP32);
        float besselCorrection = this->besselCorrectionFactor;
        float m = this->momentum;
        float oneSubM = this->oneSubMomentum;
        bool vfUseRunningMeanVar = this->useRunningMeanVar;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> var;

            RegTensor<float> sqrtVar;
            RegTensor<float> one;
            RegTensor<float> rsqrtVar;

            RegTensor<float> runningMean;
            RegTensor<float> saveMean;
            RegTensor<float> runningVar;
            RegTensor<float> saveVar;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();

            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> e;
            RegTensor<float> scalar1;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            RegTensor<float> t1;
            RegTensor<float> t2;
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
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(t1, float(1.5), pregLoop);
                Duplicate(s, float(1.0), pregLoop);
                // running var
                DataCopy(var, ((__local_mem__ float*)batchRstdInUb + k * VL_FP32));
                Muls(saveVar, var, besselCorrection, pregLoop);
                Muls(saveVar, saveVar, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningVar, runningVarInUbAddr, pregLoop, k * VL_FP32);
                    Muls(runningVar, runningVar, oneSubM, pregLoop);
                    Add(saveVar, saveVar, runningVar, pregLoop);
                }
                StoreTensorForDtypeT<T2>(runningVarOutUbAddr, saveVar, pregLoop, k * VL_FP32);

                // rstd
                Adds(var, var, epsilon, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(-0.5), pregLoop);
                Mul(t, t, y, pregLoop);                 // -0.5 * x * y
                Mula(t1, t, y, pregLoop);               // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);             // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);   // -1 * x
                Mula(s, t3, r, pregLoop);               // 1 + (-1) * x * r
                Muls(t4, rstd, float(-1.0), pregLoop);  // (-1) * y
                Mula(r, t4, rstd, pregLoop);            // r + (-1) * y * y
                Mula(s, var, r, pregLoop);              // s + x * t
                Mul(s, s, rstd, pregLoop);              // e * y
                Mula(rstd, s, scalar1, pregLoop);       // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(((__local_mem__ float*)batchRstdInUb + k * VL_FP32), rstd, pregLoop);

                // running mean
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUb + k * VL_FP32));
                Muls(saveMean, mean, m, pregLoop);
                if (vfUseRunningMeanVar) {
                    LoadTensorForDtypeT<T2>(runningMean, runningMeanInUbAddr, pregLoop, k * VL_FP32);
                    Muls(runningMean, runningMean, oneSubM, pregLoop);
                    Add(saveMean, saveMean, runningMean, pregLoop);
                }

                StoreTensorForDtypeT<T2>(runningMeanOutUbAddr, saveMean, pregLoop, k * VL_FP32);
            }
        }
    }

    __aicore__ inline void CalculateNormalizeVF(__local_mem__ float* xInUb, __local_mem__ T1* yInUb,
                                                __local_mem__ T2* betaInUb, __local_mem__ T2* gammaInUb,
                                                __local_mem__ float* batchMeanInUb, __local_mem__ float* batchRstdInUb,
                                                uint16_t currentANum)
    {
        uint16_t rLoopCount = this->r1;
        uint16_t aLoopCount = ops::CeilDiv(currentANum, VL_FP32);
        uint32_t rStride = currentANumAlign;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;

            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrtVar;

            RegTensor<float> beta;
            RegTensor<float> gamma;

            MaskReg pregLoop;
            uint32_t sreg2 = currentANum;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                LoadTwoTensorForDtypeT<T2>(beta, gamma, betaInUb, gammaInUb, pregLoop, pregLoop, k * VL_FP32,
                                           k * VL_FP32);
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUb + k * VL_FP32));
                DataCopy(rsqrtVar, ((__local_mem__ float*)batchRstdInUb + k * VL_FP32));
                for (uint16_t r = 0; r < rLoopCount; r++) {
                    DataCopy(x2, ((__local_mem__ float*)xInUb + r * rStride + k * VL_FP32));
                    Sub(x2, x2, mean, pregLoop);
                    Mul(y2, x2, rsqrtVar, pregLoop);
                    Mul(y2, y2, beta, pregLoop);
                    Add(y2, y2, gamma, pregLoop);
                    if constexpr (IsSameType<T1, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y2, pregLoop);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ half*)yInUb + r * rStride + k * VL_FP32), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y2, pregLoop);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ bfloat16_t*)yInUb + r * rStride + k * VL_FP32), xBf16, pregLoop);
                    } else {
                        DataCopy(((__local_mem__ float*)yInUb + r * rStride + k * VL_FP32), y2, pregLoop);
                    }
                }
            }
        }
    }

    /* global memory address */
    GlobalTensor<T1> xGm;
    GlobalTensor<T2> betaGm;
    GlobalTensor<T2> gammaGm;
    GlobalTensor<T2> runningMeanGm;
    GlobalTensor<T2> runningVarGm;

    GlobalTensor<T1> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchRstdGm;
    GlobalTensor<T2> runningMeanOutGm;
    GlobalTensor<T2> runningVarOutGm;

    /* variable */
    int64_t r1;
    int64_t powerOfTwoForR;
    int64_t aFactor;
    int64_t a;

    int64_t blockNum;
    int64_t aBlockFactor;
    int64_t singleA;
    int64_t currentANumAlign;

    int64_t r1Quotient;
    int64_t binaryAddK;
    int64_t binaryAddLast;

    static constexpr int64_t SCALE_COEF_TWO = 2;
    static constexpr int64_t SCALE_COEF_FOUR = 4;
    static constexpr int64_t SCALE_COEF_EIGHT = 8;
    static constexpr int64_t NDDMA_DIM_NUM = 2;

    static constexpr uint16_t ROW_ZERO = 0;
    static constexpr uint16_t ROW_ONE = 1;
    static constexpr uint16_t ROW_TWO = 2;
    static constexpr uint16_t ROW_THREE = 3;
    static constexpr uint16_t ROW_FOUR = 4;
    static constexpr uint16_t ROW_FIVE = 5;
    static constexpr uint16_t ROW_SIX = 6;
    static constexpr uint16_t ROW_SEVEN = 7;

    static constexpr uint32_t ROW_TWO_OFFSET = 2;
    static constexpr uint32_t ROW_THREE_OFFSET = 3;
    static constexpr uint32_t ROW_FOUR_OFFSET = 4;
    static constexpr uint32_t ROW_FIVE_OFFSET = 5;
    static constexpr uint32_t ROW_SIX_OFFSET = 6;
    static constexpr uint32_t ROW_SEVEN_OFFSET = 7;
    static constexpr float POS_INF = 3.40282366920938E+38;

    float epsilon = 1e-5;
    float momentum = 0.1;
    float besselCorrectionFactor;
    float oneSubMomentum;
    float nFactor;
    float nCorrectionFactor;

    bool useRunningMeanVar = true;

    /* ascendc variable */
    TPipe* pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> betaQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> gammaQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> runningMeanInQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> runningVarInQueue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> batchMeanQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> batchRstdQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> runningVarOutQueue;

    TBuf<TPosition::VECCALC> castBuf;
};
}  // namespace BatchNormOps

#endif // NORM_BATCH_NORM_RA_FULL_REDUCE_H