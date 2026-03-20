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
 * \file batch_norm_full_reduce.h
 * \brief
 */
#ifndef NORM_BATCH_NORM_FULL_REDUCE_H
#define NORM_BATCH_NORM_FULL_REDUCE_H

#include "batch_norm_base.h"

namespace BatchNormOps
{
template <typename T1, typename T2>
class BatchNormFullReduce
{
public:

    __aicore__ inline BatchNormFullReduce(const BatchNormFullReduceRegbaseTilingData* tilingData, TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->r1 = tilingData->r1;
        this->aFactor = tilingData->aFactor;
        this->a = tilingData->a;
        this->r0 = tilingData->r0;
        this->blockNum = tilingData->blockNum;
        this->aBlockFactor = tilingData->aBlockFactor;
        this->r1r0LoopCount = tilingData->r1r0LoopCount;
        this->epsilon = tilingData->epsilon;
        this->momentum = tilingData->momentum;
        this->useRunningMeanVar = tilingData->useRunningMeanVar > 0 ? true : false;

        if (useRunningMeanVar) {
            float one = 1.0;
            this->oneSubMomentum = one - this->momentum;
        }

        int64_t reduceNum = this->r1 * this->r0;
        this->besselCorrectionFactor = (static_cast<float>(reduceNum) / static_cast<float>(reduceNum - 1));

        this->powerOfTwoForR = tilingData->powerOfTwoForR;
        this->binaryAddQuotient = tilingData->binaryAddQuotient;
        this->binaryAddK = tilingData->binaryAddK;
        this->binaryAddLastNum = tilingData->binaryAddLastNum;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_rstd)
    {
        auto blockIdx = GetBlockIdx();

        this->r1r0Align =
            (((this->r1 * this->r0 * sizeof(T1) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T1);
        this->singleA = (blockIdx == this->blockNum - 1) ? (this->a - this->aBlockFactor * (this->blockNum - 1))
                                                         : this->aBlockFactor;

        int64_t aGmOffset = this->aBlockFactor * blockIdx;
        int64_t arGmOffset = aGmOffset * this->r0;
        xGm.SetGlobalBuffer((__gm__ T1*)x + arGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T2*)beta + aGmOffset);
        gammaGm.SetGlobalBuffer((__gm__ T2*)gamma + aGmOffset);
        if (useRunningMeanVar) {
            runningMeanGm.SetGlobalBuffer((__gm__ T2*)mean + aGmOffset);
            runningVarGm.SetGlobalBuffer((__gm__ T2*)var + aGmOffset);
        }

        yGm.SetGlobalBuffer((__gm__ T1*)y + arGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ T2*)batch_mean + aGmOffset);
        batchRstdGm.SetGlobalBuffer((__gm__ T2*)batch_rstd + aGmOffset);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T2*)mean_out + aGmOffset);
        runningVarOutGm.SetGlobalBuffer((__gm__ T2*)var_out + aGmOffset);

        int64_t aFactorAlign = (((this->aFactor * sizeof(T2) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T2);
        pipe->InitBuffer(betaQueue, DOUBLE_BUFFER, aFactorAlign * sizeof(T2));
        pipe->InitBuffer(gammaQueue, DOUBLE_BUFFER, aFactorAlign * sizeof(T2));
        int64_t aFactorAlignF32 =
            (((this->aFactor * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / FLOAT_BYTES;
        pipe->InitBuffer(batchMeanQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe->InitBuffer(batchRstdQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        if (useRunningMeanVar) {
            pipe->InitBuffer(runningMeanInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
            pipe->InitBuffer(runningVarInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        }
        pipe->InitBuffer(runningMeanOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe->InitBuffer(runningVarOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        int64_t xBufferSize = this->aFactor * this->r1r0Align;
        pipe->InitBuffer(xQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T1));
        pipe->InitBuffer(yQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T1));

        int64_t binaryAddBufSize =
            (((binaryAddQuotient / VL_FP32) * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        if (binaryAddBufSize > 0) {
            pipe->InitBuffer(binaryAddBuf, binaryAddBufSize);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t quotient = (this->singleA + this->aFactor - 1) / this->aFactor;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
            int64_t offset = ubLoopIdx * this->aFactor * this->r0;
            int64_t aOffset = ubLoopIdx * this->aFactor;
            int64_t currentA =
                (ubLoopIdx == (quotient - 1)) ? (this->singleA - (quotient - 1) * this->aFactor) : this->aFactor;
            ProcessUB(offset, aOffset, currentA);
        }
    }

private:
    __aicore__ inline void ProcessUB(int64_t raOffset, int64_t aOffset, int64_t currentANum)
    {
        CopyInX(raOffset, currentANum);

        LocalTensor<T1> xInUb = xQueue.template DeQue<T1>();
        __local_mem__ T1* xInUbAddr = (__local_mem__ T1*)xInUb.GetPhyAddr();
        LocalTensor<float> batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchRstdOutUb = batchRstdQueue.AllocTensor<float>();
        __local_mem__ float* batchMeanInUbAddr = (__local_mem__ float*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ float* batchRstdInUbAddr = (__local_mem__ float*)batchRstdOutUb.GetPhyAddr();
        if (this->r1 * this->r0 <= VL_FP32) {
            CalculateMeanVarRLessThanVL64VF(xInUbAddr, batchMeanInUbAddr, batchRstdInUbAddr, currentANum);
        } else {
            CalculateMeanVarVF(xInUbAddr, batchMeanInUbAddr, batchRstdInUbAddr, currentANum);
        }

        CopyInGammaBeta(aOffset, currentANum);
        if (useRunningMeanVar) {
            CopyInRunningMeanVar(aOffset, currentANum);
        }

        LocalTensor<T2> betaInUb = betaQueue.template DeQue<T2>();
        LocalTensor<T2> gammaInUb = gammaQueue.template DeQue<T2>();
        __local_mem__ T2* betaInUbAddr = (__local_mem__ T2*)betaInUb.GetPhyAddr();
        __local_mem__ T2* gammaInUbAddr = (__local_mem__ T2*)gammaInUb.GetPhyAddr();

        LocalTensor<T2> runningMeanInUb;
        LocalTensor<T2> runningVarInUb;
        __local_mem__ T2* runningMeanInUbAddr = nullptr;
        __local_mem__ T2* runningVarInUbAddr = nullptr;

        if (useRunningMeanVar) {
            runningMeanInUb = runningMeanInQueue.template DeQue<T2>();
            runningVarInUb = runningVarInQueue.template DeQue<T2>();
            runningMeanInUbAddr = (__local_mem__ T2*)runningMeanInUb.GetPhyAddr();
            runningVarInUbAddr = (__local_mem__ T2*)runningVarInUb.GetPhyAddr();
        }

        LocalTensor<T1> yInUb = yQueue.AllocTensor<T1>();
        LocalTensor<T2> runningMeanOutUb = runningMeanOutQueue.AllocTensor<T2>();
        LocalTensor<T2> runningVarOutUb = runningVarOutQueue.AllocTensor<T2>();
        __local_mem__ T1* yInUbAddr = (__local_mem__ T1*)yInUb.GetPhyAddr();
        __local_mem__ T2* runningMeanOutUbAddr = (__local_mem__ T2*)runningMeanOutUb.GetPhyAddr();
        __local_mem__ T2* runningVarOutUbAddr = (__local_mem__ T2*)runningVarOutUb.GetPhyAddr();
        CalculateRuningMeanVarVF(batchMeanInUbAddr, batchRstdInUbAddr, runningMeanInUbAddr, runningVarInUbAddr,
                                 runningMeanOutUbAddr, runningVarOutUbAddr, currentANum);

        if (useRunningMeanVar) {
            runningMeanInQueue.FreeTensor(runningMeanInUb);
            runningVarInQueue.FreeTensor(runningVarInUb);
        }
        batchMeanQueue.EnQue(batchMeanOutUb);
        batchRstdQueue.EnQue(batchRstdOutUb);
        runningMeanOutQueue.EnQue(runningMeanOutUb);
        runningVarOutQueue.EnQue(runningVarOutUb);

        CopyOutBatchMeanRstd(aOffset, currentANum);
        CopyOutRunningMeanVar(aOffset, currentANum);

        CalculateNormalizeVF(xInUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanInUbAddr, batchRstdInUbAddr,
                             currentANum);

        xQueue.FreeTensor(xInUb);
        betaQueue.FreeTensor(betaInUb);
        gammaQueue.FreeTensor(gammaInUb);
        yQueue.EnQue(yInUb);

        CopyOutY(raOffset, currentANum);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T1> xInUb = xQueue.AllocTensor<T1>();
        if (this->r0 * sizeof(T1) <= NDDMA_THRESHOLD) {
            T1 constValue = 0;
            static constexpr MultiCopyConfig config = {false};

            MultiCopyLoopInfo<NDDMA_DIM_NUM> loopInfo;
            loopInfo.loopSize[0] = this->r0;
            loopInfo.loopSrcStride[0] = 1;
            loopInfo.loopDstStride[0] = 1;
            loopInfo.loopLpSize[0] = 0;
            loopInfo.loopRpSize[0] = 0;

            loopInfo.loopSize[NDDMA_SECOND_DIM] = currentANum;
            loopInfo.loopSrcStride[NDDMA_SECOND_DIM] = this->r0;
            loopInfo.loopDstStride[NDDMA_SECOND_DIM] = this->r1r0Align;
            loopInfo.loopLpSize[NDDMA_SECOND_DIM] = 0;
            loopInfo.loopRpSize[NDDMA_SECOND_DIM] = 0;

            loopInfo.loopSize[NDDMA_THIRD_DIM] = this->r1;
            loopInfo.loopSrcStride[NDDMA_THIRD_DIM] = this->a * this->r0;
            loopInfo.loopDstStride[NDDMA_THIRD_DIM] = this->r0;
            loopInfo.loopLpSize[NDDMA_THIRD_DIM] = 0;
            loopInfo.loopRpSize[NDDMA_THIRD_DIM] = 0;
            MultiCopyParams<T1, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
            DataCopy<T1, NDDMA_DIM_NUM, config>(xInUb, xGm[offset], paramsMain);
        } else {
            uint64_t r1LoopSrcStride = this->r0 * sizeof(T1);
            uint64_t r1LoopDstStride = this->r1r0Align * sizeof(T1);
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = currentANum;
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;
            loopParams.loop1SrcStride = r1LoopSrcStride;
            loopParams.loop1DstStride = r1LoopDstStride;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPadExtParams<T1> dataCopyPadExtParams;
            dataCopyPadExtParams.isPad = false;
            dataCopyPadExtParams.leftPadding = 0;
            dataCopyPadExtParams.rightPadding = 0;
            dataCopyPadExtParams.paddingValue = 0;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = this->r1;
            copyInParams.blockLen = this->r0 * sizeof(T1);
            copyInParams.srcStride = (this->a - 1) * this->r0 * sizeof(T1);
            copyInParams.dstStride = 0;
            DataCopyPad<T1, PaddingMode::Compact>(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
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

        uint64_t r1LoopSrcStride = this->r1r0Align * sizeof(T1);
        uint64_t r1LoopDstStride = this->r0 * sizeof(T1);
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = currentANum;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = r1LoopSrcStride;
        loopParams.loop1DstStride = r1LoopDstStride;
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = this->r1;
        copyInParams.blockLen = this->r0 * sizeof(T1);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->a - 1) * this->r0 * sizeof(T1);
        DataCopyPad<T1, PaddingMode::Compact>(yGm[offset], yOutUb, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
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

    __aicore__ inline void CalculateMeanVarRLessThanVL64VF(__local_mem__ T1* xInUb, __local_mem__ float* batchMeanInUb,
                                                           __local_mem__ float* batchRstdInUb, uint16_t currentANum)
    {
        int64_t calcNum = this->r1 * this->r0;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(calcNum);
        uint32_t xyUbOffset = this->r1r0Align;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> y;
            RegTensor<float> mean_sum;
            RegTensor<float> mean;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;
            RegTensor<float> var_sum;
            RegTensor<float> var;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            uint32_t sreg0 = calcNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t k = 0; k < currentANum; k++) {
                LoadTensorForDtypeT<T1>(x, xInUb, pregLoop, (k * xyUbOffset));

                Muls(mean_sum, x, n, pregLoop);
                ReduceSum(mean, mean_sum, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregMerge);

                // save mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)batchMeanInUb + k), mean,
                                                                   pregMerge);
                Duplicate(mean, mean, pregMain);
                Muls(mean, mean, (float)-1.0, pregMain);

                LoadTensorForDtypeT<T1>(x1, xInUb, pregLoop, (k * xyUbOffset));
                Add(y1, x1, mean, pregLoop);
                Mul(y1Pow, y1, y1, pregLoop);
                Muls(var_sum, y1Pow, n, pregLoop);
                ReduceSum(var, var_sum, pregLoop);
                Muls(var, var, nCorrectionFactor, pregMerge);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)batchRstdInUb + k), var,
                                                                   pregMerge);
            }
        }
    }

    __aicore__ inline void CalculateMeanVarVF(__local_mem__ T1* xInUb, __local_mem__ float* batchMeanInUb,
                                              __local_mem__ float* batchRstdInUb, uint16_t currentANum)
    {
        int64_t reduceNum = this->r1 * this->r0;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(reduceNum);
        uint32_t xyUbOffset = this->r1r0Align;

        uint32_t binaryAddQuotientOffset = this->binaryAddQuotient;
        int64_t binaryAddRemainder = reduceNum - this->binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = ops::CeilDiv(binaryAddRemainder, static_cast<int64_t>(VL_FP32));
        uint16_t binaryAddQuotientLoop = ops::CeilDiv(this->binaryAddQuotient, static_cast<int64_t>(VL_FP32));

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddLoopMean = ((this->binaryAddQuotient / VL_FP32) / VL_FP32);
        uint16_t binaryAddLoopVar = binaryAddLoopMean;
        LocalTensor<float> binaryAddTensor = binaryAddBuf.Get<float>();
        __local_mem__ float* binaryAddTensorAddr = (__local_mem__ float*)binaryAddTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> mean_sum;
            RegTensor<float> mean;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;
            RegTensor<float> var_sum;
            RegTensor<float> var;

            RegTensor<float> binaryAddQ;
            RegTensor<float> binaryAddR;
            RegTensor<float> vlMean;

            RegTensor<float> binaryAddQPow;
            RegTensor<float> binaryAddRPow;
            RegTensor<float> vlVar;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t k = 0; k < currentANum; k++) {
                uint32_t sreg0 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT<T1>(binaryAddQ, binaryAddR, xInUb, xInUb, pregLoop, pregLoop,
                                               (i * VL_FP32 + k * xyUbOffset),
                                               (i * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregLoop);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                    ReduceSum(vlMean, binaryAddQ, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)binaryAddTensorAddr + i),
                                                                       vlMean, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT<T1>(
                        binaryAddQ, binaryAddR, xInUb, xInUb, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregMain);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    ReduceSum(vlMean, binaryAddQ, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop - 1), vlMean, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadTensorForDtypeT<T1>(x, xInUb, pregMain,
                                            ((i + binaryAddRemainderLoop) * VL_FP32 + k * xyUbOffset));
                    Muls(x, x, n, pregMain);
                    ReduceSum(vlMean, x, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop + i), vlMean, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopMean = binaryAddLoopMean;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopMean = curBinaryAddLoopMean / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddTensorAddr + j * VL_FP32));
                        DataCopy(binaryAddR,
                                 ((__local_mem__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopMean) * VL_FP32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddTensorAddr + j * VL_FP32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t sreg2 = this->binaryAddLastNum;
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(mean_sum, ((__local_mem__ float*)binaryAddTensorAddr));
                    ReduceSum(mean, mean_sum, pregLoop);
                    Muls(mean, mean, nCorrectionFactor, pregMerge);
                }

                // batch mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)batchMeanInUb + k), mean,
                                                                   pregMerge);
                Duplicate(mean, mean, pregMain);
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

                uint32_t sreg1 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT<T1>(binaryAddQ, binaryAddR, xInUb, xInUb, pregLoop, pregLoop,
                                               (i * VL_FP32 + k * xyUbOffset),
                                               (i * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregLoop);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
                    ReduceSum(vlVar, binaryAddQPow, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)binaryAddTensorAddr + i),
                                                                       vlVar, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT<T1>(
                        binaryAddQ, binaryAddR, xInUb, xInUb, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_FP32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Sub(binaryAddQ, binaryAddQ, mean, pregMain);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregMain);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregMain);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregMain);
                    ReduceSum(vlVar, binaryAddQPow, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop - 1), vlVar, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadTensorForDtypeT<T1>(x1, xInUb, pregMain,
                                            ((i + binaryAddRemainderLoop) * VL_FP32 + k * xyUbOffset));
                    Sub(y1, x1, mean, pregMain);
                    Mul(y1Pow, y1, y1, pregMain);
                    Muls(y1Pow, y1Pow, n, pregMain);
                    ReduceSum(vlVar, y1Pow, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop + i), vlVar, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopVar = binaryAddLoopVar;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopVar = curBinaryAddLoopVar / 2;
                    for (uint16_t j = 0; j < curBinaryAddLoopVar; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddTensorAddr + j * VL_FP32));
                        DataCopy(binaryAddR,
                                 ((__local_mem__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopVar) * VL_FP32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddTensorAddr + j * VL_FP32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t sreg2 = this->binaryAddLastNum;
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(var_sum, ((__local_mem__ float*)binaryAddTensorAddr));
                    ReduceSum(var, var_sum, pregLoop);
                    Muls(var, var, nCorrectionFactor, pregMerge);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(((__local_mem__ float*)batchRstdInUb + k), var,
                                                                       pregMerge);
                }
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
            }
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
        bool vfUseRunningMeanVar = useRunningMeanVar;

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

    __aicore__ inline void CalculateNormalizeVF(__local_mem__ T1* xInUb, __local_mem__ T1* yInUb,
                                                __local_mem__ T2* betaInUb, __local_mem__ T2* gammaInUb,
                                                __local_mem__ float* batchMeanInUb, __local_mem__ float* batchRstdInUb,
                                                uint16_t currentANum)
    {
        int64_t calcNum = this->r1 * this->r0;
        uint32_t xyUbOffset = this->r1r0Align;
        uint16_t loopCount = this->r1r0LoopCount;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;

            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrtVar;

            RegTensor<float> beta;
            RegTensor<float> gamma;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            for (uint16_t k = 0; k < currentANum; k++) {
                LoadTwoTensorForDtypeTBrc<T2>(beta, gamma, betaInUb, gammaInUb, pregMain, pregMain, k, k);
                uint32_t sreg3 = calcNum;
                for (uint16_t i = 0; i < loopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg3);
                    LoadTensorForDtypeT<T1>(x2, xInUb, pregLoop, (i * VL_FP32 + k * xyUbOffset));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)batchMeanInUb + k));
                    Sub(x2, x2, mean, pregLoop);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtVar, ((__local_mem__ float*)batchRstdInUb + k));
                    Mul(y2, x2, rsqrtVar, pregLoop);
                    Mul(y2, y2, beta, pregLoop);
                    Add(y2, y2, gamma, pregLoop);
                    if constexpr (IsSameType<T1, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y2, pregLoop);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ half*)yInUb + i * VL_FP32 + k * xyUbOffset), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y2, pregLoop);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ bfloat16_t*)yInUb + i * VL_FP32 + k * xyUbOffset), xBf16, pregLoop);
                    } else {
                        DataCopy(((__local_mem__ float*)yInUb + i * VL_FP32 + k * xyUbOffset), y2, pregLoop);
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
    int64_t powerOfTwoForR;
    int64_t r1;
    int64_t aFactor;
    int64_t a;
    int64_t r0;
    int64_t r1r0Align;

    int64_t blockNum;
    int64_t aBlockFactor;
    int64_t singleA;

    int64_t r1r0LoopCount;

    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;

    float epsilon = 1e-5;
    float momentum = 0.1;
    float besselCorrectionFactor;
    float oneSubMomentum;

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

    TBuf<TPosition::VECCALC> binaryAddBuf;
};
}  // namespace BatchNormOps

#endif // NORM_BATCH_NORM_FULL_REDUCE_H