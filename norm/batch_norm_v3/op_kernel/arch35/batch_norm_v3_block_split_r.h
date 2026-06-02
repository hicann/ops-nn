/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_v3_block_split_r.h
 * \brief
 */
#ifndef BATCH_NORM_V3_BLOCK_SPLITR_REGBASE_H
#define BATCH_NORM_V3_BLOCK_SPLITR_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "batch_norm_v3_regbase_common.h"

namespace BatchNormV3Ops {
using namespace AscendC;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename T, typename T_GAMMA, typename T_RUNNING_MEAN>
class BatchNormV3BlockSplitR {
public:
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        return (y != 0) ? (x + y - 1) / y : 0;
    }

    __aicore__ inline uint32_t CEIL_ALIGN(uint32_t x, uint32_t y)
    {
        return CEIL_DIV(x, y) * y;
    }

    __aicore__ inline BatchNormV3BlockSplitR(const BatchNormV3BlockSplitRTilingData* tilingDataIn)
    {
        tilingData = tilingDataIn;
        this->unbiasedEstimationCoeff = tilingData->patternR == 1 ?
            AscendC::NumericLimits<float>::QuietNaN() :
            static_cast<float>(tilingData->patternR) / static_cast<float>(tilingData->patternR - 1);
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y, GM_ADDR mean_out, GM_ADDR var_out,
        GM_ADDR batch_mean, GM_ADDR batch_rstd, GM_ADDR workspace)
    {
        blockIdx = GetBlockIdx();
        usedCoreNum = GetBlockNum();
        int64_t rBlockOffset = 0;
        if (blockIdx < tilingData->formerCoreNums) {
            this->rLoop = tilingData->formerCoreBlockFactor;
            rBlockOffset = blockIdx * this->rLoop * tilingData->rUbFactor;
        } else {
            this->rLoop = tilingData->tailCoreBlockFactor;
            rBlockOffset = (tilingData->formerCoreBlockFactor * tilingData->formerCoreNums +
                            tilingData->tailCoreBlockFactor * (blockIdx - tilingData->formerCoreNums)) *
                           tilingData->rUbFactor;
        }
        uint32_t nowCoreRConut = (blockIdx == (usedCoreNum - 1)) ? tilingData->rUbFactor * rLoop + tilingData->tailR :
                                                                   tilingData->rUbFactor * rLoop;
        uint32_t nowCoreRConutPowOfTow = BatchNormV3FindCofFactor(nowCoreRConut);
        uint32_t rPowOfTow = BatchNormV3FindCofFactor(tilingData->patternR);
        this->nFactor = static_cast<float>(1) / static_cast<float>(nowCoreRConutPowOfTow);
        this->nCorrectionFactor = static_cast<float>(nowCoreRConutPowOfTow) / static_cast<float>(nowCoreRConut);
        this->lastNFactor = static_cast<float>(1) / static_cast<float>(rPowOfTow);
        this->lastNCorrectionFactor = static_cast<float>(rPowOfTow) / static_cast<float>(tilingData->patternR);

        xGm.SetGlobalBuffer((__gm__ T*)x + rBlockOffset * tilingData->patternA);
        betaGm.SetGlobalBuffer((__gm__ T_GAMMA*)beta);
        gammaGm.SetGlobalBuffer((__gm__ T_GAMMA*)gamma);
        runningMeanGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean);
        runningVarGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var);
        yGm.SetGlobalBuffer((__gm__ T*)y + rBlockOffset * tilingData->patternA);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean);
        batchRstdGm.SetGlobalBuffer((__gm__ float*)batch_rstd);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean_out);
        runningVarOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var_out);
        meanWsp.SetGlobalBuffer((__gm__ float*)workspace + blockIdx * tilingData->patternAAlign);
        varWsp.SetGlobalBuffer((__gm__ float*)workspace + (usedCoreNum + blockIdx) * tilingData->patternAAlign);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace);
        pipe.InitBuffer(xQueue, DOUBLE_BUFFER, tilingData->rUbFactor * tilingData->aUbFactor * sizeof(T));
        pipe.InitBuffer(yQueue, DOUBLE_BUFFER, tilingData->rUbFactor * tilingData->aUbFactor * sizeof(T));
        pipe.InitBuffer(gammaQueue, 1, tilingData->aUbFactor * sizeof(T_GAMMA));
        pipe.InitBuffer(betaQueue, 1, tilingData->aUbFactor * sizeof(T_GAMMA));
        pipe.InitBuffer(batchMeanQueue, 1, tilingData->aUbFactor * sizeof(float));
        pipe.InitBuffer(batchRstdQueue, 1, tilingData->aUbFactor * sizeof(float));
        pipe.InitBuffer(runningMeanInQueue, 1, tilingData->aUbFactor * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningVarInQueue, 1, tilingData->aUbFactor * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningMeanOutQueue, 1, tilingData->aUbFactor * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningVarOutQueue, 1, tilingData->aUbFactor * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(tmpTbuf1, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe.InitBuffer(tmpTbuf2, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe.InitBuffer(tmpTbuf3, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        int64_t rUbFactorAlign = CEIL_ALIGN(tilingData->rUbFactor, FP32_BLOCK_ALIGN_SIZE);
        int64_t usedCoreNumAlign = CEIL_ALIGN(usedCoreNum, FP32_BLOCK_ALIGN_SIZE);
        pipe.InitBuffer(countTbuf1, rUbFactorAlign * sizeof(float));
        pipe.InitBuffer(countTbuf2, usedCoreNumAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> meanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> m2Tensor = tmpTbuf2.Get<float>();
        LocalTensor<float> tmpTensor = tmpTbuf3.Get<float>();
        LocalTensor<float> countTensor1 = countTbuf1.Get<float>();
        LocalTensor<float> countTensor2 = countTbuf2.Get<float>();
        int64_t aLoopOffset = 0;
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < tilingData->aUbLoop; aUbLoopIdx++) {
            currentA = tilingData->aUbFactor;
            currentAAlign = tilingData->aUbFactor;
            if (aUbLoopIdx == (tilingData->aUbLoop - 1)) {
                currentA = tilingData->aUbTail;
                currentAAlign = CEIL_ALIGN(tilingData->aUbTail, T_BLOCK_ALIGN_SIZE);
            }
            aLoopOffset = aUbLoopIdx * tilingData->aUbFactor;
            uint32_t calcLen = tilingData->rUbFactor * currentAAlign;
            currentR = tilingData->rUbFactor;
            uint16_t meanM2LoopCount = CEIL_DIV(calcLen, VL_F32);
            BatchNormV3MeanM2TensorInit(meanTensor, m2Tensor, calcLen, meanM2LoopCount, VL_F32);
            int64_t xGmOffset = 0;
            int64_t count = 0;
            for (int64_t rUbLoopIdx = 0; rUbLoopIdx < this->rLoop; rUbLoopIdx++) {
                xGmOffset = rUbLoopIdx * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                WelfordParallelUpdate(count, meanTensor, m2Tensor, xGmOffset, calcLen);
            }
            if ((tilingData->tailR != 0) && (blockIdx == (usedCoreNum - 1))) {
                calcLen = tilingData->tailR * currentAAlign;
                currentR = tilingData->tailR;
                xGmOffset = this->rLoop * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                WelfordParallelUpdate(count, meanTensor, m2Tensor, xGmOffset, calcLen);
            }
            CaculateCountBuf(countTensor1, countTensor2);
            LocalTensor<float> localMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> localVarTensor = batchRstdQueue.AllocTensor<float>();
            ProcessWelfordFinalize(meanTensor, m2Tensor, countTensor1, localMeanTensor, localVarTensor, tmpTensor);
            batchMeanQueue.EnQue(localMeanTensor);
            batchRstdQueue.EnQue(localVarTensor);
            localMeanTensor = batchMeanQueue.template DeQue<float>();
            localVarTensor = batchRstdQueue.template DeQue<float>();
            DataCopy(meanWsp[aLoopOffset], localMeanTensor, currentAAlign);
            DataCopy(varWsp[aLoopOffset], localVarTensor, currentAAlign);
            batchMeanQueue.FreeTensor(localMeanTensor);
            batchRstdQueue.FreeTensor(localVarTensor);
        }
        SyncAll();
        LocalTensor<float> allMeanTensor = tmpTbuf1.Get<float>();
        LocalTensor<float> alllVarTensor = tmpTbuf2.Get<float>();
        event_t eIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eIdVecToMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < tilingData->aUbLoop; aUbLoopIdx++) {
            currentA = tilingData->aUbFactor;
            currentAAlign = tilingData->aUbFactor;
            if (aUbLoopIdx == (tilingData->aUbLoop - 1)) {
                currentA = tilingData->aUbTail;
                currentAAlign = CEIL_ALIGN(tilingData->aUbTail, T_BLOCK_ALIGN_SIZE);
            }
            aLoopOffset = aUbLoopIdx * tilingData->aUbFactor;
            // 反向同步，第2次的搬入需要第1次内存free
            if (aUbLoopIdx > 0) {
                WaitFlag<HardEvent::V_MTE2>(eIdVecToMte2);
            }
            CopyInAllMeanVarPad(allMeanTensor, alllVarTensor, workspaceGm, aLoopOffset,
                usedCoreNum * tilingData->patternAAlign + aLoopOffset, static_cast<uint32_t>(usedCoreNum),
                static_cast<uint32_t>(currentAAlign), static_cast<uint32_t>(tilingData->patternAAlign));
            SetFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            WaitFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            LocalTensor<float> batchMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> batchRstdTensor = batchRstdQueue.AllocTensor<float>();
            LastFinalizeVF(batchMeanTensor, batchRstdTensor, allMeanTensor, alllVarTensor, countTensor2, tmpTensor,
                static_cast<uint32_t>(currentAAlign), VL_F32, static_cast<uint16_t>(currentA),
                static_cast<uint16_t>(usedCoreNum), static_cast<uint16_t>(tilingData->lastBinaryAddQuotient),
                static_cast<uint16_t>(tilingData->lastBinaryAddK),
                static_cast<uint16_t>(tilingData->lastBinaryAddLast), this->lastNFactor,
                this->lastNCorrectionFactor);
            // 反向同步，最后一次循环前，都需要set
            if (aUbLoopIdx < tilingData->aUbLoop - 1) {
                SetFlag<HardEvent::V_MTE2>(eIdVecToMte2);
            }
            LocalTensor<T_GAMMA> gammaTensor = gammaQueue.AllocTensor<T_GAMMA>();
            LocalTensor<T_GAMMA> betaTensor = betaQueue.AllocTensor<T_GAMMA>();
            CopyInGammaBetaPad(
                gammaTensor, betaTensor, gammaGm, betaGm, aLoopOffset, static_cast<uint32_t>(currentA));
            gammaQueue.EnQue(gammaTensor);
            betaQueue.EnQue(betaTensor);
            if (blockIdx == 0) {
                uint16_t aLoop = CEIL_DIV(currentA, VL_F32);
                UpdateRunningMeanVarCommon<T_RUNNING_MEAN>(batchMeanTensor, batchRstdTensor, runningMeanInQueue,
                    runningVarInQueue, runningMeanOutQueue, runningVarOutQueue, runningMeanGm, runningVarGm,
                    runningMeanOutGm, runningVarOutGm, aLoopOffset, static_cast<uint32_t>(currentA), aLoop, VL_F32,
                    this->unbiasedEstimationCoeff, tilingData->momentum, tilingData->momentumReverse);
            }
            gammaTensor = gammaQueue.DeQue<T_GAMMA>();
            betaTensor = betaQueue.DeQue<T_GAMMA>();
            // 需要等runningMeanVar计算完成后，才能计算成Rstd
            NormCommon::ComputeRstdNewtonRaphson<false>(
                batchRstdTensor, batchRstdTensor, static_cast<uint32_t>(currentA), tilingData->epsilon, 1.0f, VL_F32);
            int64_t yGmOffset = 0;
            currentR = tilingData->rUbFactor;
            for (int64_t rUbLoopIdx = 0; rUbLoopIdx < this->rLoop; rUbLoopIdx++) {
                yGmOffset = rUbLoopIdx * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                NormalizeX(batchMeanTensor, batchRstdTensor, gammaTensor, betaTensor, yGmOffset);
            }
            if ((tilingData->tailR != 0) && (blockIdx == (usedCoreNum - 1))) {
                currentR = tilingData->tailR;
                yGmOffset = this->rLoop * tilingData->rUbFactor * tilingData->patternA + aLoopOffset;
                NormalizeX(batchMeanTensor, batchRstdTensor, gammaTensor, betaTensor, yGmOffset);
            }
            gammaQueue.FreeTensor(gammaTensor);
            betaQueue.FreeTensor(betaTensor);
            if (blockIdx == 0) {
                batchMeanQueue.EnQue(batchMeanTensor);
                batchRstdQueue.EnQue(batchRstdTensor);
                batchMeanTensor = batchMeanQueue.template DeQue<float>();
                batchRstdTensor = batchRstdQueue.template DeQue<float>();
                CopyOutBatchMeanRstdPad(batchMeanTensor, batchRstdTensor, batchMeanGm, batchRstdGm, aLoopOffset,
                                        static_cast<uint32_t>(currentA));
            }
            batchMeanQueue.FreeTensor(batchMeanTensor);
            batchRstdQueue.FreeTensor(batchRstdTensor);
        }
    }

private:

    __aicore__ inline void CaculateCountBuf(LocalTensor<float>& tCountTensor1, LocalTensor<float>& tCountTensor2)
    {
        __local_mem__ float* tmpCountLocal1 = (__local_mem__ float*)tCountTensor1.GetPhyAddr();
        __local_mem__ float* tmpCountLocal2 = (__local_mem__ float*)tCountTensor2.GetPhyAddr();
        float baseAddCount = static_cast<float>(this->rLoop);
        float tailAddCount = static_cast<float>(this->rLoop + 1);
        uint32_t baseNum = tilingData->rUbFactor;
        uint32_t tailNum = (blockIdx == (usedCoreNum - 1)) ? tilingData->tailR : 0;
        uint16_t baseLoopCount = CEIL_DIV(baseNum, VL_F32);
        uint16_t tailLoopCount = CEIL_DIV(tailNum, VL_F32);
        float lastCoreAddCount =
            static_cast<float>(tilingData->tailR + tilingData->tailCoreBlockFactor * tilingData->rUbFactor);
        if (tilingData->tailCoreNums == 0) {
            lastCoreAddCount =
                static_cast<float>(tilingData->tailR + tilingData->formerCoreBlockFactor * tilingData->rUbFactor);
        }
        float tailCoreAddCount = static_cast<float>(tilingData->tailCoreBlockFactor * tilingData->rUbFactor);
        float formerCoreAddCount = static_cast<float>(tilingData->formerCoreBlockFactor * tilingData->rUbFactor);
        uint32_t firstNum = usedCoreNum;
        uint32_t secondNum = usedCoreNum - 1;
        uint32_t thirdNum = usedCoreNum - tilingData->tailCoreNums;
        if (tilingData->tailCoreNums == 0) {
            secondNum = 0;
            thirdNum = usedCoreNum - 1;
        }
        uint16_t firstLoopCount = CEIL_DIV(firstNum, VL_F32);
        uint16_t secondLoopCount = CEIL_DIV(secondNum, VL_F32);
        uint16_t thirdLoopCount = CEIL_DIV(thirdNum, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> tmpCount;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sreg1 = baseNum;
            Duplicate(tmpCount, baseAddCount, pregMain);
            for (uint16_t i = 0; i < baseLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                DataCopy(((__local_mem__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg2 = tailNum;
            Duplicate(tmpCount, tailAddCount, pregMain);
            for (uint16_t i = 0; i < tailLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg2);
                DataCopy(((__local_mem__ float*)tmpCountLocal1 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg3 = firstNum;
            Duplicate(tmpCount, lastCoreAddCount, pregMain);
            for (uint16_t i = 0; i < firstLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg3);
                DataCopy(((__local_mem__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg4 = secondNum;
            Duplicate(tmpCount, tailCoreAddCount, pregMain);
            for (uint16_t i = 0; i < secondLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg4);
                DataCopy(((__local_mem__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg5 = thirdNum;
            Duplicate(tmpCount, formerCoreAddCount, pregMain);
            for (uint16_t i = 0; i < thirdLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg5);
                DataCopy(((__local_mem__ float*)tmpCountLocal2 + i * VL_F32), tmpCount, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyInX(LocalTensor<T>& xInUb, int64_t offset)
    {
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = (currentA != currentAAlign);
        dataCopyPadExtParams.leftPadding = 0;
        // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentR;
        copyInParams.blockLen = currentA * sizeof(T);
        copyInParams.srcStride = (tilingData->patternA - currentA) * sizeof(T);
        copyInParams.dstStride = 0;
        DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
    }

    __aicore__ inline void WelfordParallelUpdate(
        int64_t& count, LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor, int64_t xGmOffset, uint32_t len)
    {
        // copy in x
        LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
        CopyInX(xTensor, xGmOffset);
        xQueue.EnQue(xTensor);
        xTensor = xQueue.DeQue<T>();
        // ---------
        count += 1;
        float scale = (float)1.0 / static_cast<float>(count);
        __local_mem__ float* meanTensorAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* m2TensorAddr = (__local_mem__ float*)m2Tensor.GetPhyAddr();
        __local_mem__ T* xTensorAddr = (__local_mem__ T*)xTensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpM2;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delta4;
            MaskReg mask0;
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensorForDtypeT(xTensorAddr, x1, mask0, i * VL_F32);
                DataCopy(tmpMean, meanTensorAddr + i * VL_F32);
                DataCopy(tmpM2, m2TensorAddr + i * VL_F32);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, mask0);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, mask0);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, mask0);
                DataCopy(meanTensorAddr + i * VL_F32, tmpMean, mask0);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, mask0);
                // delta4 = delta1 * delta3
                Mul(delta4, delta1, delta3, mask0);
                // M2 = M2 + delta4
                Add(tmpM2, tmpM2, delta4, mask0);
                DataCopy(m2TensorAddr + i * VL_F32, tmpM2, mask0);
            }
        }
        xQueue.FreeTensor(xTensor);
    }

    __aicore__ inline void ProcessWelfordFinalize(
        LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor, LocalTensor<float>& countTensor,
        LocalTensor<float>& finalMeanTensor, LocalTensor<float>& finalVarTensor, LocalTensor<float>& tmpTensor)

    {
        __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* tmpVarLocal = (__local_mem__ float*)m2Tensor.GetPhyAddr();
        __local_mem__ float* tmpCountLocal = (__local_mem__ float*)countTensor.GetPhyAddr();
        __local_mem__ float* batchMeanInUbAddr = (__local_mem__ float*)finalMeanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdInUbAddr = (__local_mem__ float*)finalVarTensor.GetPhyAddr();
        __local_mem__ float* tmpUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr();
        WelfordFinalizeMeanVF(
            tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchRstdInUbAddr, tmpUbAddr);
        WelfordFinalizeVarVF(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchRstdInUbAddr, tmpUbAddr);
    }

    __aicore__ inline void WelfordFinalizeMeanVF(
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* tmpCountLocal,
        __local_mem__ float* batchMeanInUbAddr, __local_mem__ float* batchRstdInUbAddr,
        __local_mem__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = tilingData->rUbFactor;
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;

        uint16_t remainderLoopCount = (tilingData->rUbFactor - tilingData->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (tilingData->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = tilingData->binaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->binaryAddQuotient;

        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;

        float scaleCorrection = this->nCorrectionFactor;
        float numScale = this->nFactor;

        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {

            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x4;
            RegTensor<float> x3;

            RegTensor<float> rem;
            RegTensor<float> nextRow;
            RegTensor<float> remNextRow;

            RegTensor<float> rowCount;
            RegTensor<float> nextRemCount;
            RegTensor<float> remCount;
            RegTensor<float> nextRowCount;


            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // 尾块部分四行，和前面四行相加，最终是一行
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    // 前两行
                    TwoRowAddPartialMeanWithTail(
                        x1, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset, remOffset, quotOffset + rLoopStride,
                        remOffset + rLoopStride, quotCountOffset, remCountOffset, quotCountOffset + 1,
                        remCountOffset + 1, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount, nextRemCount,
                        numScale);
                    // 后两行
                    TwoRowAddPartialMeanWithTail(
                        x2, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset + twoRLoopSize, remOffset + twoRLoopSize,
                        quotOffset + threeRLoopSize, remOffset + threeRLoopSize, quotCountOffset + ROW_TWO_OFFSET,
                        remCountOffset + ROW_TWO_OFFSET, quotCountOffset + ROW_THREE_OFFSET,
                        remCountOffset + ROW_THREE_OFFSET, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount,
                        nextRemCount, numScale);
                    Add(x1, x1, x2, pregLoop);
                    DataCopy(((__local_mem__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), x1, pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理4行，4行加成1行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddPartialMean(
                        x1, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset, baseOffset + rLoopStride,
                        baseCountOffset, baseCountOffset + 1, rem, rowCount, nextRowCount, numScale);
                    TwoRowAddPartialMean(
                        x2, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset + twoRLoopSize,
                        baseOffset + threeRLoopSize, baseCountOffset + ROW_TWO_OFFSET,
                        baseCountOffset + ROW_THREE_OFFSET, rem, rowCount, nextRowCount, numScale);
                    Add(x1, x1, x2, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                        x1, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
                BinaryAddVF(
                    binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop, pregLoop,
                    aLoopOffset, x1, x2, x3, x4);
                DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + aLoopOffset));
                Muls(x1, x1, scaleCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void WelfordFinalizeVarVF(
        __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal, __local_mem__ float* tmpCountLocal,
        __local_mem__ float* batchMeanInUbAddr, __local_mem__ float* batchRstdInUbAddr,
        __local_mem__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = tilingData->rUbFactor;
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;

        uint16_t remainderLoopCount = (tilingData->rUbFactor - tilingData->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (tilingData->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = tilingData->binaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->binaryAddQuotient;

        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;

        float numScale = this->nFactor;
        float scaleCorrection = this->nCorrectionFactor;

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> saveMean;

            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            RegTensor<float> rowCount;
            RegTensor<float> nextRowCount;
            RegTensor<float> remCount;
            RegTensor<float> nextRemCount;

            RegTensor<float> rowM2;
            RegTensor<float> nextRowM2;
            RegTensor<float> remM2;
            RegTensor<float> nextRemM2;

            uint32_t sreg0 = currentA;
            MaskReg pregLoop;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(saveMean, ((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset));
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    TwoRowAddPartialVarWithTail(
                        x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, quotOffset, remOffset,
                        quotOffset + rLoopStride, remOffset + rLoopStride, quotCountOffset, remCountOffset,
                        quotCountOffset + 1, remCountOffset + 1, saveMean, rem, nextRow, remNextRow, rowCount,
                        nextRowCount, remCount, nextRemCount, rowM2, nextRowM2, remM2, nextRemM2, numScale);
                    TwoRowAddPartialVarWithTail(
                        x2, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, quotOffset + twoRLoopSize,
                        remOffset + twoRLoopSize, quotOffset + threeRLoopSize, remOffset + threeRLoopSize,
                        quotCountOffset + ROW_TWO_OFFSET, remCountOffset + ROW_TWO_OFFSET,
                        quotCountOffset + ROW_THREE_OFFSET, remCountOffset + ROW_THREE_OFFSET, saveMean, rem, nextRow,
                        remNextRow, rowCount, nextRowCount, remCount, nextRemCount, rowM2, nextRowM2, remM2, nextRemM2,
                        numScale);
                    Add(x1, x1, x2, pregLoop);
                    DataCopy(((__local_mem__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), x1, pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理4行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddPartialVar(
                        x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset, baseOffset + rLoopStride,
                        baseCountOffset, baseCountOffset + 1, saveMean, rem, rowCount, nextRowCount, rowM2, remM2,
                        numScale);
                    TwoRowAddPartialVar(
                        x2, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset + twoRLoopSize,
                        baseOffset + threeRLoopSize, baseCountOffset + ROW_TWO_OFFSET,
                        baseCountOffset + ROW_THREE_OFFSET, saveMean, rem, rowCount, nextRowCount, rowM2, remM2,
                        numScale);
                    Add(x1, x1, x2, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                        x1, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF(
                    binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop, pregLoop,
                    aLoopOffset, x1, x2, x3, x4);
                DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + aLoopOffset));
                Muls(x1, x1, scaleCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchRstdInUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void NormalizeX(
        LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor, LocalTensor<T_GAMMA>& gammaTensor,
        LocalTensor<T_GAMMA>& betaTensor, int64_t yGmOffset)
    {
        LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
        CopyInX(xTensor, yGmOffset);
        xQueue.EnQue(xTensor);
        xTensor = xQueue.DeQue<T>();
        LocalTensor<T> yTensor = yQueue.AllocTensor<T>();
        CalcY(batchMeanTensor, batchRstdTensor, gammaTensor, betaTensor, xTensor, yTensor);
        xQueue.FreeTensor(xTensor);
        yQueue.EnQue(yTensor);
        yTensor = yQueue.template DeQue<T>();
        CopyOutY(yTensor, yGmOffset);
        yQueue.FreeTensor(yTensor);
    }

    __aicore__ inline void CalcY(
        LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor, LocalTensor<T_GAMMA>& gammaTensor,
        LocalTensor<T_GAMMA>& betaTensor, LocalTensor<T>& xTensor, LocalTensor<T>& yTensor)
    {
        __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
        __local_mem__ T* xTensorAddr = (__local_mem__ T*)xTensor.GetPhyAddr();
        __local_mem__ T* yTensorAddr = (__local_mem__ T*)yTensor.GetPhyAddr();
        __local_mem__ T_GAMMA* gammaTensorAddr = (__local_mem__ T_GAMMA*)gammaTensor.GetPhyAddr();
        __local_mem__ T_GAMMA* betaTensorAddr = (__local_mem__ T_GAMMA*)betaTensor.GetPhyAddr();
        uint16_t numLoop = CEIL_DIV(currentA, VL_F32);
        uint16_t lineLoop = currentR;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> y;
            MaskReg mask0;
            uint32_t sreg0 = currentA;
            for (uint16_t i = 0; i < numLoop; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(mean, batchMeanTensorAddr + i * VL_F32);
                DataCopy(rstd, batchRstdTensorAddr + i * VL_F32);
                LoadOneTensorForDtypeT(gammaTensorAddr, gamma, mask0, i * VL_F32);
                LoadOneTensorForDtypeT(betaTensorAddr, beta, mask0, i * VL_F32);
                for (uint16_t j = 0; j < lineLoop; j++) {
                    LoadOneTensorForDtypeT(xTensorAddr, x1, mask0, i * VL_F32 + j * currentAAlign);
                    Sub(x1, x1, mean, mask0);
                    Mul(x1, x1, rstd, mask0);
                    Mul(x1, x1, gamma, mask0);
                    Add(y, x1, beta, mask0);
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, NormCommon::castTraitB322B16>(yFp16, y, mask0);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            yTensorAddr + i * VL_F32 + j * currentAAlign, yFp16, mask0);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, NormCommon::castTraitB322B16>(xBf16, y, mask0);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            yTensorAddr + i * VL_F32 + j * currentAAlign, xBf16, mask0);
                    } else {
                        DataCopy(yTensorAddr + i * VL_F32 + j * currentAAlign, y, mask0);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutY(LocalTensor<T>& yOutUb, int64_t offset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentR;
        copyInParams.blockLen = currentA * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (tilingData->patternA - currentA) * sizeof(T);
        DataCopyPad(yGm[offset], yOutUb, copyInParams);
    }

    /* global memory address */
    GlobalTensor<T> xGm;
    GlobalTensor<T_GAMMA> betaGm;
    GlobalTensor<T_GAMMA> gammaGm;
    GlobalTensor<T_RUNNING_MEAN> runningMeanGm;
    GlobalTensor<T_RUNNING_MEAN> runningVarGm;

    GlobalTensor<T> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchRstdGm;
    GlobalTensor<T_RUNNING_MEAN> runningMeanOutGm;
    GlobalTensor<T_RUNNING_MEAN> runningVarOutGm;
    GlobalTensor<float> meanWsp;
    GlobalTensor<float> varWsp;
    GlobalTensor<float> workspaceGm;

    const BatchNormV3BlockSplitRTilingData* tilingData;
    TPipe pipe;

    /* variable */
    int64_t rLoop = 0;
    int64_t currentA = 0;
    int64_t currentAAlign = 0;
    int64_t currentR = 0;
    float unbiasedEstimationCoeff = 0;

    float nFactor = 0;
    float nCorrectionFactor = 0;
    float lastNFactor = 0;
    float lastNCorrectionFactor = 0;

    uint32_t usedCoreNum = 0;
    uint32_t blockIdx = 0;

    static constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = 32;
    static constexpr int64_t FP32_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(float);
    static constexpr int64_t T_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(T);
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t SCALE_COEF_FOUR = 4;
    static constexpr uint32_t ROW_THREE_OFFSET = 3;
    static constexpr uint32_t ROW_TWO_OFFSET = 2;
    static constexpr uint32_t ROW_FOUR_OFFSET = 4;    /* ascendc variable */
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECIN, 1> betaQueue;
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchRstdQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> tmpTbuf1;
    TBuf<TPosition::VECCALC> tmpTbuf2;
    TBuf<TPosition::VECCALC> tmpTbuf3;
    TBuf<TPosition::VECCALC> countTbuf1;
    TBuf<TPosition::VECCALC> countTbuf2;
};
} // namespace BatchNormV3Ops
#endif
