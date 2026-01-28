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

namespace BatchNormV3Ops {
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

template <typename T, typename T_GAMMA, typename T_RUNNING_MEAN>
class BatchNormV3BlockSplitR {
    static constexpr int32_t INDEXTWO = 2;
    static constexpr int32_t INDEXFOUR = 4;
    static constexpr int32_t INDEXEIGHT = 8;
    static constexpr int32_t INDEXSIXTEEN = 16;
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
        this->unbiasedEstimationCoeff =
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
        uint32_t nowCoreRConutPowOfTow = FindCofFactor(nowCoreRConut);
        uint32_t rPowOfTow = FindCofFactor(tilingData->patternR);
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
            MeanM2TensorInit(meanTensor, m2Tensor, calcLen);
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
            CopyInAllMeanVar(allMeanTensor, alllVarTensor, aLoopOffset);
            SetFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            WaitFlag<HardEvent::MTE2_V>(eIdMte2ToVec);
            LocalTensor<float> batchMeanTensor = batchMeanQueue.AllocTensor<float>();
            LocalTensor<float> batchRstdTensor = batchRstdQueue.AllocTensor<float>();
            LastFinalize(batchMeanTensor, batchRstdTensor, allMeanTensor, alllVarTensor, countTensor2, tmpTensor);
            // 反向同步，最后一次循环前，都需要set
            if (aUbLoopIdx < tilingData->aUbLoop - 1) {
                SetFlag<HardEvent::V_MTE2>(eIdVecToMte2);
            }
            LocalTensor<T_GAMMA> gammaTensor = gammaQueue.AllocTensor<T_GAMMA>();
            LocalTensor<T_GAMMA> betaTensor = betaQueue.AllocTensor<T_GAMMA>();
            CopyInGammaBeta(gammaTensor, betaTensor, aLoopOffset);
            gammaQueue.EnQue(gammaTensor);
            betaQueue.EnQue(betaTensor);
            if (blockIdx == 0) {
                LocalTensor<T_RUNNING_MEAN> runningMeanInTensor = runningMeanInQueue.AllocTensor<T_RUNNING_MEAN>();
                LocalTensor<T_RUNNING_MEAN> runningVarInTensor = runningVarInQueue.AllocTensor<T_RUNNING_MEAN>();
                CopyInRunningMeanVar(runningMeanInTensor, runningVarInTensor, aLoopOffset);
                runningMeanInQueue.EnQue(runningMeanInTensor);
                runningVarInQueue.EnQue(runningVarInTensor);
                runningMeanInTensor = runningMeanInQueue.template DeQue<T_RUNNING_MEAN>();
                runningVarInTensor = runningVarInQueue.template DeQue<T_RUNNING_MEAN>();
                LocalTensor<T_RUNNING_MEAN> runningMeanOutTensor = runningMeanOutQueue.AllocTensor<T_RUNNING_MEAN>();
                LocalTensor<T_RUNNING_MEAN> runningVarOutTensor = runningVarOutQueue.AllocTensor<T_RUNNING_MEAN>();
                CalculateRunningMeanVar(
                    runningMeanInTensor, runningVarInTensor, runningMeanOutTensor, runningVarOutTensor, batchMeanTensor,
                    batchRstdTensor);
                runningMeanInQueue.FreeTensor(runningMeanInTensor);
                runningVarInQueue.FreeTensor(runningVarInTensor);
                runningMeanOutQueue.EnQue(runningMeanOutTensor);
                runningVarOutQueue.EnQue(runningVarOutTensor);
                runningMeanOutTensor = runningMeanOutQueue.template DeQue<T_RUNNING_MEAN>();
                runningVarOutTensor = runningVarOutQueue.template DeQue<T_RUNNING_MEAN>();
                CopyOutRunningMeanVar(runningMeanOutTensor, runningVarOutTensor, aLoopOffset);
                runningMeanOutQueue.FreeTensor(runningMeanOutTensor);
                runningVarOutQueue.FreeTensor(runningVarOutTensor);
            }
            gammaTensor = gammaQueue.DeQue<T_GAMMA>();
            betaTensor = betaQueue.DeQue<T_GAMMA>();
            // 需要等runningMeanVar计算完成后，才能计算成Rstd
            CalculateBatchRstd(batchRstdTensor);
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
                CopyOutBatchMeanRstd(batchMeanTensor, batchRstdTensor, aLoopOffset);
            }
            batchMeanQueue.FreeTensor(batchMeanTensor);
            batchRstdQueue.FreeTensor(batchRstdTensor);
        }
    }

private:
    __aicore__ inline uint32_t FindCofFactor(uint32_t n)
    {
        // 找到比n大的最邻近的二次幂数, n = 15，结果为16
        if ((n & (n - 1)) != 0) {
            uint32_t temp = n - 1;
            temp |= temp >> 1;
            temp |= temp >> INDEXTWO;
            temp |= temp >> INDEXFOUR;
            temp |= temp >> INDEXEIGHT;
            temp |= temp >> INDEXSIXTEEN;
            return (temp + 1);
        } else {
            return n;
        }
    }

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
        uint16_t fisrstLoopCount = CEIL_DIV(firstNum, VL_F32);
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
            for (uint16_t i = 0; i < fisrstLoopCount; i++) {
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

    __aicore__ inline void MeanM2TensorInit(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor, uint32_t len)
    {
        __local_mem__ float* meanTensorAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* m2TensorAddr = (__local_mem__ float*)m2Tensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> tmpMean;
            RegTensor<float> tmpM2;
            MaskReg mask0 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            Duplicate(tmpMean, 0.0, mask0);
            Duplicate(tmpM2, 0.0, mask0);
            MaskReg mask1;
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask1 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(meanTensorAddr + i * VL_F32, tmpMean, mask1);
                DataCopy(m2TensorAddr + i * VL_F32, tmpM2, mask1);
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

    __aicore__ inline void CopyInAllMeanVar(
        LocalTensor<float>& allMeanTensor, LocalTensor<float>& alllVarTensor, int64_t aLoopOffset)
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
        copyInMeanVarParams.srcStride = (tilingData->patternAAlign - currentAAlign) * sizeof(float);
        DataCopyPad(allMeanTensor, workspaceGm[aLoopOffset], copyInMeanVarParams, meanVarPadParams);
        DataCopyPad(
            alllVarTensor, workspaceGm[usedCoreNum * tilingData->patternAAlign + aLoopOffset], copyInMeanVarParams,
            meanVarPadParams);
    }

    template <typename T_SRC>
    __aicore__ inline void LoadTensorForDtypeT(
        RegTensor<float>& dst, __local_mem__ T_SRC* input, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_SRC, half>::value) {
            RegTensor<half> xFp16;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
            Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
        } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xBf16;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
        } else {
            DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
        }
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
            RegTensor<float> delat4;
            MaskReg mask0;
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadTensorForDtypeT(x1, xTensorAddr, mask0, i * VL_F32);
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
                Mul(delat4, delta1, delta3, mask0);
                // M2 = M2 + delta4
                Add(tmpM2, tmpM2, delat4, mask0);
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

        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;

        float numScale = this->nFactor;
        float scaleCorrection = this->nCorrectionFactor;

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> tmpMean;
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

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // 尾块部分四行，和前面四行相加，最终是一行
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    // 前两行
                    TwoRowAddForMeanWithTail(
                        x1, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset, remOffset, quotOffset + rLoopStride,
                        remOffset + rLoopStride, quotCountOffset, remCountOffset, quotCountOffset + 1,
                        remCountOffset + 1, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount, nextRemCount,
                        numScale);
                    // 后两行
                    TwoRowAddForMeanWithTail(
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
                    TwoRowAddForMean(
                        x1, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset, baseOffset + rLoopStride,
                        baseCountOffset, baseCountOffset + 1, rem, rowCount, nextRowCount, numScale);
                    TwoRowAddForMean(
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
            RegTensor<float> tmpMean;
            RegTensor<float> saveMean;
            RegTensor<float> saveVar;

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

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(saveMean, ((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset));
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    TwoRowAddForVarWithTail(
                        x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, quotOffset, remOffset,
                        quotOffset + rLoopStride, remOffset + rLoopStride, quotCountOffset, remCountOffset,
                        quotCountOffset + 1, remCountOffset + 1, saveMean, rem, nextRow, remNextRow, rowCount,
                        nextRowCount, remCount, nextRemCount, rowM2, nextRowM2, remM2, nextRemM2, numScale);
                    TwoRowAddForVarWithTail(
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
                    TwoRowAddForVar(
                        x1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset, baseOffset + rLoopStride,
                        baseCountOffset, baseCountOffset + 1, saveMean, rem, rowCount, nextRowCount, rowM2, remM2,
                        numScale);
                    TwoRowAddForVar(
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

    __aicore__ inline void BinaryAddVF(
        __local_mem__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint16_t binaryAddKLoop,
        uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop, uint32_t offset,
        RegTensor<float>& x1, RegTensor<float>& x2, RegTensor<float>& x3, RegTensor<float>& x4)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                DataCopy(
                    x2, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                Add(x1, x1, x2, pregLoop);
                DataCopy(
                    x3, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride +
                         offset));
                DataCopy(
                    x4, ((__local_mem__ float*)binaryAddTmpAddr +
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

    __aicore__ inline void TwoRowAddForMeanWithTail(
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

    __aicore__ inline void TwoRowAddForMean(
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

    __aicore__ inline void TwoRowAddForVarWithTail(
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

    __aicore__ inline void TwoRowAddForVar(
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

    __aicore__ inline void LastFinalize(
        LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor, LocalTensor<float>& meanTensor,
        LocalTensor<float>& varTensor, LocalTensor<float>& countTensor, LocalTensor<float>& tmpTensor)
    {
        __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
        __local_mem__ float* tmpVarLocal = (__local_mem__ float*)varTensor.GetPhyAddr();
        __local_mem__ float* tmpCountLocal = (__local_mem__ float*)countTensor.GetPhyAddr();
        __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
        __local_mem__ float* tmpUbAddr = (__local_mem__ float*)tmpTensor.GetPhyAddr();
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        uint32_t rLoopStride = currentAAlign;
        uint16_t remainderLoopCount = (usedCoreNum - tilingData->lastBinaryAddQuotient);
        uint16_t quotientLoopCount = tilingData->lastBinaryAddQuotient - remainderLoopCount;
        uint32_t baseLineOffset = rLoopStride;
        uint32_t remainderOffset = tilingData->lastBinaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->lastBinaryAddQuotient;
        uint16_t binaryAddKLoop = tilingData->lastBinaryAddK;
        uint16_t binaryAddInnerLoop = tilingData->lastBinaryAddQuotient;
        uint16_t binaryAddLastLoop = tilingData->lastBinaryAddLast;
        float numScale = this->lastNFactor;
        float scaleCorrection = this->lastNCorrectionFactor;
        __VEC_SCOPE__
        {
            RegTensor<float> tmpMean;
            RegTensor<float> saveMean;
            RegTensor<float> quot;
            RegTensor<float> rem;
            RegTensor<float> quotCount;
            RegTensor<float> remCount;
            RegTensor<float> oriQuotMean;
            RegTensor<float> oriRemMean;
            RegTensor<float> resMean;
            RegTensor<float> resVar;

            MaskReg pregLoop;
            uint32_t sreg0 = currentA;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // 尾块部分按行加至前面
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i;
                    uint32_t remCountOffset = i + remainderCountOffset;
                    DataCopy(quot, ((__local_mem__ float*)(tmpMeanLocal) + (quotOffset)));
                    DataCopy(rem, ((__local_mem__ float*)(tmpMeanLocal) + (remOffset)));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(
                        quotCount, ((__local_mem__ float*)(tmpCountLocal) + quotCountOffset));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(
                        remCount, ((__local_mem__ float*)(tmpCountLocal) + remCountOffset));
                    Mul(quot, quot, quotCount, pregLoop);
                    Mul(rem, rem, remCount, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    Muls(rem, rem, numScale, pregLoop);
                    Add(quot, quot, rem, pregLoop);
                    DataCopy(((__local_mem__ float*)tmpUbAddr + i * rLoopStride + aLoopOffset), quot, pregLoop);
                }
                // 整块部分除已经叠加了尾块的，需要乘count和scale
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = remainderLoopCount + i;
                    DataCopy(quot, ((__local_mem__ float*)(tmpMeanLocal) + (baseOffset)));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(
                        quotCount, ((__local_mem__ float*)(tmpCountLocal) + baseCountOffset));
                    Mul(quot, quot, quotCount, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
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
                    DataCopy<float, LoadDist::DIST_BRC_B32>(
                        quotCount, ((__local_mem__ float*)(tmpCountLocal) + quotCountOffset));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(
                        remCount, ((__local_mem__ float*)(tmpCountLocal) + remCountOffset));
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
                    DataCopy<float, LoadDist::DIST_BRC_B32>(
                        quotCount, ((__local_mem__ float*)(tmpCountLocal) + baseCountOffset));
                    Sub(oriQuotMean, oriQuotMean, resMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, oriQuotMean, pregLoop);
                    Mul(oriQuotMean, oriQuotMean, quotCount, pregLoop);
                    Mul(quot, quot, quotCount, pregLoop);
                    Add(quot, quot, oriQuotMean, pregLoop);
                    Muls(quot, quot, numScale, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)tmpUbAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset), quot,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 最后对2的幂次行 二分累加
                BinaryAddVF(
                    tmpUbAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop, pregLoop,
                    aLoopOffset, quot, rem, quotCount, remCount);
                DataCopy(resVar, ((__local_mem__ float*)tmpUbAddr + aLoopOffset));
                Muls(resVar, resVar, scaleCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchRstdTensorAddr + aLoopOffset), resVar, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyInGammaBeta(LocalTensor<T_GAMMA>& gammaInUb, LocalTensor<T_GAMMA>& betaInUb, int64_t gmOffset)
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

    __aicore__ inline void CopyInRunningMeanVar(
        LocalTensor<T_RUNNING_MEAN>& runningMeanInUb, LocalTensor<T_RUNNING_MEAN>& runningVarInUb, int64_t gmOffset)
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

    __aicore__ inline void CalculateRunningMeanVar(
        LocalTensor<T_RUNNING_MEAN>& runningMeanInUb, LocalTensor<T_RUNNING_MEAN>& runningVarInUb,
        LocalTensor<T_RUNNING_MEAN>& runningMeanOutUb, LocalTensor<T_RUNNING_MEAN>& runningVarOutUb,
        LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor)
    {
        __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningMeanInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr =
            (__local_mem__ T_RUNNING_MEAN*)runningMeanOutUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarOutUb.GetPhyAddr();
        __local_mem__ float* batchMeanTensorAddr = (__local_mem__ float*)batchMeanTensor.GetPhyAddr();
        __local_mem__ float* batchRstdTensorTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
        uint16_t aLoop = CEIL_DIV(currentA, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> var;
            RegTensor<float> runningMean;
            RegTensor<float> saveMean;
            RegTensor<float> runningVar;
            RegTensor<float> saveVar;
            MaskReg pregLoop;
            uint32_t sreg2 = currentA;
            for (uint16_t k = 0; k < aLoop; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                // running var
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> runningVarTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        runningVarTmp, ((__local_mem__ T_RUNNING_MEAN*)runningVarInUbAddr + k * VL_F32));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(
                        runningVar, runningVarTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(runningVar, ((__local_mem__ float*)runningVarInUbAddr + k * VL_F32));
                }
                DataCopy(var, ((__local_mem__ float*)batchRstdTensorTensorAddr + k * VL_F32));
                Muls(saveVar, var, this->unbiasedEstimationCoeff, pregLoop);
                Muls(saveVar, saveVar, tilingData->momentum, pregLoop);
                Muls(runningVar, runningVar, tilingData->momentumReverse, pregLoop);
                Add(saveVar, saveVar, runningVar, pregLoop);
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把float的结果cast回T_RUNNING_MEAN
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> saveVarTmp;
                    AscendC::MicroAPI::Cast<T_RUNNING_MEAN, float, castTraitB322B16>(saveVarTmp, saveVar, pregLoop);
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        ((__local_mem__ T_RUNNING_MEAN*)runningVarOutUbAddr + k * VL_F32), saveVarTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(
                        ((__local_mem__ float*)runningVarOutUbAddr + k * VL_F32), saveVar, pregLoop);
                }
                // running mean
                DataCopy(mean, ((__local_mem__ float*)batchMeanTensorAddr + k * VL_F32));
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把T_RUNNING_MEAN的输入cast到float
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> runningMeanTmp;
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        runningMeanTmp, ((__local_mem__ T_RUNNING_MEAN*)runningMeanInUbAddr + k * VL_F32));
                    AscendC::MicroAPI::Cast<float, T_RUNNING_MEAN, castTraitB162B32>(
                        runningMean, runningMeanTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(runningMean, ((__local_mem__ float*)runningMeanInUbAddr + k * VL_F32));
                }
                Muls(saveMean, mean, tilingData->momentum, pregLoop);
                Muls(runningMean, runningMean, tilingData->momentumReverse, pregLoop);
                Add(saveMean, saveMean, runningMean, pregLoop);
                if constexpr (!IsSameType<T_RUNNING_MEAN, float>::value) {
                    // 需要把float的结果cast回T_RUNNING_MEAN
                    AscendC::MicroAPI::RegTensor<T_RUNNING_MEAN> saveMeanTmp;
                    AscendC::MicroAPI::Cast<T_RUNNING_MEAN, float, castTraitB322B16>(saveMeanTmp, saveMean, pregLoop);
                    AscendC::MicroAPI::DataCopy<T_RUNNING_MEAN, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                        ((__local_mem__ T_RUNNING_MEAN*)runningMeanOutUbAddr + k * VL_F32), saveMeanTmp, pregLoop);
                } else {
                    AscendC::MicroAPI::DataCopy(
                        ((__local_mem__ float*)runningMeanOutUbAddr + k * VL_F32), saveMean, pregLoop);
                }
            }
        }
    }

    __aicore__ inline void CalculateBatchRstd(LocalTensor<float>& batchRstdTensor)
    {
        __local_mem__ float* batchRstdTensorTensorAddr = (__local_mem__ float*)batchRstdTensor.GetPhyAddr();
        uint16_t aLoop = CEIL_DIV(currentA, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> one;
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
            RegTensor<float> rstd;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregLoop;
            Duplicate(one, 1.0, pregMain);
            uint32_t sreg2 = currentA;
            for (uint16_t k = 0; k < aLoop; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                DataCopy(var, ((__local_mem__ float*)batchRstdTensorTensorAddr + k * VL_F32));
                Duplicate(scalar1, float(0.5), pregLoop);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(t1, float(1.5), pregLoop);
                Duplicate(s, float(1.0), pregLoop);
                Adds(var, var, tilingData->epsilon, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(-0.5), pregLoop);
                Mul(t, t, y, pregLoop);                // -0.5 * x * y
                Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t2, var, float(-1.0), pregLoop);  // -1 * x
                Mula(s, t2, r, pregLoop);              // 1 + (-1) * x * r
                Muls(t3, rstd, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t3, rstd, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);             // s + x * t
                Mul(s, s, rstd, pregLoop);             // e * y
                Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(((__local_mem__ float*)batchRstdTensorTensorAddr + k * VL_F32), rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyOutRunningMeanVar(
        LocalTensor<T_RUNNING_MEAN>& runningMeanOutUb, LocalTensor<T_RUNNING_MEAN>& runningVarOutUb, int64_t gmOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(T_RUNNING_MEAN);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanOutGm[gmOffset], runningMeanOutUb, copyInParams);
        DataCopyPad(runningVarOutGm[gmOffset], runningVarOutUb, copyInParams);
    }

    __aicore__ inline void CopyOutBatchMeanRstd(
        LocalTensor<float>& batchMeanInUb, LocalTensor<float>& batchRstdInUb, int64_t gmOffset)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentA * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(batchMeanGm[gmOffset], batchMeanInUb, copyInParams);
        DataCopyPad(batchRstdGm[gmOffset], batchRstdInUb, copyInParams);
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
                LoadTensorForDtypeT(gamma, gammaTensorAddr, mask0, i * VL_F32);
                LoadTensorForDtypeT(beta, betaTensorAddr, mask0, i * VL_F32);
                for (uint16_t j = 0; j < lineLoop; j++) {
                    LoadTensorForDtypeT(x1, xTensorAddr, mask0, i * VL_F32 + j * currentAAlign);
                    Sub(x1, x1, mean, mask0);
                    Mul(x1, x1, rstd, mask0);
                    Mul(x1, x1, gamma, mask0);
                    Add(y, x1, beta, mask0);
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y, mask0);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            yTensorAddr + i * VL_F32 + j * currentAAlign, yFp16, mask0);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y, mask0);
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
    static constexpr int64_t T_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(T);
    static constexpr int64_t FP32_BLOCK_ALIGN_SIZE = BLOCK_SIZE / sizeof(float);
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t SCALE_COEF_FOUR = 4;
    static constexpr uint32_t ROW_TWO_OFFSET = 2;
    static constexpr uint32_t ROW_THREE_OFFSET = 3;
    static constexpr uint32_t ROW_FOUR_OFFSET = 4;
    static constexpr float POS_INF = 3.40282366920938E+38;

    constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN,
    };

    constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT,
    };

    /* ascendc variable */
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
