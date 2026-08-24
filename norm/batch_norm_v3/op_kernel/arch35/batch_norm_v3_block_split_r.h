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
public:
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y) { return (y != 0) ? (x + y - 1) / y : 0; }

    __aicore__ inline uint32_t CEIL_ALIGN(uint32_t x, uint32_t y) { return CEIL_DIV(x, y) * y; }

    __aicore__ inline BatchNormV3BlockSplitR(const BatchNormV3BlockSplitRTilingData* tilingDataIn)
    {
        tilingData = tilingDataIn;
        this->unbiasedEstimationCoeff = tilingData->patternR == 1 ? AscendC::NumericLimits<float>::QuietNaN() :
                                                                    static_cast<float>(tilingData->patternR) /
                                                                        static_cast<float>(tilingData->patternR - 1);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_rstd,
                                GM_ADDR workspace)
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
        // gamma/beta 同生命周期,合并为一个 que:一次 alloc,beta 在 gamma 之上按 32B 对齐偏移
        gammaBetaHalf = CEIL_ALIGN(tilingData->aUbFactor, BLOCK_SIZE / sizeof(T_GAMMA));
        pipe.InitBuffer(gammaBetaQueue, 1, MERGED_QUE_NODE_NUM * gammaBetaHalf * sizeof(T_GAMMA));
        // batchMean/batchRstd 同生命周期(stage1 finalize 与 stage2 输出),合并为一个 VECOUT que
        batchMeanRstdHalf = CEIL_ALIGN(tilingData->aUbFactor, FP32_BLOCK_ALIGN_SIZE);
        pipe.InitBuffer(batchMeanRstdQueue, 1, MERGED_QUE_NODE_NUM * batchMeanRstdHalf * sizeof(float));
        // running mean/var 的 in 对、out 对各自同生命周期,分别合并为一个 que;var 在 mean 之上对齐偏移
        runningHalf = CEIL_ALIGN(tilingData->aUbFactor, BLOCK_SIZE / sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningMeanVarInQueue, 1, MERGED_QUE_NODE_NUM * runningHalf * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(runningMeanVarOutQueue, 1, MERGED_QUE_NODE_NUM * runningHalf * sizeof(T_RUNNING_MEAN));
        pipe.InitBuffer(tmpTbuf1, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe.InitBuffer(tmpTbuf2, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        pipe.InitBuffer(tmpTbuf3, tilingData->tBufUbFactor * tilingData->aUbFactor * sizeof(float));
        int64_t rUbFactorAlign = CEIL_ALIGN(tilingData->rUbFactor, FP32_BLOCK_ALIGN_SIZE);
        int64_t usedCoreNumAlign = CEIL_ALIGN(usedCoreNum, FP32_BLOCK_ALIGN_SIZE);
        pipe.InitBuffer(countTbuf1, rUbFactorAlign * sizeof(float));
        pipe.InitBuffer(countTbuf2, usedCoreNumAlign * sizeof(float));
        // 跨核 partial mean/var 同生命周期,合并为一个 VECIN 队列,DOUBLE_BUFFER 做跨块预取;var 在 mean 之上偏移
        allMeanVarHalf = usedCoreNumAlign * tilingData->aUbFactor; // usedCoreNumAlign 为 8 倍数,*4B 天然 32B 对齐
        pipe.InitBuffer(allMeanVarQueue, DOUBLE_BUFFER, MERGED_QUE_NODE_NUM * allMeanVarHalf * sizeof(float));
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
            LocalTensor<float> batchMeanRstdTensor = batchMeanRstdQueue.AllocTensor<float>();
            LocalTensor<float> localMeanTensor = batchMeanRstdTensor;
            LocalTensor<float> localVarTensor = batchMeanRstdTensor[batchMeanRstdHalf];
            ProcessWelfordFinalize(meanTensor, m2Tensor, countTensor1, localMeanTensor, localVarTensor, tmpTensor);
            batchMeanRstdQueue.EnQue(batchMeanRstdTensor);
            batchMeanRstdTensor = batchMeanRstdQueue.template DeQue<float>();
            localMeanTensor = batchMeanRstdTensor;
            localVarTensor = batchMeanRstdTensor[batchMeanRstdHalf];
            DataCopy(meanWsp[aLoopOffset], localMeanTensor, currentAAlign);
            DataCopy(varWsp[aLoopOffset], localVarTensor, currentAAlign);
            batchMeanRstdQueue.FreeTensor(batchMeanRstdTensor);
        }
        SyncAll();
        for (int64_t aUbLoopIdx = 0; aUbLoopIdx < tilingData->aUbLoop; aUbLoopIdx++) {
            currentA = tilingData->aUbFactor;
            currentAAlign = tilingData->aUbFactor;
            if (aUbLoopIdx == (tilingData->aUbLoop - 1)) {
                currentA = tilingData->aUbTail;
                currentAAlign = CEIL_ALIGN(tilingData->aUbTail, T_BLOCK_ALIGN_SIZE);
            }
            aLoopOffset = aUbLoopIdx * tilingData->aUbFactor;
            // 跨核 partial mean/var 同生命周期合并为一个 VECIN 队列,EnQue/DeQue 自动管 MTE2->V 同步;var 在 mean
            // 之上偏移
            LocalTensor<float> allMeanVarTensor = allMeanVarQueue.AllocTensor<float>();
            LocalTensor<float> allMeanTensor = allMeanVarTensor;
            LocalTensor<float> allVarTensor = allMeanVarTensor[allMeanVarHalf];
            CopyInAllMeanVarPad(allMeanTensor, allVarTensor, workspaceGm, aLoopOffset,
                                usedCoreNum * tilingData->patternAAlign + aLoopOffset,
                                static_cast<uint32_t>(usedCoreNum), static_cast<uint32_t>(currentAAlign),
                                static_cast<uint32_t>(tilingData->patternAAlign));
            allMeanVarQueue.EnQue(allMeanVarTensor);
            allMeanVarTensor = allMeanVarQueue.DeQue<float>();
            allMeanTensor = allMeanVarTensor;
            allVarTensor = allMeanVarTensor[allMeanVarHalf];
            LocalTensor<float> batchMeanRstdTensor = batchMeanRstdQueue.AllocTensor<float>();
            LocalTensor<float> batchMeanTensor = batchMeanRstdTensor;
            LocalTensor<float> batchRstdTensor = batchMeanRstdTensor[batchMeanRstdHalf];
            LastFinalizeVF<true>(
                batchMeanTensor, batchRstdTensor, allMeanTensor, allVarTensor, countTensor2, tmpTensor,
                static_cast<uint32_t>(currentAAlign), VL_F32, static_cast<uint16_t>(currentA),
                static_cast<uint16_t>(usedCoreNum), static_cast<uint16_t>(tilingData->lastBinaryAddQuotient),
                static_cast<uint16_t>(tilingData->lastBinaryAddK), static_cast<uint16_t>(tilingData->lastBinaryAddLast),
                this->lastNFactor, this->lastNCorrectionFactor);
            allMeanVarQueue.FreeTensor(allMeanVarTensor);
            LocalTensor<T_GAMMA> gammaBetaTensor = gammaBetaQueue.AllocTensor<T_GAMMA>();
            LocalTensor<T_GAMMA> gammaTensor = gammaBetaTensor;
            LocalTensor<T_GAMMA> betaTensor = gammaBetaTensor[gammaBetaHalf];
            CopyInGammaBetaPad(gammaTensor, betaTensor, gammaGm, betaGm, aLoopOffset, static_cast<uint32_t>(currentA));
            gammaBetaQueue.EnQue(gammaBetaTensor);
            if (blockIdx == 0) {
                uint16_t aLoop = CEIL_DIV(currentA, VL_F32);
                UpdateRunningMeanVarCommon<T_RUNNING_MEAN>(
                    batchMeanTensor, batchRstdTensor, runningMeanVarInQueue, runningMeanVarOutQueue, runningMeanGm,
                    runningVarGm, runningMeanOutGm, runningVarOutGm, aLoopOffset, static_cast<uint32_t>(currentA),
                    aLoop, VL_F32, this->unbiasedEstimationCoeff, tilingData->momentum, tilingData->momentumReverse,
                    runningHalf);
            }
            gammaBetaTensor = gammaBetaQueue.DeQue<T_GAMMA>();
            gammaTensor = gammaBetaTensor;
            betaTensor = gammaBetaTensor[gammaBetaHalf];
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
            gammaBetaQueue.FreeTensor(gammaBetaTensor);
            if (blockIdx == 0) {
                batchMeanRstdQueue.EnQue(batchMeanRstdTensor);
                batchMeanRstdTensor = batchMeanRstdQueue.template DeQue<float>();
                batchMeanTensor = batchMeanRstdTensor;
                batchRstdTensor = batchMeanRstdTensor[batchMeanRstdHalf];
                CopyOutBatchMeanRstdPad(batchMeanTensor, batchRstdTensor, batchMeanGm, batchRstdGm, aLoopOffset,
                                        static_cast<uint32_t>(currentA));
            }
            batchMeanRstdQueue.FreeTensor(batchMeanRstdTensor);
        }
    }

private:
    __aicore__ inline void CaculateCountBuf(LocalTensor<float>& tCountTensor1, LocalTensor<float>& tCountTensor2)
    {
        __ubuf__ float* tmpCountLocal1 = (__ubuf__ float*)tCountTensor1.GetPhyAddr();
        __ubuf__ float* tmpCountLocal2 = (__ubuf__ float*)tCountTensor2.GetPhyAddr();
        float baseAddCount = static_cast<float>(this->rLoop);
        float tailAddCount = static_cast<float>(this->rLoop + 1);
        uint32_t baseNum = tilingData->rUbFactor;
        uint32_t tailNum = (blockIdx == (usedCoreNum - 1)) ? tilingData->tailR : 0;
        uint16_t baseLoopCount = CEIL_DIV(baseNum, VL_F32);
        uint16_t tailLoopCount = CEIL_DIV(tailNum, VL_F32);
        float lastCoreAddCount = static_cast<float>(tilingData->tailR +
                                                    tilingData->tailCoreBlockFactor * tilingData->rUbFactor);
        if (tilingData->tailCoreNums == 0) {
            lastCoreAddCount = static_cast<float>(tilingData->tailR +
                                                  tilingData->formerCoreBlockFactor * tilingData->rUbFactor);
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
            FillCountBlock(tmpCountLocal1, tmpCount, pregMain, pregLoop, baseAddCount, baseNum, baseLoopCount, VL_F32);
            FillCountBlock(tmpCountLocal1, tmpCount, pregMain, pregLoop, tailAddCount, tailNum, tailLoopCount, VL_F32);
            FillCountBlock(tmpCountLocal2, tmpCount, pregMain, pregLoop, lastCoreAddCount, firstNum, firstLoopCount,
                           VL_F32);
            FillCountBlock(tmpCountLocal2, tmpCount, pregMain, pregLoop, tailCoreAddCount, secondNum, secondLoopCount,
                           VL_F32);
            FillCountBlock(tmpCountLocal2, tmpCount, pregMain, pregLoop, formerCoreAddCount, thirdNum, thirdLoopCount,
                           VL_F32);
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

    __aicore__ inline void WelfordParallelUpdate(int64_t& count, LocalTensor<float>& meanTensor,
                                                 LocalTensor<float>& m2Tensor, int64_t xGmOffset, uint32_t len)
    {
        // copy in x
        LocalTensor<T> xTensor = xQueue.AllocTensor<T>();
        CopyInX(xTensor, xGmOffset);
        xQueue.EnQue(xTensor);
        xTensor = xQueue.DeQue<T>();
        // ---------
        count += 1;
        float scale = (float)1.0 / static_cast<float>(count);
        __ubuf__ float* meanTensorAddr = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* m2TensorAddr = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        __ubuf__ T* xTensorAddr = (__ubuf__ T*)xTensor.GetPhyAddr();
        uint16_t loopCount = CEIL_DIV(len, VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpM2;
            RegTensor<float> delta1;
            RegTensor<float> delta3;
            MaskReg mask0;
            uint32_t sreg0 = len;
            for (uint16_t i = 0; i < loopCount; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                // x 读 / mean 读写 / M2 读写 五处访存共用同一个偏移 i*VL_F32(单层循环 → 一维地址寄存器)。
                // stride 单位是各自元素数,x 是 T、mean/M2 是 float,故按元素宽度分别建。
                AscendC::MicroAPI::AddrReg fAddr = AscendC::MicroAPI::CreateAddrReg<float>(i, VL_F32);
                if constexpr (IsSameType<T, half>::value) {
                    RegTensor<half> xFp16;
                    AscendC::Reg::LoadAlign<half, LoadDist::DIST_UNPACK_B16>(
                        xFp16, (__ubuf__ half*)xTensorAddr, AscendC::MicroAPI::CreateAddrReg<half>(i, VL_F32));
                    Cast<float, half, NormCommon::castTraitB162B32>(x1, xFp16, mask0);
                } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                    RegTensor<bfloat16_t> xBf16;
                    AscendC::Reg::LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(
                        xBf16, (__ubuf__ bfloat16_t*)xTensorAddr,
                        AscendC::MicroAPI::CreateAddrReg<bfloat16_t>(i, VL_F32));
                    Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(x1, xBf16, mask0);
                } else {
                    AscendC::Reg::LoadAlign(x1, (__ubuf__ float*)xTensorAddr, fAddr);
                }
                AscendC::Reg::LoadAlign(tmpMean, (__ubuf__ float*)meanTensorAddr, fAddr);
                AscendC::Reg::LoadAlign(tmpM2, (__ubuf__ float*)m2TensorAddr, fAddr);
                // delta1 = x1 - mean
                Sub(delta1, x1, tmpMean, mask0);
                // mean = mean + delta1 * scale (FMA: 乘加单次舍入,省 Muls+Add 两条为一条 Axpy)
                Axpy(tmpMean, delta1, scale, mask0);
                AscendC::Reg::StoreAlign((__ubuf__ float*)meanTensorAddr, tmpMean, fAddr, mask0);
                // delta3 = x1 - mean(new)
                Sub(delta3, x1, tmpMean, mask0);
                // M2 = M2 + delta1 * delta3 (FMA: 省 Mul+Add 两条为一条 Mula)
                Mula(tmpM2, delta1, delta3, mask0);
                AscendC::Reg::StoreAlign((__ubuf__ float*)m2TensorAddr, tmpM2, fAddr, mask0);
            }
        }
        xQueue.FreeTensor(xTensor);
    }

    __aicore__ inline void ProcessWelfordFinalize(LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
                                                  LocalTensor<float>& countTensor, LocalTensor<float>& finalMeanTensor,
                                                  LocalTensor<float>& finalVarTensor, LocalTensor<float>& tmpTensor)

    {
        __ubuf__ float* tmpMeanLocal = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* tmpVarLocal = (__ubuf__ float*)m2Tensor.GetPhyAddr();
        __ubuf__ float* tmpCountLocal = (__ubuf__ float*)countTensor.GetPhyAddr();
        __ubuf__ float* batchMeanInUbAddr = (__ubuf__ float*)finalMeanTensor.GetPhyAddr();
        __ubuf__ float* batchRstdInUbAddr = (__ubuf__ float*)finalVarTensor.GetPhyAddr();
        __ubuf__ float* tmpUbAddr = (__ubuf__ float*)tmpTensor.GetPhyAddr();
        WelfordFinalizeVF<false>(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchRstdInUbAddr,
                                 tmpUbAddr);
        WelfordFinalizeVF<true>(tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchRstdInUbAddr,
                                tmpUbAddr);
    }

    // ---- finalize 的 post-increment 读 + 加权 helper(去重 Mean/Var × remainder/quotient 四处重复)----
    // 读 4 个相邻行(meanPtr 自增 stride)+ 4 个广播 count(cntPtr 自增 1),mean 加权: a = mean*count*n
    __aicore__ inline void LoadWeightMean4(__ubuf__ float*& meanPtr, __ubuf__ float*& cntPtr, int32_t stride, float n,
                                           MaskReg& preg, RegTensor<float>& a, RegTensor<float>& b, RegTensor<float>& c,
                                           RegTensor<float>& d)
    {
        RegTensor<float> w0, w1, w2, w3;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(a, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(b, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(c, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(d, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w0, cntPtr, 1);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w1, cntPtr, 1);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w2, cntPtr, 1);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w3, cntPtr, 1);
        Mul(a, a, w0, preg);
        Mul(b, b, w1, preg);
        Mul(c, c, w2, preg);
        Mul(d, d, w3, preg);
        Muls(a, a, n, preg);
        Muls(b, b, n, preg);
        Muls(c, c, n, preg);
        Muls(d, d, n, preg);
    }

    // 读 4 行 mean + 4 行 M2 + 4 个 count,var 加权: a = (M2 + count*(mean-saveMean)^2)*n
    __aicore__ inline void LoadWeightVar4(__ubuf__ float*& meanPtr, __ubuf__ float*& m2Ptr, __ubuf__ float*& cntPtr,
                                          int32_t stride, RegTensor<float>& saveMean, float n, MaskReg& preg,
                                          RegTensor<float>& a, RegTensor<float>& b, RegTensor<float>& c,
                                          RegTensor<float>& d)
    {
        RegTensor<float> q0, q1, q2, q3, w0, w1, w2, w3;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(a, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(b, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(c, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(d, meanPtr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(q0, m2Ptr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(q1, m2Ptr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(q2, m2Ptr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(q3, m2Ptr, stride);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w0, cntPtr, 1);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w1, cntPtr, 1);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w2, cntPtr, 1);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                AscendC::Reg::LoadDist::DIST_BRC_B32>(w3, cntPtr, 1);
        Sub(a, a, saveMean, preg);
        Mul(a, a, a, preg);
        Mul(a, a, w0, preg);
        Add(a, q0, a, preg);
        Muls(a, a, n, preg);
        Sub(b, b, saveMean, preg);
        Mul(b, b, b, preg);
        Mul(b, b, w1, preg);
        Add(b, q1, b, preg);
        Muls(b, b, n, preg);
        Sub(c, c, saveMean, preg);
        Mul(c, c, c, preg);
        Mul(c, c, w2, preg);
        Add(c, q2, c, preg);
        Muls(c, c, n, preg);
        Sub(d, d, saveMean, preg);
        Mul(d, d, d, preg);
        Mul(d, d, w3, preg);
        Add(d, q3, d, preg);
        Muls(d, d, n, preg);
    }

    // 组内 radix-4 二分: a = (a+b)+(c+d)
    __aicore__ inline void Radix4Add(RegTensor<float>& a, RegTensor<float>& b, RegTensor<float>& c, RegTensor<float>& d,
                                     MaskReg& preg)
    {
        Add(a, a, b, preg);
        Add(c, c, d, preg);
        Add(a, a, c, preg);
    }

    // Mean 与 Var 两趟的骨架完全相同:aIndex 外层 + "尾折组/纯主行组"两段加权归约 + BinaryAddVF 二分树 +
    // 尾部乘 nCorrection 写出。差异只有三处 —— Var 需先载入上一步算出的 batchMean 作 saveMean、每行的加权
    // 公式不同、结果写向不同的输出。故按 CALC_VAR 编译期分叉合并为一个函数(与本仓 CalculateRLessThanVF
    // <CALC_VAR, SCALE_COEF> 的惯例一致),两个实例化各自生成的代码与合并前逐条对应。
    //   CALC_VAR=false: batchMean = nCorrection * Σ_r mean[r]*count[r]*nFactor
    //   CALC_VAR=true : batchVar  = nCorrection * Σ_r (M2[r] + count[r]*(mean[r]-saveMean)^2)*nFactor
    // 寻址分工:行/count 的载入维持 post-increment(一个自增指针要连推 4/8 次访存,自增把地址推进折进
    // load 里最划算);只有每组产出一行的 bin 输出、以及 BinaryAddVF 内部,用地址寄存器(VAG)按循环计数器生成。
    template <bool CALC_VAR>
    __aicore__ inline void WelfordFinalizeVF(__ubuf__ float* tmpMeanLocal, __ubuf__ float* tmpVarLocal,
                                             __ubuf__ float* tmpCountLocal, __ubuf__ float* batchMeanInUbAddr,
                                             __ubuf__ float* batchRstdInUbAddr, __ubuf__ float* binaryAddTmpAddr)
    {
        uint16_t aLoopCount = CEIL_DIV(currentA, VL_F32);
        int32_t rLoopStride = static_cast<int32_t>(currentAAlign);
        uint16_t remainderLoopCount = (tilingData->rUbFactor - tilingData->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (tilingData->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t remainderOffset = tilingData->binaryAddQuotient * rLoopStride;
        uint32_t remainderCountOffset = tilingData->binaryAddQuotient;
        // 纯主行组的输出起点比尾折组靠后 remainderLoopCount 行,该常量并入 base
        uint32_t quotBinOffset = static_cast<uint32_t>(remainderLoopCount) * rLoopStride;
        uint16_t binaryAddInnerLoop = tilingData->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddKLoop = tilingData->binaryAddK;
        uint16_t binaryAddLastLoop = tilingData->binaryAddLast;
        float numScale = this->nFactor;
        float scaleCorrection = this->nCorrectionFactor;
        uint32_t sreg0 = currentA;
        __ubuf__ float* resultAddr = CALC_VAR ? batchRstdInUbAddr : batchMeanInUbAddr;
        __VEC_SCOPE__
        {
            RegTensor<float> saveMean;
            RegTensor<float> r0, r1, r2, r3;
            RegTensor<float> t0, t1, t2, t3;
            MaskReg pregLoop;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                if constexpr (CALC_VAR) {
                    LoadAlign(saveMean, ((__ubuf__ float*)batchMeanInUbAddr + aLoopOffset));
                }
                __ubuf__ float* quotPtr = (__ubuf__ float*)(tmpMeanLocal + aLoopOffset);
                __ubuf__ float* remPtr = (__ubuf__ float*)(tmpMeanLocal + remainderOffset + aLoopOffset);
                __ubuf__ float* quotM2Ptr = (__ubuf__ float*)(tmpVarLocal + aLoopOffset);
                __ubuf__ float* remM2Ptr = (__ubuf__ float*)(tmpVarLocal + remainderOffset + aLoopOffset);
                __ubuf__ float* quotCntPtr = (__ubuf__ float*)(tmpCountLocal);
                __ubuf__ float* remCntPtr = (__ubuf__ float*)(tmpCountLocal + remainderCountOffset);
                // 尾折组:4 主行 + 4 尾行 → 1
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    if constexpr (CALC_VAR) {
                        LoadWeightVar4(quotPtr, quotM2Ptr, quotCntPtr, rLoopStride, saveMean, numScale, pregLoop, r0,
                                       r1, r2, r3);
                        LoadWeightVar4(remPtr, remM2Ptr, remCntPtr, rLoopStride, saveMean, numScale, pregLoop, t0, t1,
                                       t2, t3);
                    } else {
                        LoadWeightMean4(quotPtr, quotCntPtr, rLoopStride, numScale, pregLoop, r0, r1, r2, r3);
                        LoadWeightMean4(remPtr, remCntPtr, rLoopStride, numScale, pregLoop, t0, t1, t2, t3);
                    }
                    Add(r0, r0, t0, pregLoop);
                    Add(r1, r1, t1, pregLoop);
                    Add(r2, r2, t2, pregLoop);
                    Add(r3, r3, t3, pregLoop);
                    Radix4Add(r0, r1, r2, r3, pregLoop);
                    AscendC::MicroAPI::AddrReg binAddr = AscendC::MicroAPI::CreateAddrReg<float>(aIndex, VL_F32, i,
                                                                                                 rLoopStride);
                    AscendC::Reg::StoreAlign((__ubuf__ float*)binaryAddTmpAddr, r0, binAddr, pregLoop);
                }
                // 纯主行组:4 主行 → 1
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    if constexpr (CALC_VAR) {
                        LoadWeightVar4(quotPtr, quotM2Ptr, quotCntPtr, rLoopStride, saveMean, numScale, pregLoop, r0,
                                       r1, r2, r3);
                    } else {
                        LoadWeightMean4(quotPtr, quotCntPtr, rLoopStride, numScale, pregLoop, r0, r1, r2, r3);
                    }
                    Radix4Add(r0, r1, r2, r3, pregLoop);
                    AscendC::MicroAPI::AddrReg binAddr = AscendC::MicroAPI::CreateAddrReg<float>(aIndex, VL_F32, i,
                                                                                                 rLoopStride);
                    AscendC::Reg::StoreAlign((__ubuf__ float*)(binaryAddTmpAddr + quotBinOffset), r0, binAddr,
                                             pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF<true>(binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                                  pregLoop, aLoopOffset, r0, r1, r2, r3, aIndex, VL_F32);
                LoadAlign(r0, ((__ubuf__ float*)binaryAddTmpAddr + aLoopOffset));
                Muls(r0, r0, scaleCorrection, pregLoop);
                StoreAlign(((__ubuf__ float*)resultAddr + aLoopOffset), r0, pregLoop);
            }
        }
    }

    __aicore__ inline void NormalizeX(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
                                      LocalTensor<T_GAMMA>& gammaTensor, LocalTensor<T_GAMMA>& betaTensor,
                                      int64_t yGmOffset)
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

    __aicore__ inline void CalcY(LocalTensor<float>& batchMeanTensor, LocalTensor<float>& batchRstdTensor,
                                 LocalTensor<T_GAMMA>& gammaTensor, LocalTensor<T_GAMMA>& betaTensor,
                                 LocalTensor<T>& xTensor, LocalTensor<T>& yTensor)
    {
        __ubuf__ float* batchMeanTensorAddr = (__ubuf__ float*)batchMeanTensor.GetPhyAddr();
        __ubuf__ float* batchRstdTensorAddr = (__ubuf__ float*)batchRstdTensor.GetPhyAddr();
        __ubuf__ T* xTensorAddr = (__ubuf__ T*)xTensor.GetPhyAddr();
        __ubuf__ T* yTensorAddr = (__ubuf__ T*)yTensor.GetPhyAddr();
        __ubuf__ T_GAMMA* gammaTensorAddr = (__ubuf__ T_GAMMA*)gammaTensor.GetPhyAddr();
        __ubuf__ T_GAMMA* betaTensorAddr = (__ubuf__ T_GAMMA*)betaTensor.GetPhyAddr();
        uint16_t numLoop = CEIL_DIV(currentA, VL_F32);
        uint16_t lineLoop = currentR;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> mean;
            RegTensor<float> rstd;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            MaskReg mask0;
            uint32_t sreg0 = currentA;
            for (uint16_t i = 0; i < numLoop; i++) {
                mask0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadAlign(mean, batchMeanTensorAddr + i * VL_F32);
                LoadAlign(rstd, batchRstdTensorAddr + i * VL_F32);
                LoadOneTensorForDtypeT(gammaTensorAddr, gamma, mask0, i * VL_F32);
                LoadOneTensorForDtypeT(betaTensorAddr, beta, mask0, i * VL_F32);
                for (uint16_t j = 0; j < lineLoop; j++) {
                    // x 读 / y 写共用同一个二维偏移 i*VL_F32 + j*currentAAlign,交给地址寄存器(VAG)按硬件循环
                    // 计数器直接生成,循环内不再有标量地址计算。三点约束:
                    //   1) CreateAddrReg 的 stride 个数必须与所在循环嵌套深度一致(此处 i/j 两层给两个 stride),
                    //      少给会编译报 "arguments of VAG intrinsic exceeds the loop nesting depth";
                    //   2) stride 单位是 T 的元素数,不是字节;
                    //   3) index 参数按位置与由外到内的循环计数器绑定,顺序不能与实际嵌套顺序对调。
                    AscendC::MicroAPI::AddrReg xyAddr = AscendC::MicroAPI::CreateAddrReg<T>(
                        i, VL_F32, j, static_cast<uint32_t>(currentAAlign));
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> xFp16;
                        AscendC::Reg::LoadAlign<half, LoadDist::DIST_UNPACK_B16>(xFp16, (__ubuf__ half*)xTensorAddr,
                                                                                 xyAddr);
                        Cast<float, half, NormCommon::castTraitB162B32>(x1, xFp16, mask0);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        AscendC::Reg::LoadAlign<bfloat16_t, LoadDist::DIST_UNPACK_B16>(
                            xBf16, (__ubuf__ bfloat16_t*)xTensorAddr, xyAddr);
                        Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(x1, xBf16, mask0);
                    } else {
                        AscendC::Reg::LoadAlign(x1, (__ubuf__ float*)xTensorAddr, xyAddr);
                    }
                    Sub(x1, x1, mean, mask0);
                    Mul(x1, x1, rstd, mask0);
                    // 保持原分组 ((x-mean)*rstd)*gamma+beta,仅把最后的乘加融成一条 FusedMulDstAdd(dst=dst*gamma+beta):
                    // gamma 仍乘在 (x-mean)*rstd 上,gamma=0 且该值为 inf 时融合乘内部 0*inf 照样得 nan,inf/nan 行为不变
                    MulDstAdd(x1, gamma, beta, mask0);
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, NormCommon::castTraitB322B16>(yFp16, x1, mask0);
                        AscendC::Reg::StoreAlign<half, StoreDist::DIST_PACK_B32>((__ubuf__ half*)yTensorAddr, yFp16,
                                                                                 xyAddr, mask0);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, NormCommon::castTraitB322B16>(xBf16, x1, mask0);
                        AscendC::Reg::StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            (__ubuf__ bfloat16_t*)yTensorAddr, xBf16, xyAddr, mask0);
                    } else {
                        AscendC::Reg::StoreAlign((__ubuf__ float*)yTensorAddr, x1, xyAddr, mask0);
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
    int64_t gammaBetaHalf = 0;     // gamma/beta 合并 que 中 beta 相对 gamma 的对齐偏移(元素数)
    int64_t batchMeanRstdHalf = 0; // batchMean/batchRstd 合并 que 中 rstd 相对 mean 的对齐偏移(元素数)
    int64_t allMeanVarHalf = 0;    // allMean/allVar 合并 que 中 var 相对 mean 的对齐偏移(元素数)
    int64_t runningHalf = 0;       // running mean/var 合并 que 中 var 相对 mean 的对齐偏移(元素数)
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
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> gammaBetaQueue;
    TQue<QuePosition::VECIN, 1> runningMeanVarInQueue;
    TQue<QuePosition::VECIN, 1> allMeanVarQueue;

    TQue<QuePosition::VECOUT, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanRstdQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanVarOutQueue;

    TBuf<TPosition::VECCALC> tmpTbuf1;
    TBuf<TPosition::VECCALC> tmpTbuf2;
    TBuf<TPosition::VECCALC> tmpTbuf3;
    TBuf<TPosition::VECCALC> countTbuf1;
    TBuf<TPosition::VECCALC> countTbuf2;
};
} // namespace BatchNormV3Ops
#endif
