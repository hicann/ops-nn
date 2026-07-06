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
 * \file batch_norm_v3_ra_welford.h
 * \brief
 */
#ifndef BATCH_NORM_V3_RA_WELFORD_REGBASE_H
#define BATCH_NORM_V3_RA_WELFORD_REGBASE_H

#include "batch_norm_v3_regbase_common.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

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

template <typename T, typename T_BETA, typename T_RUNNING_MEAN>
class BatchNormV3RAWelford {
public:
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline BatchNormV3RAWelford(const BatchNormV3RAWelfordTilingData* tilingData)
    {
        this->r = tilingData->r;
        this->rFactor = tilingData->rFactor;
        this->a = tilingData->a;

        this->blockNum = tilingData->blockNum;
        this->aBlockFactor = tilingData->aBlockFactor;
        this->aFactor = tilingData->aFactor;

        this->binaryAddQuotient = tilingData->binaryAddQuotient;
        this->binaryAddK = tilingData->binaryAddK;
        this->binaryAddLast = tilingData->binaryAddLast;

        this->epsilon = tilingData->epsilon;
        this->momentum = tilingData->momentum;

        float one = 1.0;
        this->oneSubMomentum = one - this->momentum;
        this->besselCorrectionFactor = this->r == 1 ? AscendC::NumericLimits<float>::QuietNaN() :
                                                      (static_cast<float>(this->r) / static_cast<float>(this->r - 1));

        int64_t powerOfTwoForR = tilingData->powerOfTwoForR;
        this->nFactor = static_cast<float>(1) / static_cast<float>(powerOfTwoForR);
        this->nCorrectionFactor = static_cast<float>(powerOfTwoForR) / static_cast<float>(this->r);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR var, GM_ADDR y,
                                GM_ADDR mean_out, GM_ADDR var_out, GM_ADDR batch_mean, GM_ADDR batch_rstd)
    {
        auto blockIdx = GetBlockIdx();

        this->singleA = (blockIdx == this->blockNum - 1) ? (this->a - this->aBlockFactor * (this->blockNum - 1)) :
                                                           this->aBlockFactor;
        int64_t aGmOffset = this->aBlockFactor * blockIdx;
        xGm.SetGlobalBuffer((__gm__ T*)x + aGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T_BETA*)beta + aGmOffset);
        gammaGm.SetGlobalBuffer((__gm__ T_BETA*)gamma + aGmOffset);
        runningMeanGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean + aGmOffset);
        runningVarGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var + aGmOffset);

        yGm.SetGlobalBuffer((__gm__ T*)y + aGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean + aGmOffset);
        batchRstdGm.SetGlobalBuffer((__gm__ float*)batch_rstd + aGmOffset);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean_out + aGmOffset);
        runningVarOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var_out + aGmOffset);

        pipe.InitBuffer(xQueue, DOUBLE_BUFFER, this->rFactor * this->aFactor * sizeof(T));
        pipe.InitBuffer(yQueue, DOUBLE_BUFFER, this->rFactor * this->aFactor * sizeof(T));

        // 可能T为fp16, T_BETA为float的场景，aFactor对齐值为fp16的对齐值，betaQueue按照fp16的对齐值申请block对齐的ub
        pipe.InitBuffer(betaQueue, 1, this->aFactor * sizeof(T_BETA));
        pipe.InitBuffer(gammaQueue, 1, this->aFactor * sizeof(T_BETA));

        pipe.InitBuffer(batchMeanQueue, 1, this->aFactor * sizeof(float));
        pipe.InitBuffer(batchRstdQueue, 1, this->aFactor * sizeof(float));

        pipe.InitBuffer(runningMeanInQueue, 1, this->aFactor * sizeof(float));
        pipe.InitBuffer(runningVarInQueue, 1, this->aFactor * sizeof(float));
        pipe.InitBuffer(runningMeanOutQueue, 1, this->aFactor * sizeof(float));
        pipe.InitBuffer(runningVarOutQueue, 1, this->aFactor * sizeof(float));

        pipe.InitBuffer(tMeanBuff, this->rFactor * this->aFactor * sizeof(float));
        pipe.InitBuffer(tVarBuff, this->rFactor * this->aFactor * sizeof(float));
        int64_t rFactorAlign = (((this->rFactor * sizeof(float) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) /
                               sizeof(float);
        pipe.InitBuffer(tCountBuff, rFactorAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CaculateCountBuf();
        int64_t quotient = (this->singleA + this->aFactor - 1) / this->aFactor;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
            int64_t aOffset = ubLoopIdx * this->aFactor;
            int64_t currentA = (ubLoopIdx == (quotient - 1)) ? (this->singleA - (quotient - 1) * this->aFactor) :
                                                               this->aFactor;
            currentANumAlign = (((currentA * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T);
            currentALoopCount = CEIL_DIV(currentA, VL_F32);
            ProcessAInner(aOffset, currentA);
        }
    }

private:
    __aicore__ inline void CaculateCountBuf()
    {
        LocalTensor<float> tCountTensor = tCountBuff.Get<float>();
        __local_mem__ float* tmpCountLocal = (__local_mem__ float*)tCountTensor.GetPhyAddr();
        int64_t parallelCount = this->r / this->rFactor;
        int64_t parallelReminder = this->r % this->rFactor;
        float quotientAddCount = static_cast<float>(parallelCount);
        float remaninderAddCount = static_cast<float>(parallelCount + 1);
        uint16_t quotientLoopCount = CEIL_DIV(this->rFactor, VL_F32);
        uint16_t remainderLoopCount = CEIL_DIV(parallelReminder, VL_F32);
        uint32_t quotientNum = this->rFactor;
        uint32_t remainderNum = parallelReminder;

        __VEC_SCOPE__
        {
            RegTensor<float> tmpCount;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg1 = quotientNum;
            Duplicate(tmpCount, quotientAddCount, pregMain);
            for (uint16_t i = 0; i < quotientLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg1);
                DataCopy(((__local_mem__ float*)tmpCountLocal + i * VL_F32), tmpCount, pregLoop);
            }
            uint32_t sreg2 = remainderNum;
            Duplicate(tmpCount, remaninderAddCount, pregMain);
            for (uint16_t i = 0; i < remainderLoopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg2);
                DataCopy(((__local_mem__ float*)tmpCountLocal + i * VL_F32), tmpCount, pregLoop);
            }
        }
    }

    __aicore__ inline void ProcessAInner(int64_t aOffset, int64_t currentANum)
    {
        LocalTensor<float> tMeanTensor = tMeanBuff.Get<float>();
        LocalTensor<float> tVarTensor = tVarBuff.Get<float>();
        LocalTensor<float> tCountTensor = tCountBuff.Get<float>();
        __local_mem__ float* tmpMeanLocal = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* tmpVarLocal = (__local_mem__ float*)tVarTensor.GetPhyAddr();
        __local_mem__ float* tmpCountLocal = (__local_mem__ float*)tCountTensor.GetPhyAddr();

        ProcessWelfordUpdate(aOffset, currentANum, tmpMeanLocal, tmpVarLocal);
        CopyInRunningMeanVar(aOffset, currentANum);

        LocalTensor<float> batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchRstdOutUb = batchRstdQueue.AllocTensor<float>();
        __local_mem__ float* batchMeanInUbAddr = (__local_mem__ float*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ float* batchRstdInUbAddr = (__local_mem__ float*)batchRstdOutUb.GetPhyAddr();
        ProcessWelfordFinalize(aOffset, currentANum, tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr,
                               batchRstdInUbAddr);
        CalculateRunningMeanVar(aOffset, currentANum, batchMeanInUbAddr, batchRstdInUbAddr);
        batchMeanQueue.EnQue(batchMeanOutUb);
        batchRstdQueue.EnQue(batchRstdOutUb);
        CopyOutRunningMeanVar(aOffset, currentANum);
        Normalize(aOffset, currentANum, batchMeanInUbAddr, batchRstdInUbAddr);
        CopyOutSaveMeanVar(aOffset, currentANum);
    }

    __aicore__ inline void ProcessWelfordUpdate(int64_t aOffset, int64_t currentANum, __local_mem__ float* tmpMeanLocal,
                                                __local_mem__ float* tmpVarLocal)
    {
        int64_t quotient = (this->r + this->rFactor - 1) / this->rFactor;
        for (int64_t rLoopIdx = 0; rLoopIdx < quotient; rLoopIdx++) {
            int64_t copyXOffset = rLoopIdx * this->rFactor * this->a + aOffset;
            int64_t currentR = (rLoopIdx == (quotient - 1)) ? (this->r - (quotient - 1) * this->rFactor) :
                                                              this->rFactor;

            CopyInX(copyXOffset, currentR, currentANum);

            LocalTensor<T> xInUb = xQueue.DeQue<T>();
            __local_mem__ T* xLocal = (__local_mem__ T*)xInUb.GetPhyAddr();
            // process welford after copy ubSize data into ub.
            float scale = (float)1.0 / static_cast<float>(rLoopIdx + 1);
            uint64_t processNum = currentR * currentANumAlign;
            uint16_t updateLoopCount = CEIL_DIV(processNum, VL_F32);
            if (rLoopIdx == 0) {
                // 第一次更新时，需要将tmp mean和tmp var清0
                BatchNormV3Ops::WelfordParallelUpdateVF<true>(xLocal, tmpMeanLocal, tmpVarLocal, processNum,
                                                              updateLoopCount, scale, VL_F32);
            } else {
                BatchNormV3Ops::WelfordParallelUpdateVF<false>(xLocal, tmpMeanLocal, tmpVarLocal, processNum,
                                                               updateLoopCount, scale, VL_F32);
            }
            xQueue.FreeTensor(xInUb);
        }
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentRNum, int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue.AllocTensor<T>();
        if (currentANum * sizeof(T) <= NDDMA_THRESHOLD) {
            T constValue = 0;
            static constexpr MultiCopyConfig config = {false};

            MultiCopyLoopInfo<NDDMA_DIM_NUM> loopInfo;
            loopInfo.loopSize[0] = currentANum;
            loopInfo.loopSrcStride[0] = 1;
            loopInfo.loopDstStride[0] = 1;
            loopInfo.loopLpSize[0] = 0;
            loopInfo.loopRpSize[0] = 0;

            loopInfo.loopSize[1] = currentRNum;
            loopInfo.loopSrcStride[1] = this->a;
            loopInfo.loopDstStride[1] = currentANumAlign;
            loopInfo.loopLpSize[1] = 0;
            loopInfo.loopRpSize[1] = 0;
            MultiCopyParams<T, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
            DataCopy<T, NDDMA_DIM_NUM, config>(xInUb, xGm[offset], paramsMain);
        } else {
            DataCopyPadExtParams<T> dataCopyPadExtParams;
            dataCopyPadExtParams.isPad = (currentANum != currentANumAlign);
            dataCopyPadExtParams.leftPadding = 0;
            // isPad配置True，rightPadding配置0，表示自动Pad到32B对齐
            dataCopyPadExtParams.rightPadding = 0;
            dataCopyPadExtParams.paddingValue = 0;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = currentRNum;
            copyInParams.blockLen = currentANum * sizeof(T);
            copyInParams.srcStride = (this->a - currentANum) * sizeof(T);
            copyInParams.dstStride = 0;
            DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
        }
        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void CopyInRunningMeanVar(int64_t aOffset, int64_t currentANum)
    {
        LocalTensor<T_BETA> betaInUb = betaQueue.AllocTensor<T_BETA>();
        LocalTensor<T_BETA> gammaInUb = gammaQueue.AllocTensor<T_BETA>();
        DataCopyPadExtParams<T_BETA> dataCopyPadExtParamsT;
        dataCopyPadExtParamsT.leftPadding = 0;
        dataCopyPadExtParamsT.isPad = false;
        dataCopyPadExtParamsT.rightPadding = 0;
        dataCopyPadExtParamsT.paddingValue = 0;
        DataCopyExtParams copyInParamsT;
        copyInParamsT.blockLen = currentANum * sizeof(T_BETA);
        copyInParamsT.blockCount = 1;
        copyInParamsT.srcStride = 0;
        copyInParamsT.dstStride = 0;
        DataCopyPad(betaInUb, betaGm[aOffset], copyInParamsT, dataCopyPadExtParamsT);
        DataCopyPad(gammaInUb, gammaGm[aOffset], copyInParamsT, dataCopyPadExtParamsT);
        betaQueue.EnQue(betaInUb);
        gammaQueue.EnQue(gammaInUb);

        LocalTensor<T_RUNNING_MEAN> runningMeanInUb = runningMeanInQueue.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarInUb = runningVarInQueue.AllocTensor<T_RUNNING_MEAN>();
        DataCopyPadExtParams<T_RUNNING_MEAN> dataCopyPadExtParams;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockLen = currentANum * sizeof(T_RUNNING_MEAN);
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanInUb, runningMeanGm[aOffset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(runningVarInUb, runningVarGm[aOffset], copyInParams, dataCopyPadExtParams);
        runningMeanInQueue.EnQue(runningMeanInUb);
        runningVarInQueue.EnQue(runningVarInUb);
    }

    __aicore__ inline void ProcessWelfordFinalize(int64_t aOffset, int64_t currentANum,
                                                  __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
                                                  __local_mem__ float* tmpCountLocal,
                                                  __local_mem__ float* batchMeanInUbAddr,
                                                  __local_mem__ float* batchRstdInUbAddr)
    {
        LocalTensor<T> yInUb = yQueue.AllocTensor<T>();
        __local_mem__ float* yInUbAddr = (__local_mem__ float*)yInUb.GetPhyAddr();
        WelfordFinalizeMeanVF(aOffset, currentANum, tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr,
                              batchRstdInUbAddr, yInUbAddr);
        WelfordFinalizeVarVF(aOffset, currentANum, tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr,
                             batchRstdInUbAddr, yInUbAddr);
        yQueue.FreeTensor(yInUb);
    }

    __aicore__ inline void WelfordFinalizeMeanVF(int64_t aOffset, int64_t currentANum,
                                                 __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
                                                 __local_mem__ float* tmpCountLocal,
                                                 __local_mem__ float* batchMeanInUbAddr,
                                                 __local_mem__ float* batchRstdInUbAddr,
                                                 __local_mem__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = this->rFactor;
        uint16_t aLoopCount = this->currentALoopCount;
        uint32_t rLoopStride = currentANumAlign;

        uint16_t remainderLoopCount = (this->rFactor - this->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (this->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = this->binaryAddQuotient * currentANumAlign;
        uint32_t remainderCountOffset = this->binaryAddQuotient;

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddInnerLoop = this->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = this->binaryAddLast;

        uint16_t finalLoopCount = this->binaryAddQuotient / SCALE_COEF_FOUR;
        float numScale = this->nFactor;
        float scaleCorrection = this->nCorrectionFactor;

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        uint32_t fourRLoopSize = ROW_FOUR_OFFSET * rLoopStride;
        uint32_t fiveRLoopSize = ROW_FIVE_OFFSET * rLoopStride;
        uint32_t sixRLoopSize = ROW_SIX_OFFSET * rLoopStride;
        uint32_t sevenRLoopSize = ROW_SEVEN_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> raMeanAcc0;
            RegTensor<float> raMeanAcc1;
            RegTensor<float> raBinaryTmp0;
            RegTensor<float> raBinaryTmp1;

            RegTensor<float> raNextRow;
            RegTensor<float> raRem;
            RegTensor<float> raRemNextRow;

            RegTensor<float> raRowCount;
            RegTensor<float> raNextRowCount;
            RegTensor<float> raRemCount;
            RegTensor<float> raNextRemCount;

            MaskReg pregLoop;
            uint32_t sreg0 = currentANum;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_FOUR;
                    uint32_t remCountOffset = i * SCALE_COEF_FOUR + remainderCountOffset;
                    TwoRowAddPartialMeanWithTail(raMeanAcc0, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset,
                                                 remOffset, quotOffset + rLoopStride, remOffset + rLoopStride,
                                                 quotCountOffset, remCountOffset, quotCountOffset + 1,
                                                 remCountOffset + 1, raRem, raNextRow, raRemNextRow, raRowCount,
                                                 raNextRowCount, raRemCount, raNextRemCount, numScale);
                    TwoRowAddPartialMeanWithTail(
                        raMeanAcc1, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset + twoRLoopSize,
                        remOffset + twoRLoopSize, quotOffset + threeRLoopSize, remOffset + threeRLoopSize,
                        quotCountOffset + ROW_TWO_OFFSET, remCountOffset + ROW_TWO_OFFSET,
                        quotCountOffset + ROW_THREE_OFFSET, remCountOffset + ROW_THREE_OFFSET, raRem, raNextRow,
                        raRemNextRow, raRowCount, raNextRowCount, raRemCount, raNextRemCount, numScale);
                    Add(raMeanAcc0, raMeanAcc0, raMeanAcc1, pregLoop);
                    DataCopy(((__local_mem__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), raMeanAcc0,
                             pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddPartialMean(raMeanAcc0, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset,
                                         baseOffset + rLoopStride, baseCountOffset, baseCountOffset + 1, raRem,
                                         raRowCount, raNextRowCount, numScale);
                    TwoRowAddPartialMean(raMeanAcc1, tmpMeanLocal, tmpCountLocal, pregLoop, baseOffset + twoRLoopSize,
                                         baseOffset + threeRLoopSize, baseCountOffset + ROW_TWO_OFFSET,
                                         baseCountOffset + ROW_THREE_OFFSET, raRem, raRowCount, raNextRowCount,
                                         numScale);
                    Add(raMeanAcc0, raMeanAcc0, raMeanAcc1, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                        raMeanAcc0, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF(binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                            pregLoop, aLoopOffset, raMeanAcc0, raMeanAcc1, raBinaryTmp0, raBinaryTmp1);
                DataCopy(raMeanAcc0, ((__local_mem__ float*)binaryAddTmpAddr + aLoopOffset));
                Muls(raMeanAcc0, raMeanAcc0, scaleCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset), raMeanAcc0, pregLoop);
            }
        }
    }

    __aicore__ inline void WelfordFinalizeVarVF(int64_t aOffset, int64_t currentANum, __local_mem__ float* tmpMeanLocal,
                                                __local_mem__ float* tmpVarLocal, __local_mem__ float* tmpCountLocal,
                                                __local_mem__ float* batchMeanInUbAddr,
                                                __local_mem__ float* batchRstdInUbAddr,
                                                __local_mem__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = this->rFactor;
        uint16_t aLoopCount = this->currentALoopCount;
        uint32_t rLoopStride = currentANumAlign;

        uint16_t remainderLoopCount = (this->rFactor - this->binaryAddQuotient) / SCALE_COEF_FOUR;
        uint16_t quotientLoopCount = (this->binaryAddQuotient / SCALE_COEF_FOUR) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_FOUR * rLoopStride;
        uint32_t remainderOffset = this->binaryAddQuotient * currentANumAlign;
        uint32_t remainderCountOffset = this->binaryAddQuotient;

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddInnerLoop = this->binaryAddQuotient / SCALE_COEF_FOUR;
        uint16_t binaryAddLastLoop = this->binaryAddLast;

        uint16_t finalLoopCount = this->binaryAddQuotient / SCALE_COEF_FOUR;
        float numScale = (float)1.0 / static_cast<float>(this->r);

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        uint32_t fourRLoopSize = ROW_FOUR_OFFSET * rLoopStride;
        uint32_t fiveRLoopSize = ROW_FIVE_OFFSET * rLoopStride;
        uint32_t sixRLoopSize = ROW_SIX_OFFSET * rLoopStride;
        uint32_t sevenRLoopSize = ROW_SEVEN_OFFSET * rLoopStride;
        __VEC_SCOPE__
        {
            RegTensor<float> saveMean;

            RegTensor<float> raVarAcc0;
            RegTensor<float> raVarAcc1;
            RegTensor<float> raVarBinaryTmp0;
            RegTensor<float> raVarBinaryTmp1;

            RegTensor<float> raVarNextRow;
            RegTensor<float> raVarRem;
            RegTensor<float> raVarRemNextRow;

            RegTensor<float> raVarRowCount;
            RegTensor<float> raVarNextRowCount;
            RegTensor<float> raVarRemCount;
            RegTensor<float> raVarNextRemCount;

            RegTensor<float> rowM2;
            RegTensor<float> nextRowM2;
            RegTensor<float> remM2;
            RegTensor<float> nextRemM2;

            MaskReg pregLoop;
            uint32_t sreg0 = currentANum;
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
                        raVarAcc0, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, quotOffset, remOffset,
                        quotOffset + rLoopStride, remOffset + rLoopStride, quotCountOffset, remCountOffset,
                        quotCountOffset + 1, remCountOffset + 1, saveMean, raVarRem, raVarNextRow, raVarRemNextRow,
                        raVarRowCount, raVarNextRowCount, raVarRemCount, raVarNextRemCount, rowM2, nextRowM2, remM2,
                        nextRemM2, numScale);
                    TwoRowAddPartialVarWithTail(
                        raVarAcc1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, quotOffset + twoRLoopSize,
                        remOffset + twoRLoopSize, quotOffset + threeRLoopSize, remOffset + threeRLoopSize,
                        quotCountOffset + ROW_TWO_OFFSET, remCountOffset + ROW_TWO_OFFSET,
                        quotCountOffset + ROW_THREE_OFFSET, remCountOffset + ROW_THREE_OFFSET, saveMean, raVarRem,
                        raVarNextRow, raVarRemNextRow, raVarRowCount, raVarNextRowCount, raVarRemCount,
                        raVarNextRemCount, rowM2, nextRowM2, remM2, nextRemM2, numScale);
                    Add(raVarAcc0, raVarAcc0, raVarAcc1, pregLoop);
                    DataCopy(((__local_mem__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), raVarAcc0,
                             pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_FOUR;
                    TwoRowAddPartialVar(raVarAcc0, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop, baseOffset,
                                        baseOffset + rLoopStride, baseCountOffset, baseCountOffset + 1, saveMean,
                                        raVarRem, raVarRowCount, raVarNextRowCount, rowM2, remM2, numScale);
                    TwoRowAddPartialVar(raVarAcc1, tmpMeanLocal, tmpVarLocal, tmpCountLocal, pregLoop,
                                        baseOffset + twoRLoopSize, baseOffset + threeRLoopSize,
                                        baseCountOffset + ROW_TWO_OFFSET, baseCountOffset + ROW_THREE_OFFSET, saveMean,
                                        raVarRem, raVarRowCount, raVarNextRowCount, rowM2, remM2, numScale);
                    Add(raVarAcc0, raVarAcc0, raVarAcc1, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)binaryAddTmpAddr + (remainderLoopCount + i) * rLoopStride + aLoopOffset),
                        raVarAcc0, pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF(binaryAddTmpAddr, rLoopStride, binaryAddKLoop, binaryAddInnerLoop, binaryAddLastLoop,
                            pregLoop, aLoopOffset, raVarAcc0, raVarAcc1, raVarBinaryTmp0, raVarBinaryTmp1);
                DataCopy(raVarAcc0, ((__local_mem__ float*)binaryAddTmpAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)batchRstdInUbAddr + aLoopOffset), raVarAcc0, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateRunningMeanVar(int64_t aOffset, int64_t currentANum,
                                                   __local_mem__ float* batchMeanInUbAddr,
                                                   __local_mem__ float* batchRstdInUbAddr)
    {
        uint16_t aLoop = currentALoopCount;
        LocalTensor<T_RUNNING_MEAN> runningMeanInUb = runningMeanInQueue.template DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarInUb = runningVarInQueue.template DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningMeanOutUb = runningMeanOutQueue.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarOutUb = runningVarOutQueue.AllocTensor<T_RUNNING_MEAN>();

        __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningMeanInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr = (__local_mem__ T_RUNNING_MEAN*)
                                                                 runningMeanOutUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarOutUb.GetPhyAddr();

        CalculateRunningMeanVarWithRstdVF<T_RUNNING_MEAN>(
            batchMeanInUbAddr, batchRstdInUbAddr, runningMeanInUbAddr, runningVarInUbAddr, runningMeanOutUbAddr,
            runningVarOutUbAddr, static_cast<uint16_t>(currentANum), aLoop, VL_F32, this->besselCorrectionFactor,
            this->momentum, this->oneSubMomentum, this->epsilon);

        runningMeanInQueue.FreeTensor(runningMeanInUb);
        runningVarInQueue.FreeTensor(runningVarInUb);
        runningMeanOutQueue.EnQue(runningMeanOutUb);
        runningVarOutQueue.EnQue(runningVarOutUb);
    }

    __aicore__ inline void Normalize(int64_t aOffset, int64_t currentANum, __local_mem__ float* batchMeanInUbAddr,
                                     __local_mem__ float* batchRstdInUbAddr)
    {
        LocalTensor<T_BETA> betaInUb = betaQueue.template DeQue<T_BETA>();
        LocalTensor<T_BETA> gammaInUb = gammaQueue.template DeQue<T_BETA>();
        __local_mem__ T_BETA* betaInUbAddr = (__local_mem__ T_BETA*)betaInUb.GetPhyAddr();
        __local_mem__ T_BETA* gammaInUbAddr = (__local_mem__ T_BETA*)gammaInUb.GetPhyAddr();
        int64_t quotient = (this->r + this->rFactor - 1) / this->rFactor;
        for (int64_t rLoopIdx = 0; rLoopIdx < quotient; rLoopIdx++) {
            int64_t copyXOffset = rLoopIdx * this->rFactor * this->a + aOffset;
            int64_t currentR = (rLoopIdx == (quotient - 1)) ? (this->r - (quotient - 1) * this->rFactor) :
                                                              this->rFactor;

            CopyInX(copyXOffset, currentR, currentANum);
            NormalizeVF(currentR, currentANum, batchMeanInUbAddr, batchRstdInUbAddr, betaInUbAddr, gammaInUbAddr);
            CopyOutY(copyXOffset, currentR, currentANum);
        }
        betaQueue.FreeTensor(betaInUb);
        gammaQueue.FreeTensor(gammaInUb);
    }

    __aicore__ inline void NormalizeVF(int64_t currentR, int64_t currentANum, __local_mem__ float* batchMeanInUbAddr,
                                       __local_mem__ float* batchRstdInUbAddr, __local_mem__ T_BETA* betaInUbAddr,
                                       __local_mem__ T_BETA* gammaInUbAddr)
    {
        LocalTensor<T> xInUb = xQueue.DeQue<T>();
        LocalTensor<T> yInUb = yQueue.AllocTensor<T>();
        __local_mem__ T* xInUbAddr = (__local_mem__ T*)xInUb.GetPhyAddr();
        __local_mem__ T* yInUbAddr = (__local_mem__ T*)yInUb.GetPhyAddr();

        uint16_t rLoopCount = currentR;
        uint16_t aLoopCount = currentALoopCount;
        uint32_t rLoopStride = currentANumAlign;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;
            RegTensor<float> rstd;

            RegTensor<float> gamma;
            RegTensor<float> beta;

            RegTensor<float> x2;
            RegTensor<float> y2;

            MaskReg pregLoop;
            uint32_t sreg = currentANum;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = UpdateMask<float>(sreg);

                LoadOneTensorForDtypeT(betaInUbAddr, beta, pregLoop, aLoopOffset);
                LoadOneTensorForDtypeT(gammaInUbAddr, gamma, pregLoop, aLoopOffset);
                DataCopy(mean, (__local_mem__ float*)batchMeanInUbAddr + aLoopOffset);
                DataCopy(rstd, (__local_mem__ float*)batchRstdInUbAddr + aLoopOffset);
                for (uint16_t rIndex = 0; rIndex < rLoopCount; rIndex++) {
                    LoadOneTensorForDtypeT(xInUbAddr, x2, pregLoop, rIndex * rLoopStride + aLoopOffset);
                    Sub(x2, x2, mean, pregLoop);
                    Mul(y2, x2, rstd, pregLoop);
                    Mul(y2, y2, beta, pregLoop);
                    Add(y2, y2, gamma, pregLoop);
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, NormCommon::castTraitB322B16>(yFp16, y2, pregLoop);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ half*)yInUbAddr + rIndex * rLoopStride + aLoopOffset), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, NormCommon::castTraitB322B16>(xBf16, y2, pregLoop);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ bfloat16_t*)yInUbAddr + rIndex * rLoopStride + aLoopOffset), xBf16,
                            pregLoop);
                    } else {
                        DataCopy(((__local_mem__ float*)yInUbAddr + rIndex * rLoopStride + aLoopOffset), y2, pregLoop);
                    }
                }
            }
        }
        yQueue.EnQue(yInUb);
        xQueue.FreeTensor(xInUb);
    }

    __aicore__ inline void CopyOutRunningMeanVar(int64_t aOffset, int64_t currentANum)
    {
        LocalTensor<T_RUNNING_MEAN> runningMeanOutUb = runningMeanOutQueue.template DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarOutUb = runningVarOutQueue.template DeQue<T_RUNNING_MEAN>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T_RUNNING_MEAN);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanOutGm[aOffset], runningMeanOutUb, copyInParams);
        DataCopyPad(runningVarOutGm[aOffset], runningVarOutUb, copyInParams);
        runningMeanOutQueue.FreeTensor(runningMeanOutUb);
        runningVarOutQueue.FreeTensor(runningVarOutUb);
    }

    __aicore__ inline void CopyOutSaveMeanVar(int64_t aOffset, int64_t currentANum)
    {
        LocalTensor<float> batchMeanInUb = batchMeanQueue.template DeQue<float>();
        LocalTensor<float> batchRstdInUb = batchRstdQueue.template DeQue<float>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(batchMeanGm[aOffset], batchMeanInUb, copyInParams);
        DataCopyPad(batchRstdGm[aOffset], batchRstdInUb, copyInParams);
        batchMeanQueue.FreeTensor(batchMeanInUb);
        batchRstdQueue.FreeTensor(batchRstdInUb);
    }

    __aicore__ inline void CopyOutY(int64_t offset, int64_t currentRNum, int64_t currentANum)
    {
        LocalTensor<T> yOutUb = yQueue.template DeQue<T>();

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = currentRNum;
        copyInParams.blockLen = currentANum * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->a - currentANum) * sizeof(T);
        DataCopyPad(yGm[offset], yOutUb, copyInParams);
        yQueue.FreeTensor(yOutUb);
    }

    /* global memory address */
    GlobalTensor<T> xGm;
    GlobalTensor<T_BETA> betaGm;
    GlobalTensor<T_BETA> gammaGm;
    GlobalTensor<T_RUNNING_MEAN> runningMeanGm;
    GlobalTensor<T_RUNNING_MEAN> runningVarGm;

    GlobalTensor<T> yGm;
    GlobalTensor<float> batchMeanGm;
    GlobalTensor<float> batchRstdGm;
    GlobalTensor<T_RUNNING_MEAN> runningMeanOutGm;
    GlobalTensor<T_RUNNING_MEAN> runningVarOutGm;

    /* variable */
    int64_t r;
    int64_t rFactor;
    int64_t aFactor;
    int64_t a;

    int64_t blockNum;
    int64_t aBlockFactor;
    int64_t singleA;
    int64_t currentANumAlign;
    int64_t currentALoopCount;

    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLast;

    static constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);
    static constexpr int64_t NDDMA_THRESHOLD = 32;
    static constexpr int64_t BLOCK_SIZE = 32;
    static constexpr int64_t DOUBLE_BUFFER = 2;
    static constexpr int64_t SCALE_COEF_FOUR = 4;
    constexpr static int64_t NDDMA_DIM_NUM = 2;

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

    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQueue;
    TQue<QuePosition::VECIN, 1> betaQueue;
    TQue<QuePosition::VECIN, 1> gammaQueue;
    TQue<QuePosition::VECIN, 1> runningMeanInQueue;
    TQue<QuePosition::VECIN, 1> runningVarInQueue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue;
    TQue<QuePosition::VECOUT, 1> batchMeanQueue;
    TQue<QuePosition::VECOUT, 1> batchRstdQueue;
    TQue<QuePosition::VECOUT, 1> runningMeanOutQueue;
    TQue<QuePosition::VECOUT, 1> runningVarOutQueue;

    TBuf<TPosition::VECCALC> tMeanBuff;
    TBuf<TPosition::VECCALC> tVarBuff;
    TBuf<TPosition::VECCALC> tCountBuff;
};
} // namespace BatchNormV3Ops
#endif
