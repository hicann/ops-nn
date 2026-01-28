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
        this->besselCorrectionFactor = (static_cast<float>(this->r) / static_cast<float>(this->r - 1));

        int64_t powerOfTwoForR = tilingData->powerOfTwoForR;
        this->nFactor = static_cast<float>(1) / static_cast<float>(powerOfTwoForR);
        this->nCorrectionFactor = static_cast<float>(powerOfTwoForR) / static_cast<float>(this->r);
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR var, GM_ADDR y, GM_ADDR mean_out, GM_ADDR var_out,
        GM_ADDR batch_mean, GM_ADDR batch_rstd)
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
        int64_t rFactorAlign =
            (((this->rFactor * sizeof(float) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(float);
        pipe.InitBuffer(tCountBuff, rFactorAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CaculateCountBuf();
        int64_t quotient = (this->singleA + this->aFactor - 1) / this->aFactor;
        for (int64_t ubLoopIdx = 0; ubLoopIdx < quotient; ubLoopIdx++) {
            int64_t aOffset = ubLoopIdx * this->aFactor;
            int64_t currentA =
                (ubLoopIdx == (quotient - 1)) ? (this->singleA - (quotient - 1) * this->aFactor) : this->aFactor;
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

        ProcessWelfordUpdate(aOffset, currentANum, tmpMeanLocal, tmpVarLocal, tmpCountLocal);
        CopyInRunningMeanVar(aOffset, currentANum);

        LocalTensor<float> batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchRstdOutUb = batchRstdQueue.AllocTensor<float>();
        __local_mem__ float* batchMeanInUbAddr = (__local_mem__ float*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ float* batchRstdInUbAddr = (__local_mem__ float*)batchRstdOutUb.GetPhyAddr();
        ProcessWelfordFinalize(
            aOffset, currentANum, tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchRstdInUbAddr);
        CalculateRunningMeanVar(aOffset, currentANum, batchMeanInUbAddr, batchRstdInUbAddr);
        batchMeanQueue.EnQue(batchMeanOutUb);
        batchRstdQueue.EnQue(batchRstdOutUb);
        CopyOutRunningMeanVar(aOffset, currentANum);
        Normalize(aOffset, currentANum, batchMeanInUbAddr, batchRstdInUbAddr);
        CopyOutSaveMeanVar(aOffset, currentANum);
    }

    __aicore__ inline void ProcessWelfordUpdate(
        int64_t aOffset, int64_t currentANum, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        __local_mem__ float* tmpCountLocal)
    {
        int64_t quotient = (this->r + this->rFactor - 1) / this->rFactor;
        for (int64_t rLoopIdx = 0; rLoopIdx < quotient; rLoopIdx++) {
            int64_t copyXOffset = rLoopIdx * this->rFactor * this->a + aOffset;
            int64_t currentR =
                (rLoopIdx == (quotient - 1)) ? (this->r - (quotient - 1) * this->rFactor) : this->rFactor;

            CopyInX(copyXOffset, currentR, currentANum);

            LocalTensor<T> xInUb = xQueue.DeQue<T>();
            __local_mem__ T* xLocal = (__local_mem__ T*)xInUb.GetPhyAddr();
            // process welford after copy ubSize data into ub.
            float scale = (float)1.0 / static_cast<float>(rLoopIdx + 1);
            uint64_t processNum = currentR * currentANumAlign;
            uint16_t updateLoopCount = CEIL_DIV(processNum, VL_F32);
            if (rLoopIdx == 0) {
                // 第一次更新时，需要将tmp mean和tmp var清0
                WelfordParallelUpdateWithInitVF(
                    xLocal, tmpMeanLocal, tmpVarLocal, tmpCountLocal, processNum, updateLoopCount, scale);
            } else {
                WelfordParallelUpdateVF(
                    xLocal, tmpMeanLocal, tmpVarLocal, tmpCountLocal, processNum, updateLoopCount, scale);
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

    template <typename T_SRC>
    __aicore__ inline void LoadOneTensorForDtypeT(
        __local_mem__ T_SRC* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
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

    __aicore__ inline void WelfordParallelUpdateWithInitVF(
        __local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        __local_mem__ float* tmpCountLocal, uint64_t calLen, uint16_t loopCount, float scale)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpVar;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delat4;
            vector_bool pregMain = pset_b8(PAT_ALL);
            vector_bool pregLoop;
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = plt_b32(sreg0, POST_UPDATE);
                LoadOneTensorForDtypeT(x1Local, x1, pregLoop, i * VL_F32);
                Duplicate(tmpMean, 0.0, pregLoop);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, pregLoop);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, pregLoop);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, pregLoop);
                DataCopy(tmpMeanLocal + i * VL_F32, tmpMean, pregLoop);

                Duplicate(tmpVar, 0.0, pregLoop);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, pregLoop);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, pregLoop);
                // var = var + delta4
                Add(tmpVar, tmpVar, delat4, pregLoop);
                DataCopy(tmpVarLocal + i * VL_F32, tmpVar, pregLoop);
            }
        }
    }

    __aicore__ inline void WelfordParallelUpdateVF(
        __local_mem__ T* x1Local, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        __local_mem__ float* tmpCountLocal, uint64_t calLen, uint16_t loopCount, float scale)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> tmpMean;
            RegTensor<float> tmpVar;
            RegTensor<float> delta1;
            RegTensor<float> delta2;
            RegTensor<float> delta3;
            RegTensor<float> delat4;
            MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            MaskReg pregLoop;
            uint32_t sreg0 = calLen;
            for (uint16_t i = 0; i < loopCount; i++) {
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                LoadOneTensorForDtypeT(x1Local, x1, pregLoop, i * VL_F32);

                DataCopy(tmpMean, tmpMeanLocal + i * VL_F32);
                // delata1 = x1 - mean
                Sub(delta1, x1, tmpMean, pregLoop);
                // delta2 = delta1 * scale
                Muls(delta2, delta1, scale, pregLoop);
                // mean = mean + delta2
                Add(tmpMean, tmpMean, delta2, pregLoop);
                DataCopy(tmpMeanLocal + i * VL_F32, tmpMean, pregLoop);

                DataCopy(tmpVar, tmpVarLocal + i * VL_F32);
                // delta3 = x1 - mean
                Sub(delta3, x1, tmpMean, pregLoop);
                // delta4 = delta1 * delta3
                Mul(delat4, delta1, delta3, pregLoop);
                // var = var + delta4
                Add(tmpVar, tmpVar, delat4, pregLoop);
                DataCopy(tmpVarLocal + i * VL_F32, tmpVar, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyInRunningMeanVar(int64_t aOffset, int64_t currentANum)
    {
        LocalTensor<T_BETA> betaInUb = betaQueue.AllocTensor<T_BETA>();
        LocalTensor<T_BETA> gammaInUb = gammaQueue.AllocTensor<T_BETA>();
        DataCopyPadExtParams<T_BETA> dataCopyPadExtParamsT;
        dataCopyPadExtParamsT.isPad = false;
        dataCopyPadExtParamsT.leftPadding = 0;
        dataCopyPadExtParamsT.rightPadding = 0;
        dataCopyPadExtParamsT.paddingValue = 0;
        DataCopyExtParams copyInParamsT;
        copyInParamsT.blockCount = 1;
        copyInParamsT.blockLen = currentANum * sizeof(T_BETA);
        copyInParamsT.srcStride = 0;
        copyInParamsT.dstStride = 0;
        DataCopyPad(betaInUb, betaGm[aOffset], copyInParamsT, dataCopyPadExtParamsT);
        DataCopyPad(gammaInUb, gammaGm[aOffset], copyInParamsT, dataCopyPadExtParamsT);
        betaQueue.EnQue(betaInUb);
        gammaQueue.EnQue(gammaInUb);

        LocalTensor<T_RUNNING_MEAN> runningMeanInUb = runningMeanInQueue.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarInUb = runningVarInQueue.AllocTensor<T_RUNNING_MEAN>();
        DataCopyPadExtParams<T_RUNNING_MEAN> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = currentANum * sizeof(T_RUNNING_MEAN);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanInUb, runningMeanGm[aOffset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(runningVarInUb, runningVarGm[aOffset], copyInParams, dataCopyPadExtParams);
        runningMeanInQueue.EnQue(runningMeanInUb);
        runningVarInQueue.EnQue(runningVarInUb);
    }

    __aicore__ inline void ProcessWelfordFinalize(
        int64_t aOffset, int64_t currentANum, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        __local_mem__ float* tmpCountLocal, __local_mem__ float* batchMeanInUbAddr,
        __local_mem__ float* batchRstdInUbAddr)
    {
        LocalTensor<T> yInUb = yQueue.AllocTensor<T>();
        __local_mem__ float* yInUbAddr = (__local_mem__ float*)yInUb.GetPhyAddr();
        WelfordFinalizeMeanVF(
            aOffset, currentANum, tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchRstdInUbAddr,
            yInUbAddr);
        WelfordFinalizeVarVF(
            aOffset, currentANum, tmpMeanLocal, tmpVarLocal, tmpCountLocal, batchMeanInUbAddr, batchRstdInUbAddr,
            yInUbAddr);
        yQueue.FreeTensor(yInUb);
    }

    __aicore__ inline void WelfordFinalizeMeanVF(
        int64_t aOffset, int64_t currentANum, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        __local_mem__ float* tmpCountLocal, __local_mem__ float* batchMeanInUbAddr,
        __local_mem__ float* batchRstdInUbAddr, __local_mem__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = this->rFactor;
        uint16_t aLoopCount = this->currentALoopCount;
        uint32_t rLoopStride = currentANumAlign;

        uint16_t remainderLoopCount = (this->rFactor - this->binaryAddQuotient) / SCALE_COEF_EIGHT;
        uint16_t quotientLoopCount = (this->binaryAddQuotient / SCALE_COEF_EIGHT) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_EIGHT * rLoopStride;
        uint32_t remainderOffset = this->binaryAddQuotient * currentANumAlign;
        uint32_t remainderCountOffset = this->binaryAddQuotient;

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddInnerLoop = this->binaryAddQuotient / SCALE_COEF_EIGHT;
        uint16_t binaryAddLastLoop = this->binaryAddLast;

        uint16_t finalLoopCount = this->binaryAddQuotient / SCALE_COEF_EIGHT;
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
            uint32_t sreg0 = currentANum;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_EIGHT;
                    uint32_t remCountOffset = i * SCALE_COEF_EIGHT + remainderCountOffset;
                    TwoRowAddForMeanWithTail(
                        x1, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset, remOffset, quotOffset + rLoopStride,
                        remOffset + rLoopStride, quotCountOffset, remCountOffset, quotCountOffset + 1,
                        remCountOffset + 1, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount, nextRemCount,
                        numScale);
                    TwoRowAddForMeanWithTail(
                        x2, tmpMeanLocal, tmpCountLocal, pregLoop, quotOffset + twoRLoopSize, remOffset + twoRLoopSize,
                        quotOffset + threeRLoopSize, remOffset + threeRLoopSize, quotCountOffset + ROW_TWO_OFFSET,
                        remCountOffset + ROW_TWO_OFFSET, quotCountOffset + ROW_THREE_OFFSET,
                        remCountOffset + ROW_THREE_OFFSET, rem, nextRow, remNextRow, rowCount, nextRowCount, remCount,
                        nextRemCount, numScale);
                    Add(x1, x1, x2, pregLoop);
                    DataCopy(((__local_mem__ float*)binaryAddTmpAddr + i * rLoopStride + aLoopOffset), x1, pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_EIGHT;
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
        int64_t aOffset, int64_t currentANum, __local_mem__ float* tmpMeanLocal, __local_mem__ float* tmpVarLocal,
        __local_mem__ float* tmpCountLocal, __local_mem__ float* batchMeanInUbAddr,
        __local_mem__ float* batchRstdInUbAddr, __local_mem__ float* binaryAddTmpAddr)
    {
        uint16_t rLoopCount = this->rFactor;
        uint16_t aLoopCount = this->currentALoopCount;
        uint32_t rLoopStride = currentANumAlign;

        uint16_t remainderLoopCount = (this->rFactor - this->binaryAddQuotient) / SCALE_COEF_EIGHT;
        uint16_t quotientLoopCount = (this->binaryAddQuotient / SCALE_COEF_EIGHT) - remainderLoopCount;
        uint32_t baseLineOffset = SCALE_COEF_EIGHT * rLoopStride;
        uint32_t remainderOffset = this->binaryAddQuotient * currentANumAlign;
        uint32_t remainderCountOffset = this->binaryAddQuotient;

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddInnerLoop = this->binaryAddQuotient / SCALE_COEF_EIGHT;
        uint16_t binaryAddLastLoop = this->binaryAddLast;

        uint16_t finalLoopCount = this->binaryAddQuotient / SCALE_COEF_EIGHT;
        float numScale = (float)1.0 / static_cast<float>(this->r);

        uint32_t twoRLoopSize = ROW_TWO_OFFSET * rLoopStride;
        uint32_t threeRLoopSize = ROW_THREE_OFFSET * rLoopStride;
        uint32_t fourRLoopSize = ROW_FOUR_OFFSET * rLoopStride;
        uint32_t fiveRLoopSize = ROW_FIVE_OFFSET * rLoopStride;
        uint32_t sixRLoopSize = ROW_SIX_OFFSET * rLoopStride;
        uint32_t sevenRLoopSize = ROW_SEVEN_OFFSET * rLoopStride;
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
            uint32_t sreg0 = currentANum;
            for (uint16_t aIndex = 0; aIndex < aLoopCount; aIndex++) {
                uint32_t aLoopOffset = aIndex * VL_F32;
                pregLoop = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                DataCopy(saveMean, ((__local_mem__ float*)batchMeanInUbAddr + aLoopOffset));
                for (uint16_t i = 0; i < remainderLoopCount; i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    uint32_t quotCountOffset = i * SCALE_COEF_EIGHT;
                    uint32_t remCountOffset = i * SCALE_COEF_EIGHT + remainderCountOffset;
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
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    uint32_t baseCountOffset = (remainderLoopCount + i) * SCALE_COEF_EIGHT;
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

    __aicore__ inline void CalculateRunningMeanVar(
        int64_t aOffset, int64_t currentANum, __local_mem__ float* batchMeanInUbAddr,
        __local_mem__ float* batchRstdInUbAddr)
    {
        uint16_t aLoop = currentALoopCount;
        LocalTensor<T_RUNNING_MEAN> runningMeanInUb = runningMeanInQueue.template DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarInUb = runningVarInQueue.template DeQue<T_RUNNING_MEAN>();

        LocalTensor<T_RUNNING_MEAN> runningMeanOutUb = runningMeanOutQueue.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarOutUb = runningVarOutQueue.AllocTensor<T_RUNNING_MEAN>();

        __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningMeanInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr =
            (__local_mem__ T_RUNNING_MEAN*)runningMeanOutUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarOutUb.GetPhyAddr();

        float besselCorrectionFactorVf = this->besselCorrectionFactor;
        float momentumVf = this->momentum;
        float oneSubMomentumVf = this->oneSubMomentum;
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
                DataCopy(var, ((__local_mem__ float*)batchRstdInUbAddr + k * VL_F32));
                Muls(saveVar, var, besselCorrectionFactorVf, pregLoop);
                Muls(saveVar, saveVar, momentumVf, pregLoop);
                Muls(runningVar, runningVar, oneSubMomentumVf, pregLoop);
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

                // rstd
                Adds(var, var, epsilon, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(-0.5), pregLoop);
                Mul(t, t, y, pregLoop);                // -0.5 * x * y
                Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);  // -1 * x
                Mula(s, t3, r, pregLoop);              // 1 + (-1) * x * r
                Muls(t4, rstd, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t4, rstd, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);             // s + x * t
                Mul(s, s, rstd, pregLoop);             // e * y
                Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(((__local_mem__ float*)batchRstdInUbAddr + k * VL_F32), rstd, pregLoop);

                // running mean
                DataCopy(mean, ((__local_mem__ float*)batchMeanInUbAddr + k * VL_F32));
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
                Muls(saveMean, mean, momentumVf, pregLoop);
                Muls(runningMean, runningMean, oneSubMomentumVf, pregLoop);
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
        runningMeanInQueue.FreeTensor(runningMeanInUb);
        runningVarInQueue.FreeTensor(runningVarInUb);
        runningMeanOutQueue.EnQue(runningMeanOutUb);
        runningVarOutQueue.EnQue(runningVarOutUb);
    }

    __aicore__ inline void Normalize(
        int64_t aOffset, int64_t currentANum, __local_mem__ float* batchMeanInUbAddr,
        __local_mem__ float* batchRstdInUbAddr)
    {
        LocalTensor<T_BETA> betaInUb = betaQueue.template DeQue<T_BETA>();
        LocalTensor<T_BETA> gammaInUb = gammaQueue.template DeQue<T_BETA>();
        __local_mem__ T_BETA* betaInUbAddr = (__local_mem__ T_BETA*)betaInUb.GetPhyAddr();
        __local_mem__ T_BETA* gammaInUbAddr = (__local_mem__ T_BETA*)gammaInUb.GetPhyAddr();
        int64_t quotient = (this->r + this->rFactor - 1) / this->rFactor;
        for (int64_t rLoopIdx = 0; rLoopIdx < quotient; rLoopIdx++) {
            int64_t copyXOffset = rLoopIdx * this->rFactor * this->a + aOffset;
            int64_t currentR =
                (rLoopIdx == (quotient - 1)) ? (this->r - (quotient - 1) * this->rFactor) : this->rFactor;

            CopyInX(copyXOffset, currentR, currentANum);
            NormalizeVF(currentR, currentANum, batchMeanInUbAddr, batchRstdInUbAddr, betaInUbAddr, gammaInUbAddr);
            CopyOutY(copyXOffset, currentR, currentANum);
        }
        betaQueue.FreeTensor(betaInUb);
        gammaQueue.FreeTensor(gammaInUb);
    }

    __aicore__ inline void NormalizeVF(
        int64_t currentR, int64_t currentANum, __local_mem__ float* batchMeanInUbAddr,
        __local_mem__ float* batchRstdInUbAddr, __local_mem__ T_BETA* betaInUbAddr, __local_mem__ T_BETA* gammaInUbAddr)
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
                        Cast<half, float, castTraitB322B16>(yFp16, y2, pregLoop);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ half*)yInUbAddr + rIndex * rLoopStride + aLoopOffset), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y2, pregLoop);
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
    static constexpr int64_t SCALE_COEF_EIGHT = 4;
    constexpr static int64_t NDDMA_DIM_NUM = 2;

    static constexpr uint32_t ROW_TWO_OFFSET = 2;
    static constexpr uint32_t ROW_THREE_OFFSET = 3;
    static constexpr uint32_t ROW_FOUR_OFFSET = 4;
    static constexpr uint32_t ROW_FIVE_OFFSET = 5;
    static constexpr uint32_t ROW_SIX_OFFSET = 6;
    static constexpr uint32_t ROW_SEVEN_OFFSET = 7;

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
