/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_norm_v3.h
 * \brief
 */
#ifndef BATCH_NORM_V3_H
#define BATCH_NORM_V3_H

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
using AscendC::MicroAPI::UpdateMask;
using AscendC::MicroAPI::StoreDist;

constexpr static int64_t BLOCK_SIZE = 32;
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static int64_t MAX_STRIDE = 65535;
constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int64_t LOOP_HIGH_SHIFT = 21;
constexpr static int64_t LOOP_STRIDE_HIGH_SHIFT = 40;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static float ZERO = 0.0f;
constexpr static int64_t NDDMA_THRESHOLD = 32;
constexpr static int64_t NDDMA_SECOND_DIM = 1;
constexpr static int64_t NDDMA_THIRD_DIM = 2;
constexpr static int64_t NDDMA_DIM_NUM = 3;
constexpr static int64_t DICHOTOMY_ADD_COEFF = 2;

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310 || __NPU_ARCH == 5102
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    return a / b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    using type = typename std::conditional<sizeof(T) == sizeof(uint8_t) || sizeof(T) == sizeof(uint16_t), uint32_t, uint64_t>::type;
    type res = (static_cast<type>(a) + static_cast<type>(b) - 1) / static_cast<type>(b);
    return static_cast<T>(res);
}

template <typename T, typename T_BETA, typename T_RUNNING_MEAN>
class BatchNormV3FullReduce {
public:
    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline BatchNormV3FullReduce(const BatchNormV3FullReduceRegbaseTilingData* tilingData)
    {
        this->r1 = tilingData->r1;
        this->aFactor = tilingData->aFactor;
        this->a = tilingData->a;
        this->r0 = tilingData->r0;
        this->blockNum = tilingData->blockNum;
        this->aBlockFactor = tilingData->aBlockFactor;
        this->r1r0LoopCount = tilingData->r1r0LoopCount;
        this->epsilon = tilingData->epsilon;
        this->momentum = tilingData->momentum;

        float one = 1.0;
        this->oneSubMomentum = one - this->momentum;
        int64_t reduceNum = this->r1 * this->r0;
        this->besselCorrectionFactor = (static_cast<float>(reduceNum) / static_cast<float>(reduceNum - 1));

        this->powerOfTwoForR = tilingData->powerOfTwoForR;
        this->binaryAddQuotient = tilingData->binaryAddQuotient;
        this->binaryAddK = tilingData->binaryAddK;
        this->binaryAddLastNum = tilingData->binaryAddLastNum;
    }

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR var, GM_ADDR y, GM_ADDR mean_out, GM_ADDR var_out,
        GM_ADDR batch_mean, GM_ADDR batch_rstd)
    {
        auto blockIdx = GetBlockIdx();

        this->r1r0Align = (((this->r1 * this->r0 * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T);
        this->singleA = (blockIdx == this->blockNum - 1) ? (this->a - this->aBlockFactor * (this->blockNum - 1)) :
                                                           this->aBlockFactor;

        int64_t aGmOffset = this->aBlockFactor * blockIdx;
        int64_t arGmOffset = aGmOffset * this->r0;
        xGm.SetGlobalBuffer((__gm__ T*)x + arGmOffset);
        betaGm.SetGlobalBuffer((__gm__ T_BETA*)beta + aGmOffset);
        gammaGm.SetGlobalBuffer((__gm__ T_BETA*)gamma + aGmOffset);
        runningMeanGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean + aGmOffset);
        runningVarGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var + aGmOffset);

        yGm.SetGlobalBuffer((__gm__ T*)y + arGmOffset);
        batchMeanGm.SetGlobalBuffer((__gm__ float*)batch_mean + aGmOffset);
        batchRstdGm.SetGlobalBuffer((__gm__ float*)batch_rstd + aGmOffset);
        runningMeanOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)mean_out + aGmOffset);
        runningVarOutGm.SetGlobalBuffer((__gm__ T_RUNNING_MEAN*)var_out + aGmOffset);

        int64_t aFactorAlign = (((this->aFactor * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / sizeof(T);
        pipe.InitBuffer(betaQueue, DOUBLE_BUFFER, aFactorAlign * sizeof(T_BETA));
        pipe.InitBuffer(gammaQueue, DOUBLE_BUFFER, aFactorAlign * sizeof(T_BETA));
        int64_t aFactorAlignF32 =
            (((this->aFactor * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE) / FLOAT_BYTES;
        pipe.InitBuffer(batchMeanQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe.InitBuffer(batchRstdQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        pipe.InitBuffer(runningMeanInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe.InitBuffer(runningVarInQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe.InitBuffer(runningMeanOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
        pipe.InitBuffer(runningVarOutQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));

        int64_t xBufferSize = this->aFactor * this->r1r0Align;
        pipe.InitBuffer(xQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T));
        pipe.InitBuffer(yQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T));

        int64_t binaryAddBufSize =
            (((binaryAddQuotient / VL_F32) * FLOAT_BYTES + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        if (binaryAddBufSize > 0) {
            pipe.InitBuffer(binaryAddBuf, binaryAddBufSize);
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

        LocalTensor<T> xInUb = xQueue.template DeQue<T>();
        __local_mem__ T* xInUbAddr = (__local_mem__ T*)xInUb.GetPhyAddr();
        LocalTensor<float> batchMeanOutUb = batchMeanQueue.AllocTensor<float>();
        LocalTensor<float> batchRstdOutUb = batchRstdQueue.AllocTensor<float>();
        __local_mem__ float* batchMeanInUbAddr = (__local_mem__ float*)batchMeanOutUb.GetPhyAddr();
        __local_mem__ float* batchRstdInUbAddr = (__local_mem__ float*)batchRstdOutUb.GetPhyAddr();
        if (this->r1 * this->r0 <= VL_F32) {
            CalculateMeanVarRLessThanVL64VF(xInUbAddr, batchMeanInUbAddr, batchRstdInUbAddr, currentANum);
        } else {
            CalculateMeanVarVF(xInUbAddr, batchMeanInUbAddr, batchRstdInUbAddr, currentANum);
        }

        CopyInGammaBetaCommon(betaQueue, gammaQueue, betaGm, gammaGm, aOffset, currentANum);
        CopyInRunningMeanVarCommon(
            runningMeanInQueue, runningVarInQueue, runningMeanGm, runningVarGm, aOffset, currentANum);

        LocalTensor<T_BETA> betaInUb = betaQueue.template DeQue<T_BETA>();
        LocalTensor<T_BETA> gammaInUb = gammaQueue.template DeQue<T_BETA>();
        LocalTensor<T_RUNNING_MEAN> runningMeanInUb = runningMeanInQueue.template DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarInUb = runningVarInQueue.template DeQue<T_RUNNING_MEAN>();

        LocalTensor<T> yInUb = yQueue.AllocTensor<T>();
        LocalTensor<T_RUNNING_MEAN> runningMeanOutUb = runningMeanOutQueue.AllocTensor<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarOutUb = runningVarOutQueue.AllocTensor<T_RUNNING_MEAN>();
        __local_mem__ T* yInUbAddr = (__local_mem__ T*)yInUb.GetPhyAddr();
        __local_mem__ T_BETA* betaInUbAddr = (__local_mem__ T_BETA*)betaInUb.GetPhyAddr();
        __local_mem__ T_BETA* gammaInUbAddr = (__local_mem__ T_BETA*)gammaInUb.GetPhyAddr();

        __local_mem__ T_RUNNING_MEAN* runningMeanInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningMeanInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarInUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarInUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningMeanOutUbAddr =
            (__local_mem__ T_RUNNING_MEAN*)runningMeanOutUb.GetPhyAddr();
        __local_mem__ T_RUNNING_MEAN* runningVarOutUbAddr = (__local_mem__ T_RUNNING_MEAN*)runningVarOutUb.GetPhyAddr();
        uint16_t aLoop = static_cast<uint16_t>(CEIL_DIV(currentANum, VL_F32));
        CalculateRunningMeanVarWithRstdVF<T_RUNNING_MEAN>(batchMeanInUbAddr, batchRstdInUbAddr,
            runningMeanInUbAddr, runningVarInUbAddr, runningMeanOutUbAddr, runningVarOutUbAddr, currentANum, aLoop,
            VL_F32, this->besselCorrectionFactor, this->momentum, this->oneSubMomentum, this->epsilon);

        runningMeanInQueue.FreeTensor(runningMeanInUb);
        runningVarInQueue.FreeTensor(runningVarInUb);
        batchMeanQueue.EnQue(batchMeanOutUb);
        batchRstdQueue.EnQue(batchRstdOutUb);
        runningMeanOutQueue.EnQue(runningMeanOutUb);
        runningVarOutQueue.EnQue(runningVarOutUb);

        CopyOutBatchMeanRstdCommon(
            batchMeanQueue, batchRstdQueue, batchMeanGm, batchRstdGm, aOffset, currentANum);
        CopyOutRunningMeanVar(aOffset, currentANum);

        CalculateNormalizeVF(
            xInUbAddr, yInUbAddr, betaInUbAddr, gammaInUbAddr, batchMeanInUbAddr, batchRstdInUbAddr, currentANum);

        xQueue.FreeTensor(xInUb);
        betaQueue.FreeTensor(betaInUb);
        gammaQueue.FreeTensor(gammaInUb);
        yQueue.EnQue(yInUb);

        CopyOutY(raOffset, currentANum);
    }

    __aicore__ inline void CopyInX(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> xInUb = xQueue.AllocTensor<T>();
        if (this->r0 * sizeof(T) <= NDDMA_THRESHOLD) {
            T constValue = 0;
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
            MultiCopyParams<T, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
            DataCopy<T, NDDMA_DIM_NUM, config>(xInUb, xGm[offset], paramsMain);
        } else {
            uint64_t r1LoopSrcStride = this->r0 * sizeof(T);
            uint64_t r1LoopDstStride = this->r1r0Align * sizeof(T);
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = currentANum;
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;
            loopParams.loop1SrcStride = r1LoopSrcStride;
            loopParams.loop1DstStride = r1LoopDstStride;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPadExtParams<T> dataCopyPadExtParams;
            dataCopyPadExtParams.isPad = false;
            dataCopyPadExtParams.leftPadding = 0;
            dataCopyPadExtParams.rightPadding = 0;
            dataCopyPadExtParams.paddingValue = 0;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = this->r1;
            copyInParams.blockLen = this->r0 * sizeof(T);
            copyInParams.srcStride = (this->a - 1) * this->r0 * sizeof(T);
            copyInParams.dstStride = 0;
            DataCopyPad<T, PaddingMode::Compact>(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
        xQueue.EnQue(xInUb);
    }

    __aicore__ inline void CopyOutY(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T> yOutUb = yQueue.template DeQue<T>();

        uint64_t r1LoopSrcStride = this->r1r0Align * sizeof(T);
        uint64_t r1LoopDstStride = this->r0 * sizeof(T);
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
        copyInParams.blockLen = this->r0 * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->a - 1) * this->r0 * sizeof(T);
        DataCopyPad<T, PaddingMode::Compact>(yGm[offset], yOutUb, copyInParams);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        yQueue.FreeTensor(yOutUb);
    }

    __aicore__ inline void CopyOutRunningMeanVar(int64_t offset, int64_t currentANum)
    {
        LocalTensor<T_RUNNING_MEAN> runningMeanOutUb = runningMeanOutQueue.template DeQue<T_RUNNING_MEAN>();
        LocalTensor<T_RUNNING_MEAN> runningVarOutUb = runningVarOutQueue.template DeQue<T_RUNNING_MEAN>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.blockLen = currentANum * sizeof(T_RUNNING_MEAN);
        copyInParams.dstStride = 0;
        DataCopyPad(runningMeanOutGm[offset], runningMeanOutUb, copyInParams);
        DataCopyPad(runningVarOutGm[offset], runningVarOutUb, copyInParams);
        runningMeanOutQueue.FreeTensor(runningMeanOutUb);
        runningVarOutQueue.FreeTensor(runningVarOutUb);
    }

    __aicore__ inline void LoadTwoTensorForDtypeT(
        __local_mem__ T* src1, __local_mem__ T* src2, RegTensor<float>& dst1, RegTensor<float>& dst2, MaskReg& dst1Preg,
        MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
    {
        if constexpr (IsSameType<T, half>::value) {
            RegTensor<half> xFp16Q;
            RegTensor<half> xFp16R;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
            Cast<float, half, NormCommon::castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, half, NormCommon::castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xFp16Q;
            RegTensor<bfloat16_t> xFp16R;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
            Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, bfloat16_t, NormCommon::castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else {
            DataCopy(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
            DataCopy(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
        }
    }

    __aicore__ inline void CalculateMeanVarRLessThanVL64VF(
        __local_mem__ T* xInUb, __local_mem__ float* batchMeanInUb, __local_mem__ float* batchRstdInUb,
        uint16_t currentANum)
    {
        int64_t calcNum = this->r1 * this->r0;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(calcNum);
        uint32_t xyUbOffset = this->r1r0Align;
        __VEC_SCOPE__
        {
            RegTensor<float> var_sum;
            RegTensor<float> var;

            RegTensor<float> x;
            RegTensor<float> mean_sum;
            RegTensor<float> mean;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            uint32_t sreg0 = calcNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            for (uint16_t k = 0; k < currentANum; k++) {
                LoadOneTensorForDtypeT(xInUb, x, pregLoop, (k * xyUbOffset));
                Muls(mean_sum, x, n, pregLoop);
                ReduceSum(mean, mean_sum, pregLoop);
                Muls(mean, mean, nCorrectionFactor, pregMerge);

                // save mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)batchMeanInUb + k), mean, pregMerge);
                Duplicate(mean, mean, pregMain);
                Muls(mean, mean, (float)-1.0, pregMain);

                LoadOneTensorForDtypeT(xInUb, x1, pregLoop, (k * xyUbOffset));
                Add(y1, x1, mean, pregLoop);
                Mul(y1Pow, y1, y1, pregLoop);
                Muls(var_sum, y1Pow, n, pregLoop);
                ReduceSum(var, var_sum, pregLoop);
                Muls(var, var, nCorrectionFactor, pregMerge);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)batchRstdInUb + k), var, pregMerge);
            }
        }
    }

    __aicore__ inline void CalculateMeanVarVF(
        __local_mem__ T* xInUb, __local_mem__ float* batchMeanInUb, __local_mem__ float* batchRstdInUb,
        uint16_t currentANum)
    {
        int64_t reduceNum = this->r1 * this->r0;
        float n = static_cast<float>(1) / static_cast<float>(this->powerOfTwoForR);
        float nCorrectionFactor = static_cast<float>(this->powerOfTwoForR) / static_cast<float>(reduceNum);
        uint32_t xyUbOffset = this->r1r0Align;

        uint32_t binaryAddQuotientOffset = this->binaryAddQuotient;
        int64_t binaryAddRemainder = reduceNum - this->binaryAddQuotient;
        uint16_t binaryAddRemainderLoop = CEIL_DIV(binaryAddRemainder, VL_F32);
        uint16_t binaryAddQuotientLoop = CEIL_DIV(this->binaryAddQuotient, VL_F32);

        uint16_t binaryAddKLoop = this->binaryAddK;
        uint16_t binaryAddLoopMean = ((this->binaryAddQuotient / VL_F32) / VL_F32);
        uint16_t binaryAddLoopVar = binaryAddLoopMean;
        LocalTensor<float> binaryAddTensor = binaryAddBuf.Get<float>();
        __local_mem__ float* binaryAddTensorAddr = (__local_mem__ float*)binaryAddTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> var_sum;
            RegTensor<float> var;

            RegTensor<float> x;
            RegTensor<float> mean_sum;
            RegTensor<float> mean;

            RegTensor<float> x1;
            RegTensor<float> y1;
            RegTensor<float> y1Pow;

            RegTensor<float> vlMean;
            RegTensor<float> vlVar;

            RegTensor<float> binaryAddQ;
            RegTensor<float> binaryAddR;

            RegTensor<float> binaryAddQPow;
            RegTensor<float> binaryAddRPow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;

            for (uint16_t k = 0; k < currentANum; k++) {
                uint32_t sreg0 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregLoop, pregLoop, (i * VL_F32 + k * xyUbOffset),
                        (i * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregLoop);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregLoop);
                    ReduceSum(vlMean, binaryAddQ, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + i), vlMean, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg0);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Muls(binaryAddQ, binaryAddQ, n, pregMain);
                    Muls(binaryAddR, binaryAddR, n, pregLoop);
                    Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                    ReduceSum(vlMean, binaryAddQ, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop - 1), vlMean, pregMerge);
                }
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderLoop); i++) {
                    LoadOneTensorForDtypeT(
                        xInUb, x, pregMain, ((i + binaryAddRemainderLoop) * VL_F32 + k * xyUbOffset));
                    Muls(x, x, n, pregMain);
                    ReduceSum(vlMean, x, pregMain);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + binaryAddRemainderLoop + i), vlMean, pregMerge);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                uint16_t curBinaryAddLoopMean = binaryAddLoopMean;
                for (uint16_t i = 0; i < binaryAddKLoop; i++) {
                    curBinaryAddLoopMean = curBinaryAddLoopMean / DICHOTOMY_ADD_COEFF;
                    for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32));
                        DataCopy(
                            binaryAddR,
                            ((__local_mem__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopMean) * VL_F32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t binaryAddLastNum = this->binaryAddLastNum;
                    pregLoop = UpdateMask<float>(binaryAddLastNum);
                    DataCopy(mean_sum, ((__local_mem__ float*)binaryAddTensorAddr));
                    ReduceSum(mean, mean_sum, pregLoop);
                    Muls(mean, mean, nCorrectionFactor, pregMerge);
                }

                // batch mean
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                    ((__local_mem__ float*)batchMeanInUb + k), mean, pregMerge);
                Duplicate(mean, mean, pregMain);
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();

                uint32_t sreg1 = binaryAddRemainder;
                for (uint16_t i = 0; i < static_cast<uint16_t>(binaryAddRemainderLoop - 1); i++) {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregLoop, pregLoop, (i * VL_F32 + k * xyUbOffset),
                        (i * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
                    Sub(binaryAddQ, binaryAddQ, mean, pregLoop);
                    Sub(binaryAddR, binaryAddR, mean, pregLoop);
                    Mul(binaryAddQPow, binaryAddQ, binaryAddQ, pregLoop);
                    Mul(binaryAddRPow, binaryAddR, binaryAddR, pregLoop);
                    Muls(binaryAddQPow, binaryAddQPow, n, pregLoop);
                    Muls(binaryAddRPow, binaryAddRPow, n, pregLoop);
                    Add(binaryAddQPow, binaryAddQPow, binaryAddRPow, pregLoop);
                    ReduceSum(vlVar, binaryAddQPow, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)binaryAddTensorAddr + i), vlVar, pregMerge);
                }
                {
                    pregLoop = UpdateMask<float>(sreg1);
                    LoadTwoTensorForDtypeT(
                        xInUb, xInUb, binaryAddQ, binaryAddR, pregMain, pregLoop,
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset),
                        ((binaryAddRemainderLoop - 1) * VL_F32 + k * xyUbOffset + binaryAddQuotientOffset));
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
                    LoadOneTensorForDtypeT(
                        xInUb, x1, pregMain, ((i + binaryAddRemainderLoop) * VL_F32 + k * xyUbOffset));
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
                    curBinaryAddLoopVar = curBinaryAddLoopVar / DICHOTOMY_ADD_COEFF;
                    for (uint16_t j = 0; j < curBinaryAddLoopVar; j++) {
                        DataCopy(binaryAddQ, ((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32));
                        DataCopy(
                            binaryAddR,
                            ((__local_mem__ float*)binaryAddTensorAddr + (j + curBinaryAddLoopVar) * VL_F32));
                        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
                        DataCopy(((__local_mem__ float*)binaryAddTensorAddr + j * VL_F32), binaryAddQ, pregMain);
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                }
                {
                    uint32_t sreg2 = this->binaryAddLastNum;
                    pregLoop = UpdateMask<float>(sreg2);
                    DataCopy(var_sum, ((__local_mem__ float*)binaryAddTensorAddr));
                    ReduceSum(var, var_sum, pregLoop);
                    Muls(var, var, nCorrectionFactor, pregMerge);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        ((__local_mem__ float*)batchRstdInUb + k), var, pregMerge);
                }
                LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>();
            }
        }
    }

    __aicore__ inline void CalculateNormalizeVF(
        __local_mem__ T* xInUb, __local_mem__ T* yInUb, __local_mem__ T_BETA* betaInUb, __local_mem__ T_BETA* gammaInUb,
        __local_mem__ float* batchMeanInUb, __local_mem__ float* batchRstdInUb, uint16_t currentANum)
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
                LoadTwoTensorForDtypeTBrc(betaInUb, gammaInUb, beta, gamma, pregMain, pregMain, k, k);
                uint32_t sreg3 = calcNum;
                for (uint16_t i = 0; i < loopCount; i++) {
                    pregLoop = UpdateMask<float>(sreg3);
                    LoadOneTensorForDtypeT(xInUb, x2, pregLoop, (i * VL_F32 + k * xyUbOffset));
                    DataCopy<float, LoadDist::DIST_BRC_B32>(mean, ((__local_mem__ float*)batchMeanInUb + k));
                    Sub(x2, x2, mean, pregLoop);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(rsqrtVar, ((__local_mem__ float*)batchRstdInUb + k));
                    Mul(y2, x2, rsqrtVar, pregLoop);
                    Mul(y2, y2, beta, pregLoop);
                    Add(y2, y2, gamma, pregLoop);
                    if constexpr (IsSameType<T, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, NormCommon::castTraitB322B16>(yFp16, y2, pregLoop);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ half*)yInUb + i * VL_F32 + k * xyUbOffset), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, NormCommon::castTraitB322B16>(xBf16, y2, pregLoop);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ bfloat16_t*)yInUb + i * VL_F32 + k * xyUbOffset), xBf16, pregLoop);
                    } else {
                        DataCopy(((__local_mem__ float*)yInUb + i * VL_F32 + k * xyUbOffset), y2, pregLoop);
                    }
                }
            }
        }
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

    static constexpr uint32_t VL_F32 = VECTOR_REG_WIDTH / sizeof(float);

    float epsilon = 1e-5;
    float momentum = 0.1;
    float besselCorrectionFactor;
    float oneSubMomentum;

    /* ascendc variable */
    TPipe pipe;
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
} // namespace BatchNormV3Ops

#endif
