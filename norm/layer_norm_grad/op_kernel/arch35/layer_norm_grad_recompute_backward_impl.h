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
 * \file layer_norm_grad_recompute_backward_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_RECOMPUTE_BACKWARD_IMPL_
#define LAYER_NORM_GRAD_RECOMPUTE_BACKWARD_IMPL_

#include "layer_norm_grad_recompute.h"
#include "layer_norm_grad_base.h"

namespace LayerNormGrad {
using namespace AscendC;

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX, GM_ADDR workspace,
    const LayerNormGradTilingDataRecompute* tilingData, TPipe* pipeIn)
{
    td_ = tilingData;
    // Init
    if (GetBlockIdx() >= td_->backwardBlockDim) {
        return;
    }

    if (GetBlockIdx() < td_->backwardMainBlockCount) {
        Mloop = td_->backwardMLoopMain;
        Mtail = td_->backwardMLoopTail;
    } else {
        Mloop = td_->backwardMLoopTailTail;
        Mtail = td_->backwardMTailTail;
    }

    // Init GM Tensor
    int64_t xInTensorSize = GetBlockIdx() * td_->backwardMainBlockFactor * td_->col;
    int64_t varTensorSize = GetBlockIdx() * td_->backwardMainBlockFactor;
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + xInTensorSize);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + xInTensorSize);
    varInTensorGM.SetGlobalBuffer((__gm__ float*)var + varTensorSize);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + varTensorSize);
    gammaInTensorGM.SetGlobalBuffer((__gm__ U*)gamma);
    pdXOutTensorGM.SetGlobalBuffer((__gm__ T*)pdX + xInTensorSize);
 
    // Init Pipe
    int64_t dyBufSize = td_->backwardMfactor * td_->backwardNfactorBlockAligned * sizeof(float);
    int64_t cacheBufSize = td_->backwardCacheBufferCountMain * td_->backwardMfactorBlockAligned * sizeof(float);
    pipe_ = pipeIn;
    pipe_->InitBuffer(inQueueDy, 3, dyBufSize);
    pipe_->InitBuffer(inQueueX, 3, dyBufSize);
    pipe_->InitBuffer(outQueueDx, 1, dyBufSize);
    pipe_->InitBuffer(inQueueParam, 4, td_->backwardMfactorBlockAligned * sizeof(float));
    pipe_->InitBuffer(tempBuffer, td_->backwardMfactorBlockAligned * sizeof(float));
    pipe_->InitBuffer(inQueueGamma, 2, td_->backwardNfactorBlockAligned * sizeof(float));
    pipe_->InitBuffer(cacheBuffer0, cacheBufSize);
    pipe_->InitBuffer(cacheBuffer1, cacheBufSize);
    pipe_->InitBuffer(reduceSumTempBuffer, td_->backwardMfactorBlockAligned * td_->backwardFoldSize * sizeof(float));
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::Process()
{
    // Process
    if (GetBlockIdx() >= td_->backwardBlockDim) {
        return;
    }

    tempTensor = tempBuffer.Get<float>();
    cacheTensor0 = cacheBuffer0.Get<float>();
    cacheTensor1 = cacheBuffer1.Get<float>();

    reduceSumTempTensor = reduceSumTempBuffer.Get<float>();
    sum1Tensor = cacheTensor0[td_->backwardResultCacheIDMain * td_->backwardMfactorBlockAligned];
    sum2Tensor = cacheTensor1[td_->backwardResultCacheIDMain * td_->backwardMfactorBlockAligned];

    PRELOAD(4);
    int64_t totalMRounds = Mloop + (Mtail > 0 ? 1 : 0);
    int64_t nfactor = td_->backwardBasicBlockLoop ? td_->backwardNfactor
        : (td_->col == td_->backwardNfactor ? td_->backwardNfactor : td_->backwardNLoopTail);
    int64_t xTotalLoop = td_->backwardNLoopMain + (td_->backwardNLoopTail > 0 ? 1 : 0);
    for (int64_t mi = 0; mi < totalMRounds; ++mi) {
        int64_t mfactor = (mi < Mloop) ? td_->backwardMfactor : Mtail;
        Prologue(mi, mfactor);
        int64_t loopCnt = td_->backwardBasicBlockLoop > 0 ? td_->backwardBasicBlockLoop : 1;
        for (int64_t basicBlockIdx = 0; basicBlockIdx < loopCnt; ++basicBlockIdx) {
            ProcessMainBlock(mi, basicBlockIdx, mfactor, nfactor);
            if (td_->backwardBasicBlockLoop && ((basicBlockIdx < td_->backwardMainFoldCount) || (basicBlockIdx == td_->backwardMainFoldCount && td_->backwardNLoopTail > 0))) {
                int64_t foldNfactor = (basicBlockIdx < td_->backwardMainFoldCount) ? td_->backwardNfactor : td_->backwardNLoopTail;
                ProcessFoldBlock(mi, basicBlockIdx, mfactor, foldNfactor);
            }
            PRELOAD(2);
            ProcessSummation(mi, basicBlockIdx, mfactor, nfactor);
        }

        PRELOAD(4);
        for (int64_t ni = 0; ni < xTotalLoop; ni++) {
            int64_t xNfactor = (ni < td_->backwardNLoopMain) ? td_->backwardNfactor : td_->backwardNLoopTail;
            ProcessX(mi, ni, mfactor, xNfactor);
        }
        Epilogue();
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::Prologue(const int64_t mi, const int64_t mfactor)
{
    // Prologue
    int64_t offset = mi * td_->backwardMfactor;
    mean_ = inQueueParam.template AllocTensor<float>();
    CopyIn(mean_, meanInTensorGM[offset], mfactor);
    inQueueParam.EnQue(mean_);
    mean_ = inQueueParam.template DeQue<float>();

    var_ = inQueueParam.template AllocTensor<float>();
    CopyIn(var_, varInTensorGM[offset], mfactor);
    inQueueParam.EnQue(var_);
    var_ = inQueueParam.template DeQue<float>();
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::ProcessMainBlock(
    const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // Process Main Block
    int64_t offset = mi * td_->backwardMfactor * td_->col + basicBlockIdx * td_->backwardNfactor;
    sum1Main_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(sum1Main_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(sum1Main_);
        sum1Main_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = sum1Main_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(sum1Main_);
        sum1Main_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(sum1Main_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }

    offset = basicBlockIdx * td_->backwardNfactor;
    LocalTensor<float> gammaMain_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gammaMain_.ReinterpretCast<U>(), gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> castTempTensor = gammaMain_.ReinterpretCast<U>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gammaMain_, castTempTensor, nfactor);
    }
    // 计算
    NlastBroadcastMul(sum1Main_, sum1Main_, gammaMain_, mfactor, td_->backwardNfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaMain_);

    offset = mi * td_->backwardMfactor * td_->col + basicBlockIdx * td_->backwardNfactor;
    sum2Main_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(sum2Main_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = sum2Main_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(sum2Main_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }
    // 计算
    Normalize(sum2Main_, sum2Main_, mean_, var_, mfactor, td_->backwardNfactorBlockAligned, td_->epsilon);
    // 计算
    VectorMul(sum2Main_, sum2Main_, sum1Main_, mfactor * td_->backwardNfactorBlockAligned);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::ProcessFoldBlock(
    const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // Process Fold Block
    int64_t offset = mi * td_->backwardMfactor * td_->col + (basicBlockIdx + td_->backwardBasicBlockLoop) * td_->backwardNfactor;
    LocalTensor<float> dyFold_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyFold_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyFold_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyFold_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }

    offset = (basicBlockIdx + td_->backwardBasicBlockLoop) * td_->backwardNfactor;
    LocalTensor<float> gammaFold_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gammaFold_.ReinterpretCast<U>(), gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaFold_);
        gammaFold_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> castTempTensor = gammaFold_.ReinterpretCast<U>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaFold_);
        gammaFold_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gammaFold_, castTempTensor, nfactor);
    }
    // 进行计算
    PRELOAD(2);
    NlastBroadcastMul(dyFold_, dyFold_, gammaFold_, mfactor, td_->backwardNfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaFold_);

    offset = mi * td_->backwardMfactor * td_->col + (basicBlockIdx + td_->backwardBasicBlockLoop) * td_->backwardNfactor;
    LocalTensor<float> xFold_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(xFold_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(xFold_);
        xFold_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = xFold_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(xFold_);
        xFold_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(xFold_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }
    Normalize(xFold_, xFold_, mean_, var_, mfactor, td_->backwardNfactorBlockAligned, td_->epsilon);
    VectorMul(xFold_, xFold_, dyFold_, mfactor * td_->backwardNfactorBlockAligned);

    VectorAdd(sum1Main_, sum1Main_, dyFold_, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    inQueueDy.FreeTensor(dyFold_);
    VectorAdd(sum2Main_, sum2Main_, xFold_, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    inQueueX.FreeTensor(xFold_);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::ProcessSummation(
    const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // Process Summation
    int64_t cacheID = GetCacheID(basicBlockIdx);
    LastReduceSum(tempTensor, sum1Main_, reduceSumTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    inQueueDy.FreeTensor(sum1Main_);
    UpdateCache(cacheTensor0, tempTensor, cacheID, td_->backwardMfactorBlockAligned, mfactor);
    LastReduceSum(tempTensor, sum2Main_, reduceSumTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    inQueueX.FreeTensor(sum2Main_);
    UpdateCache(cacheTensor1, tempTensor, cacheID, td_->backwardMfactorBlockAligned, mfactor);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::ProcessX(
    const int64_t mi, const int64_t ni, const int64_t mfactor, const int64_t nfactor)
{
    // Process X
    int64_t offset = mi * td_->backwardMfactor * td_->col + ni * td_->backwardNfactor;
    LocalTensor<float> x_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(x_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(x_);
        x_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = x_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(x_);
        x_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(x_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }
    Normalize(x_, x_, mean_, var_, mfactor, td_->backwardNfactorBlockAligned, td_->epsilon);

    LocalTensor<float> dy_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dy_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dy_);
        dy_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dy_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dy_);
        dy_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dy_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }

    LocalTensor<float> gamma_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gamma_.ReinterpretCast<U>(), gammaInTensorGM[ni * td_->backwardNfactor], nfactor);
        inQueueGamma.EnQue(gamma_);
        gamma_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> castTempTensor = gamma_.ReinterpretCast<U>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, gammaInTensorGM[ni * td_->backwardNfactor], nfactor);
        inQueueGamma.EnQue(gamma_);
        gamma_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gamma_, castTempTensor, nfactor);
    }

    LocalTensor<T> dx_ = outQueueDx.template AllocTensor<T>();
    ComputeDx(dx_, dy_, x_, gamma_, sum1Tensor, sum2Tensor, var_, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    inQueueDy.FreeTensor(dy_);
    inQueueX.FreeTensor(x_);
    inQueueGamma.FreeTensor(gamma_);

    outQueueDx.EnQue(dx_);
    dx_ = outQueueDx.template DeQue<T>();
    CopyOut(pdXOutTensorGM[offset], dx_, mfactor, nfactor, td_->col, td_->backwardNfactorBlockAligned);
    outQueueDx.FreeTensor(dx_);
}

// 二分累加
template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::ComputeDx(
    const LocalTensor<T>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& gammaTensor, const LocalTensor<float>& sum1Tensor, const LocalTensor<float>& sum2Tensor,
    const LocalTensor<float>& varTensor, const int64_t rowSize, const int64_t colSize, const int64_t stride)
{
    // Compute Dx
    constexpr static uint32_t VL = GetVRegSize() / sizeof(float);
    uint16_t outerLoopTimes = rowSize;
    uint16_t innerLoopTimes =
        CeilDiv(static_cast<int64_t>(colSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL;
    float floatN = static_cast<float>(td_->col);
    float reciprocalN = static_cast<float>(1) / floatN;
    float epsilonTmp = td_->epsilon;

    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
            __local_mem__ float* dy = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* x = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* gamma = (__local_mem__ float*)gammaTensor.GetPhyAddr();
            __local_mem__ float* sum1 = (__local_mem__ float*)sum1Tensor.GetPhyAddr();
            __local_mem__ float* sum2 = (__local_mem__ float*)sum2Tensor.GetPhyAddr();
            __local_mem__ float* var = (__local_mem__ float*)varTensor.GetPhyAddr();
            uint32_t count;

            AscendC::MicroAPI::RegTensor<float> xReg, dyReg, dxReg;
            AscendC::MicroAPI::RegTensor<float> sum1Reg, sum2Reg, varReg, rstdReg;
            AscendC::MicroAPI::RegTensor<float> gammaReg;
            AscendC::MicroAPI::RegTensor<float> Reg0, Reg1, Reg2, Reg3, Reg4, Reg5;
            AscendC::MicroAPI::MaskReg pMask;
            count = static_cast<uint32_t>(colSize);
            pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum1Reg, (__local_mem__ float*)sum1 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum2Reg, (__local_mem__ float*)sum2 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(varReg, (__local_mem__ float*)var + i);
                CalRstdByHighPrecision(varReg, rstdReg, epsilonTmp);
                DataCopy(dyReg, (__local_mem__ float*)dy + i * outerLoopStride + 0 * innerLoopStride);
                DataCopy(xReg, (__local_mem__ float*)x + i * outerLoopStride + 0 * innerLoopStride);
                DataCopy(gammaReg, (__local_mem__ float*)gamma + 0 * innerLoopStride);
                Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg0, dyReg, gammaReg, pMask);
                Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg1, Reg0, floatN, pMask);
                Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg2, Reg1, sum1Reg, pMask);
                Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg3, xReg, sum2Reg, pMask);
                Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg4, Reg2, Reg3, pMask);
                Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg5, Reg4, reciprocalN, pMask);
                Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dxReg, Reg5, rstdReg, pMask);
                StoreTensorForDtypeT<T>(dst, dxReg, pMask, i * outerLoopStride);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
            __local_mem__ float* dy = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* x = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* gamma = (__local_mem__ float*)gammaTensor.GetPhyAddr();
            __local_mem__ float* sum1 = (__local_mem__ float*)sum1Tensor.GetPhyAddr();
            __local_mem__ float* sum2 = (__local_mem__ float*)sum2Tensor.GetPhyAddr();
            __local_mem__ float* var = (__local_mem__ float*)varTensor.GetPhyAddr();
            uint32_t count;

            AscendC::MicroAPI::RegTensor<float> xReg, dyReg, dxReg;
            AscendC::MicroAPI::RegTensor<float> sum1Reg, sum2Reg, varReg, rstdReg;
            AscendC::MicroAPI::RegTensor<float> gammaReg;
            AscendC::MicroAPI::RegTensor<float> Reg0, Reg1, Reg2, Reg3, Reg4, Reg5;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                count = static_cast<uint32_t>(colSize);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum1Reg, (__local_mem__ float*)sum1 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum2Reg, (__local_mem__ float*)sum2 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(varReg, (__local_mem__ float*)var + i);
                CalRstdByHighPrecision(varReg, rstdReg, epsilonTmp);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    DataCopy(dyReg, (__local_mem__ float*)dy + i * outerLoopStride + j * innerLoopStride);
                    DataCopy(xReg, (__local_mem__ float*)x + i * outerLoopStride + j * innerLoopStride);
                    DataCopy(gammaReg, (__local_mem__ float*)gamma + j * innerLoopStride);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg0, dyReg, gammaReg, pMask);
                    Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg1, Reg0, floatN, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg2, Reg1, sum1Reg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg3, xReg, sum2Reg, pMask);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg4, Reg2, Reg3, pMask);
                    Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(Reg5, Reg4, reciprocalN, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dxReg, Reg5, rstdReg, pMask);
                    StoreTensorForDtypeT<T>(dst, dxReg, pMask, i * outerLoopStride + j * innerLoopStride);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradRecomputeBackward<T, U>::Epilogue()
{
    // Epilogue
    inQueueParam.FreeTensor(mean_);
    inQueueParam.FreeTensor(var_);
}
} // namespace LayerNormGrad
#endif
