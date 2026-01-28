/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * \file layer_norm_grad_grouped_reduce_big_n_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_GROUPED_REDUCE_BIG_N_IMPL_
#define LAYER_NORM_GRAD_GROUPED_REDUCE_BIG_N_IMPL_

#include "layer_norm_grad_grouped_reduce.h"

namespace LayerNormGrad
{
using namespace AscendC;

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
    const LayerNormGradTilingDataGroupedReduceBigN* tilingData, TPipe* pipeIn)
{
    td_ = tilingData;

    if (GetBlockIdx() >= td_->gammaBetaBlockDim) {
        return;
    }

    if (GetBlockIdx() < (td_->gammaBetaBlockDim - 1)) {
        NTotalloop = td_->gammaBetaNloopMainBlock;
        Ntail = td_->gammaBetaNtailMainBlock;
    } else {
        NTotalloop = td_->gammaBetaNloopTailBlock;
        Ntail = td_->gammaBetaNtailTailBlock;
    }

    // Init GM Tensor
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + GetBlockIdx() * td_->gammaBetaMainBlockFactor);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * td_->gammaBetaMainBlockFactor);
    varInTensorGM.SetGlobalBuffer((__gm__ float*)var, td_->row);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean, td_->row);
    pdGammaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)pdGamma + GetBlockIdx() * td_->gammaBetaMainBlockFactor);
    pdBetaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)pdBeta + GetBlockIdx() * td_->gammaBetaMainBlockFactor);

    // Init Pipe
    pipe_ = pipeIn;
    int64_t dyBufSize = td_->gammaBetaMfactor * td_->gammaBetaNfactor * sizeof(float);
    int64_t cacheBufSize = td_->gammaBetaCacheBufferCount * td_->gammaBetaNfactor * sizeof(float);
    pipe_->InitBuffer(inQueueDy, 3, dyBufSize);
    pipe_->InitBuffer(inQueueX, 3, dyBufSize);
    pipe_->InitBuffer(inQueueParam, 2, td_->gammaBetaMfactor * sizeof(float));
    pipe_->InitBuffer(outQueueSum, 2, td_->gammaBetaNfactor * sizeof(float));
    pipe_->InitBuffer(cacheBuffer0, cacheBufSize);
    pipe_->InitBuffer(cacheBuffer1, cacheBufSize);
    pipe_->InitBuffer(tempBuffer, td_->gammaBetaNfactor * sizeof(float));
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::Process()
{
    // Process
    if (GetBlockIdx() >= td_->gammaBetaBlockDim) {
        return;
    }
    tempTensor = tempBuffer.Get<float>();
    cacheTensor0 = cacheBuffer0.Get<float>();
    cacheTensor1 = cacheBuffer1.Get<float>();

    int64_t mfactorMain = td_->gammaBetaBasicBlockLoop
                              ? td_->gammaBetaMfactor
                              : (td_->row == td_->gammaBetaMfactor ? td_->gammaBetaMfactor : td_->gammaBetaMtail);
    int64_t loopCount = td_->gammaBetaBasicBlockLoop ? td_->gammaBetaBasicBlockLoop : 1;
    for (int64_t ni = 0; ni < NTotalloop; ++ni) {
        Prologue();

        int64_t nfactor = (ni == NTotalloop - 1) ? Ntail : td_->gammaBetaNfactor;

        for (int64_t basicBlockIdx = 0; basicBlockIdx < loopCount; ++basicBlockIdx) {
            ProcessMainBlock(ni, basicBlockIdx, mfactorMain, nfactor);

            if (td_->gammaBetaBasicBlockLoop > 0 &&
                ((basicBlockIdx < td_->gammaBetaMainFoldCount) ||
                 (basicBlockIdx == td_->gammaBetaMainFoldCount && td_->gammaBetaMtail > 0))) {
                const int64_t mfactorFold =
                    (basicBlockIdx < td_->gammaBetaMainFoldCount) ? td_->gammaBetaMfactor : td_->gammaBetaMtail;
                ProcessFoldBlock(ni, basicBlockIdx, mfactorFold, nfactor);
            }

            ProcessSummation(ni, basicBlockIdx, mfactorMain, nfactor);
        }

        Epilogue(ni * td_->gammaBetaNfactor, nfactor);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::Prologue()
{
    // Prologue
    if (td_->pdbetaIsRequire) {
        beta_ = outQueueSum.template AllocTensor<PD_GAMMA_TYPE>();
    }

    if (td_->pdgammaIsRequire) {
        gamma_ = outQueueSum.template AllocTensor<PD_GAMMA_TYPE>();
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::Epilogue(const int64_t offset,
                                                                                             const int64_t extent)
{
    // Epilogue
    if (td_->pdbetaIsRequire) {
        CopyUB2UBWithCast<PD_GAMMA_TYPE>(beta_, cacheTensor0[td_->gammaBetaResultCacheID * td_->gammaBetaNfactor],
                                         extent);
        outQueueSum.EnQue(beta_);
        beta_ = outQueueSum.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(pdBetaOutTensorGM[offset], beta_, extent);
        outQueueSum.FreeTensor(beta_);
    }

    if (td_->pdgammaIsRequire) {
        CopyUB2UBWithCast<PD_GAMMA_TYPE>(gamma_, cacheTensor1[td_->gammaBetaResultCacheID * td_->gammaBetaNfactor],
                                         extent);
        outQueueSum.EnQue(gamma_);
        gamma_ = outQueueSum.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(pdGammaOutTensorGM[offset], gamma_, extent);
        outQueueSum.FreeTensor(gamma_);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::ProcessMainBlock(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessMainBlock
    int64_t offset = ni * td_->gammaBetaNfactor + basicBlockIdx * td_->gammaBetaMfactor * td_->col;
    dyMain_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyMain_, dyInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactor, td_->col);
        inQueueDy.EnQue(dyMain_);
        dyMain_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyMain_.ReinterpretCast<T>()[td_->gammaBetaNfactor];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactor, td_->col);
        inQueueDy.EnQue(dyMain_);
        dyMain_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyMain_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactor);
    }

    if (td_->pdgammaIsRequire) {
        xMain_ = inQueueX.template AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            CopyIn(xMain_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactor, td_->col);
            inQueueX.EnQue(xMain_);
            xMain_ = inQueueX.template DeQue<float>();
        } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            LocalTensor<T> castTempTensor = xMain_.ReinterpretCast<T>()[td_->gammaBetaNfactor];
            CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactor, td_->col);
            inQueueX.EnQue(xMain_);
            xMain_ = inQueueX.template DeQue<float>();
            CastToFp32From<T>(xMain_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactor);
        }

        offset = basicBlockIdx * td_->gammaBetaMfactor;
        var_ = inQueueParam.template AllocTensor<float>();
        CopyIn(var_, varInTensorGM[offset], mfactor);
        inQueueParam.EnQue(var_);
        var_ = inQueueParam.template DeQue<float>();

        mean_ = inQueueParam.template AllocTensor<float>();
        CopyIn(mean_, meanInTensorGM[offset], mfactor);
        inQueueParam.EnQue(mean_);
        mean_ = inQueueParam.template DeQue<float>();

        ComputeGamma(xMain_, dyMain_, xMain_, var_, mean_, mfactor, td_->gammaBetaNfactor);
        inQueueParam.FreeTensor(var_);
        inQueueParam.FreeTensor(mean_);
        if (!td_->pdbetaIsRequire) {
            inQueueDy.FreeTensor(dyMain_);
        }
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::ProcessFoldBlock(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessFoldBlock
    int64_t offset =
        ni * td_->gammaBetaNfactor + (basicBlockIdx + td_->gammaBetaBasicBlockLoop) * td_->gammaBetaMfactor * td_->col;
    LocalTensor<float> dyFold_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyFold_, dyInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactor, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyFold_.ReinterpretCast<T>()[td_->gammaBetaNfactor];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactor, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyFold_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactor);
    }
    if (td_->pdbetaIsRequire) {
        VectorAdd(dyMain_, dyMain_, dyFold_, mfactor, nfactor, td_->gammaBetaNfactor);
    }

    if (td_->pdgammaIsRequire) {
        LocalTensor<float> xFold_ = inQueueX.template AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            CopyIn(xFold_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactor, td_->col);
            inQueueX.EnQue(xFold_);
            xFold_ = inQueueX.template DeQue<float>();
        } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            LocalTensor<T> castTempTensor = xFold_.ReinterpretCast<T>()[td_->gammaBetaNfactor];
            CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactor, td_->col);
            inQueueX.EnQue(xFold_);
            xFold_ = inQueueX.template DeQue<float>();
            CastToFp32From<T>(xFold_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactor);
        }

        offset = (basicBlockIdx + td_->gammaBetaBasicBlockLoop) * td_->gammaBetaMfactor;
        LocalTensor<float> varFold_ = inQueueParam.template AllocTensor<float>();
        CopyIn(varFold_, varInTensorGM[offset], mfactor);
        inQueueParam.EnQue(varFold_);
        varFold_ = inQueueParam.template DeQue<float>();

        LocalTensor<float> meanFold_ = inQueueParam.template AllocTensor<float>();
        CopyIn(meanFold_, meanInTensorGM[offset], mfactor);
        inQueueParam.EnQue(meanFold_);
        meanFold_ = inQueueParam.template DeQue<float>();

        ComputeGamma(xFold_, dyFold_, xFold_, varFold_, meanFold_, mfactor, td_->gammaBetaNfactor);
        inQueueParam.FreeTensor(varFold_);
        inQueueParam.FreeTensor(meanFold_);
        inQueueDy.FreeTensor(dyFold_);
        VectorAdd(xMain_, xMain_, xFold_, mfactor, nfactor, td_->gammaBetaNfactor);
        inQueueX.FreeTensor(xFold_);
    } else {
        inQueueDy.FreeTensor(dyFold_);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::ProcessSummation(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessSummation
    int64_t cacheID = GetCacheID(basicBlockIdx);
    uint32_t srcShape[2] = {static_cast<uint32_t>(mfactor), static_cast<uint32_t>(td_->gammaBetaNfactor)};

    if (td_->pdbetaIsRequire) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tempTensor, dyMain_, srcShape, false);
        inQueueDy.FreeTensor(dyMain_);
        UpdateCache(cacheTensor0, tempTensor, cacheID, td_->gammaBetaNfactor, nfactor);
    }

    if (td_->pdgammaIsRequire) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tempTensor, xMain_, srcShape, false);
        inQueueX.FreeTensor(xMain_);
        UpdateCache(cacheTensor1, tempTensor, cacheID, td_->gammaBetaNfactor, nfactor);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradGroupedReduceBigNGammaBeta<T, PD_GAMMA_TYPE>::ComputeGamma(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& varTensor, const LocalTensor<float>& meanTensor, const int64_t rowSize,
    const int64_t colSize)
{
    // ComputeGamma
    int64_t colLength = colSize * sizeof(float);
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = CeilDiv(static_cast<int64_t>(colLength), static_cast<int64_t>(GetVRegSize()));
    uint32_t outerStride = td_->gammaBetaNfactor;
    uint32_t innerStride = static_cast<uint32_t>(GetVRegSize() / sizeof(float));
    float epsilonTmp = td_->epsilon;
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
            __local_mem__ float* x = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* dy = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* mean = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* var = (__local_mem__ float*)varTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(colSize);
            AscendC::MicroAPI::MaskReg pMask;
            pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                AscendC::MicroAPI::RegTensor<float> meanReg;
                AscendC::MicroAPI::RegTensor<float> varReg, rstdReg;
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, (__local_mem__ float*)mean + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(varReg, (__local_mem__ float*)var + i);
                CalRstdByHighPrecision(varReg, rstdReg, epsilonTmp);

                AscendC::MicroAPI::RegTensor<float> xReg;
                AscendC::MicroAPI::RegTensor<float> dyReg;
                DataCopy(xReg, (__local_mem__ float*)x + i * outerStride + 0 * innerStride);
                Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                DataCopy(dyReg, (__local_mem__ float*)dy + i * outerStride + 0 * innerStride);
                Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                DataCopy((__local_mem__ float*)dst + i * outerStride + 0 * innerStride, xReg, pMask);
            }
        }
    } else {
        __VEC_SCOPE__
        {
            __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
            __local_mem__ float* x = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* dy = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* mean = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* var = (__local_mem__ float*)varTensor.GetPhyAddr();
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                AscendC::MicroAPI::RegTensor<float> meanReg;
                AscendC::MicroAPI::RegTensor<float> varReg, rstdReg;
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, (__local_mem__ float*)mean + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(varReg, (__local_mem__ float*)var + i);
                CalRstdByHighPrecision(varReg, rstdReg, epsilonTmp);

                AscendC::MicroAPI::RegTensor<float> xReg;
                AscendC::MicroAPI::RegTensor<float> dyReg;
                AscendC::MicroAPI::MaskReg pMask;
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    DataCopy(xReg, (__local_mem__ float*)x + i * outerStride + j * innerStride);
                    Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    DataCopy(dyReg, (__local_mem__ float*)dy + i * outerStride + j * innerStride);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    DataCopy((__local_mem__ float*)dst + i * outerStride + j * innerStride, xReg, pMask);
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX, GM_ADDR workspace,
    const LayerNormGradTilingDataGroupedReduceBigN* tilingData, TPipe* pipeIn)
{
    td_ = tilingData;

    if (GetBlockIdx() >= td_->backwardBlockDim) {
        return;
    }

    startN_ = GetBlockIdx() * td_->backwardNPerBlock + Arith::Min(GetBlockIdx(), td_->backwardNRem);
    if (GetBlockIdx() < td_->backwardNRem) {
        nToProcess_ = td_->nToProcessMain;
        Nloop = td_->backwardNloopMain;
        Ntail = td_->backwardNtailMain;
        BasicBlockLoop = td_->backwardBasicBlockLoopMain;
        MainFoldCount = td_->backwardMainFoldCountMain;
        CacheBufferCount = td_->backwardCacheBufferCountMain;
        ResultCacheID = td_->backwardResultCacheIDMain;

    } else {
        nToProcess_ = td_->nToProcessTail;
        Nloop = td_->backwardNloopTail;
        Ntail = td_->backwardNtailTail;
        BasicBlockLoop = td_->backwardBasicBlockLoopTail;
        MainFoldCount = td_->backwardMainFoldCountTail;
        CacheBufferCount = td_->backwardCacheBufferCountTail;
        ResultCacheID = td_->backwardResultCacheIDTail;
    }

    // GM pointers
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x);
    varInTensorGM.SetGlobalBuffer((__gm__ float*)var);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean);
    gammaInTensorGM.SetGlobalBuffer((__gm__ U*)gamma);
    pdXOutTensorGM.SetGlobalBuffer((__gm__ T*)pdX);
    partialSum1GM.SetGlobalBuffer((__gm__ float*)workspace);
    partialSum2GM.SetGlobalBuffer((__gm__ float*)workspace + td_->backwardBlockDim * td_->row);
    finalSumGM.SetGlobalBuffer((__gm__ float*)workspace + 2 * td_->backwardBlockDim * td_->row);

    // Init Pipe
    pipe_ = pipeIn;
    // inQueueX 同时用作核间累加拷贝输入，按照backwardMfactorBlockAligned分配空间
    int64_t dyBufSize = td_->backwardMfactorBlockAligned * td_->backwardNfactorBlockAligned * sizeof(float);
    pipe_->InitBuffer(inQueueDy, 3, dyBufSize);
    pipe_->InitBuffer(inQueueX, 3, dyBufSize);
    pipe_->InitBuffer(outQueueDx, 2, dyBufSize);
    int64_t mFactorSize = td_->backwardMfactorBlockAligned * sizeof(float);
    pipe_->InitBuffer(inQueueParam, 4, mFactorSize);
    pipe_->InitBuffer(tempBuffer, mFactorSize);
    pipe_->InitBuffer(outTmpQueue0, 2, mFactorSize);
    pipe_->InitBuffer(outTmpQueue1, 2, mFactorSize);
    int64_t cacheBufSize = CacheBufferCount * mFactorSize;
    pipe_->InitBuffer(cacheBuffer0, cacheBufSize);
    pipe_->InitBuffer(cacheBuffer1, cacheBufSize);

    pipe_->InitBuffer(inQueueGamma, 2, td_->backwardNfactorBlockAligned * sizeof(float));
    pipe_->InitBuffer(reduceSumTempBuffer, td_->backwardMfactorBlockAligned * td_->backwardNfactorBlockAligned * sizeof(float));
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::Process()
{
    if (GetBlockIdx() < td_->backwardBlockDim) {
        tempTensor = tempBuffer.Get<float>();
        cacheTensor0 = cacheBuffer0.Get<float>();
        cacheTensor1 = cacheBuffer1.Get<float>();
        reduceSumTempTensor = reduceSumTempBuffer.Get<float>();
        sum1Tensor = cacheTensor0[ResultCacheID * td_->backwardMfactorBlockAligned];
        sum2Tensor = cacheTensor1[ResultCacheID * td_->backwardMfactorBlockAligned];

        int64_t mainNfactor =
            BasicBlockLoop ? td_->backwardNfactorBlockAligned
                           : (nToProcess_ == td_->backwardNfactorBlockAligned ? td_->backwardNfactorBlockAligned : Ntail);
        int64_t loopCount = BasicBlockLoop ? BasicBlockLoop : 1;

        for (int64_t mi = 0; mi < td_->backwardMTotalLoop; ++mi) {
            int64_t mFactor = (mi == td_->backwardMTotalLoop - 1) ? td_->backwardMtail : td_->backwardMfactor;

            Prologue(mi, mFactor);

            for (int64_t basicBlockIdx = 0; basicBlockIdx < loopCount; ++basicBlockIdx) {
                ProcessMainBlock(mi, basicBlockIdx, mFactor, mainNfactor);
                PRELOAD(2);  // 2*2K
                if (BasicBlockLoop && (basicBlockIdx < MainFoldCount || (basicBlockIdx == MainFoldCount && Ntail > 0))) {
                    int64_t foldNFactor = basicBlockIdx < MainFoldCount ? td_->backwardNfactorBlockAligned : Ntail;
                    ProcessFoldBlock(mi, basicBlockIdx, mFactor, foldNFactor);
                }
                PRELOAD(2);  // 2*2K
                ProcessSummation(mi, basicBlockIdx, mFactor, mainNfactor);
            }

            Epilogue();

            int64_t offset = GetBlockIdx() * td_->row + mi * td_->backwardMfactor;

            LocalTensor<float> outTensor0 = outTmpQueue0.template AllocTensor<float>();
            CopyUB2UB(outTensor0, sum1Tensor, mFactor);
            outTmpQueue0.EnQue(outTensor0);
            outTensor0 = outTmpQueue0.template DeQue<float>();
            CopyOut(partialSum1GM[offset], outTensor0, mFactor);
            outTmpQueue0.FreeTensor(outTensor0);

            LocalTensor<float> outTensor1 = outTmpQueue1.template AllocTensor<float>();
            CopyUB2UB(outTensor1, sum2Tensor, mFactor);
            outTmpQueue1.EnQue(outTensor1);
            outTensor1 = outTmpQueue1.template DeQue<float>();
            CopyOut(partialSum2GM[offset], outTensor1, mFactor);
            outTmpQueue1.FreeTensor(outTensor1);
        }
    }

    SyncAll();

    if (GetBlockIdx() >= td_->backwardBlockDim) {
        return;
    }

    PRELOAD(4);  // 4*2K
    for (int64_t mi = 0; mi < td_->backwardMTotalLoop; ++mi) {
        int64_t mfactor = mi == (td_->backwardMTotalLoop - 1) ? td_->backwardMtail : td_->backwardMfactor;
        PostPrologue(mi, mfactor);

        PRELOAD(2);  // 2*2K
        int64_t totalN = Nloop + (Ntail > 0 ? 1 : 0);
        for (int64_t ni = 0; ni < totalN; ++ni) {
            const int64_t nfactor = (ni < Nloop) ? td_->backwardNfactorBlockAligned : Ntail;
            ProcessX(mi, ni, mfactor, nfactor);
        }

        Epilogue();
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::Prologue(const int64_t mi, const int64_t mfactor)
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
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::ProcessMainBlock(const int64_t mi,
                                                                                        const int64_t basicBlockIdx,
                                                                                        const int64_t mfactor,
                                                                                        const int64_t nfactor)
{
    // Process Main Block
    int64_t offset = mi * td_->backwardMfactor * td_->col + basicBlockIdx * td_->backwardNfactorBlockAligned + startN_;
    sum1Main_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(sum1Main_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
        inQueueDy.EnQue(sum1Main_);
        sum1Main_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = sum1Main_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(sum1Main_);
        sum1Main_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(sum1Main_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }

    offset = basicBlockIdx * td_->backwardNfactorBlockAligned + startN_;
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
    NlastBroadcastMul(sum1Main_, sum1Main_, gammaMain_, mfactor, td_->backwardNfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaMain_);

    offset = mi * td_->backwardMfactor * td_->col + basicBlockIdx * td_->backwardNfactorBlockAligned + startN_;
    sum2Main_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(sum2Main_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = sum2Main_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(sum2Main_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }
    Normalize(sum2Main_, sum2Main_, mean_, var_, mfactor, td_->backwardNfactorBlockAligned, td_->epsilon);
    VectorMul(sum2Main_, sum2Main_, sum1Main_, mfactor * td_->backwardNfactorBlockAligned);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::ProcessFoldBlock(const int64_t mi,
                                                                                        const int64_t basicBlockIdx,
                                                                                        const int64_t mfactor,
                                                                                        const int64_t nfactor)
{
    // Process Fold Block
    int64_t offset = mi * td_->backwardMfactor * td_->col +
                     (basicBlockIdx + BasicBlockLoop) * td_->backwardNfactorBlockAligned + startN_;
    LocalTensor<float> dyFold_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyFold_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyFold_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyFold_, castTempTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }

    offset = (basicBlockIdx + BasicBlockLoop) * td_->backwardNfactorBlockAligned + startN_;
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

    NlastBroadcastMul(dyFold_, dyFold_, gammaFold_, mfactor, td_->backwardNfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaFold_);

    offset = mi * td_->backwardMfactor * td_->col +
             (basicBlockIdx + BasicBlockLoop) * td_->backwardNfactorBlockAligned + startN_;
    LocalTensor<float> xFold_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(xFold_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
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
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::ProcessSummation(const int64_t mi,
                                                                                        const int64_t basicBlockIdx,
                                                                                        const int64_t mfactor,
                                                                                        const int64_t nfactor)
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
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::ProcessX(const int64_t mi, const int64_t ni,
                                                                                const int64_t mfactor,
                                                                                const int64_t nfactor)
{
    // Process X
    int64_t offset = mi * td_->backwardMfactor * td_->col + ni * td_->backwardNfactorBlockAligned + startN_;
    LocalTensor<float> x_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(x_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
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
        CopyIn(dy_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
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
        CopyIn(gamma_.ReinterpretCast<U>(), gammaInTensorGM[ni * td_->backwardNfactorBlockAligned + startN_], nfactor);
        inQueueGamma.EnQue(gamma_);
        gamma_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> castTempTensor = gamma_.ReinterpretCast<U>()[td_->backwardNfactorBlockAligned];
        CopyIn(castTempTensor, gammaInTensorGM[ni * td_->backwardNfactorBlockAligned + startN_], nfactor);
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

template <typename T, typename U>
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::ComputeDx(
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
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::Epilogue()
{
    // Epilogue
    inQueueParam.FreeTensor(mean_);
    inQueueParam.FreeTensor(var_);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradGroupedReduceBigNBackward<T, U>::PostPrologue(const int64_t mi,
                                                                                    const int64_t mfactor)
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

    LocalTensor<float> tempSum1 = inQueueDy.template AllocTensor<float>();
    CopyIn(tempSum1, partialSum1GM[mi * td_->backwardMfactor], td_->backwardBlockDim, mfactor,
           td_->backwardMfactorBlockAligned, td_->row);
    inQueueDy.EnQue(tempSum1);
    tempSum1 = inQueueDy.template DeQue<float>();

    uint32_t srcShape[2] = {static_cast<uint32_t>(td_->backwardBlockDim),
                            static_cast<uint32_t>(td_->backwardMfactorBlockAligned)};

    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sum1Tensor, tempSum1, srcShape, false);
    inQueueDy.FreeTensor(tempSum1);

    LocalTensor<float> tempSum2 = inQueueX.template AllocTensor<float>();
    CopyIn(tempSum2, partialSum2GM[mi * td_->backwardMfactor], td_->backwardBlockDim, mfactor,
           td_->backwardMfactorBlockAligned, td_->row);
    inQueueX.EnQue(tempSum2);
    tempSum2 = inQueueX.template DeQue<float>();

    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sum2Tensor, tempSum2, srcShape, false);
    inQueueX.FreeTensor(tempSum2);
}

}  // namespace LayerNormGrad

#endif  // LAYER_NORM_GRAD_GROUPED_REDUCE_BIG_N_IMPL_
