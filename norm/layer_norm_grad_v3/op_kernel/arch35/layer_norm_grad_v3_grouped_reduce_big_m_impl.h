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
 * \file layer_norm_grad_v3_grouped_reduce_big_m_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_GROUPED_REDUCE_BIG_M_IMPL_
#define LAYER_NORM_GRAD_V3_GROUPED_REDUCE_BIG_M_IMPL_
#include "layer_norm_grad_v3_api.h"
#include "layer_norm_grad_v3_base.h"
#include "layer_norm_grad_v3_grouped_reduce.h"

namespace LayerNormGradV3
{
using namespace AscendC;

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
    const LayerNormGradV3TilingDataGroupedReduceBigM* tilingData, TPipe* pipeIn)
{
    td_ = tilingData;

    if (GetBlockIdx() >= td_->gammaBetaUsableBlocks) {
        return;
    }

    savedWorkspaceAddr = workspace;
    savedPdGammaAddr = pdGamma;
    savedPdBetaAddr = pdBeta;

    const int64_t startM =
        (GetBlockIdx() * td_->gammaBetaMPerBlock) + Arith::Min(GetBlockIdx(), td_->gammaBetaMReminder);

    if (GetBlockIdx() < td_->gammaBetaMReminder) {
        M = td_->gammabetaMToProcessMainBlock;
        Mloop = td_->gammabetaMLoopMainBlock;
        MTotalLoop = td_->gammabetaMTotalLoopMainBlock;
        Mtail = td_->gammabetaMTailMainBlock;
        BasicBlockLoop = td_->gammabetaBasicBlockLoopMainBlock;
        MainFoldCount = td_->gammabetaMainFoldCountMainBlock;
        CacheBufferCount = td_->gammabetaCacheBufferCountMainBlock;
        ResultCacheID = td_->gammabetaResultCacheIDMainBlock;
    } else {
        M = td_->gammabetaMToProcessTailBlock;
        Mloop = td_->gammabetaMLoopTailBlock;
        MTotalLoop = td_->gammabetaMTotalLoopTailBlock;
        Mtail = td_->gammabetaMTailTailBlock;
        BasicBlockLoop = td_->gammabetaBasicBlockLoopTailBlock;
        MainFoldCount = td_->gammabetaMainFoldCountTailBlock;
        CacheBufferCount = td_->gammabetaCacheBufferCountTailBlock;
        ResultCacheID = td_->gammabetaResultCacheIDTailBlock;
    }

    // Init GM Tensor
    int64_t colOffset = GetBlockIdx() * td_->col;
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + startM * td_->col);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + startM * td_->col);
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd + startM);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + startM);
    pdGammaOutTensorGMStag1.SetGlobalBuffer((__gm__ float*)workspace + colOffset);
    pdBetaOutTensorGMStag1.SetGlobalBuffer((__gm__ float*)workspace + td_->gammaBetaUsableBlocks * td_->col +
                                      colOffset);

    // Init Pipe
    pipe_ = pipeIn;
    int64_t dyBufSize = td_->gammaBetaMfactorBlockAligned * td_->gammaBetaNfactorBlockAligned * sizeof(float);
    pipe_->InitBuffer(inQueueDy, 3, dyBufSize);
    pipe_->InitBuffer(inQueueX, 3, dyBufSize);
    pipe_->InitBuffer(inQueueParam, 2, td_->gammaBetaMfactorBlockAligned * sizeof(float));
    int64_t nFactorAlignedBufSize = td_->gammaBetaNfactorBlockAligned * sizeof(float);
    pipe_->InitBuffer(outQueueSum, 2, nFactorAlignedBufSize);
    pipe_->InitBuffer(tempBuffer, nFactorAlignedBufSize);
    int64_t cacheBufSize = CacheBufferCount * nFactorAlignedBufSize;
    pipe_->InitBuffer(cacheBuffer0, cacheBufSize);
    pipe_->InitBuffer(cacheBuffer1, cacheBufSize);
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::Process()
{
    if (GetBlockIdx() < td_->gammaBetaUsableBlocks) {
        // stage 1: 计算核内累加
        tempTensor = tempBuffer.Get<float>();
        cacheTensor0 = cacheBuffer0.Get<float>();
        cacheTensor1 = cacheBuffer1.Get<float>();

        int64_t totalRounds = td_->gammaBetaNloop + (td_->gammaBetaNtail > 0 ? 1 : 0);

        int64_t mfactor = BasicBlockLoop
                              ? td_->gammaBetaMfactorBlockAligned
                              : (M == td_->gammaBetaMfactorBlockAligned ? td_->gammaBetaMfactorBlockAligned : Mtail);
        int64_t loopCnt = BasicBlockLoop ? BasicBlockLoop : 1;

        for (int64_t round = 0; round < totalRounds; ++round) {
            int64_t ni = (round < td_->gammaBetaNloop) ? round : td_->gammaBetaNloop;
            int64_t nfactor = (round < td_->gammaBetaNloop) ? td_->gammaBetaNfactorBlockAligned : td_->gammaBetaNtail;

            PrologueStag1();

            for (int64_t i = 0; i < loopCnt; ++i) {
                ProcessMainBlock(ni, i, mfactor, nfactor);
                if (BasicBlockLoop != 0 && ((i < MainFoldCount) || (i == MainFoldCount && Mtail > 0))) {
                    ProcessFoldBlock(ni, i, (i < MainFoldCount) ? td_->gammaBetaMfactorBlockAligned : Mtail, nfactor);
                }
                ProcessSummation(ni, i, mfactor, td_->gammaBetaNfactorBlockAligned);
            }

            int64_t epilogueBase = (round < td_->gammaBetaNloop)
                                       ? (round * td_->gammaBetaNfactorBlockAligned)
                                       : (td_->gammaBetaNloop * td_->gammaBetaNfactorBlockAligned);
            EpilogueStag1(epilogueBase, nfactor);
        }
    }

    SyncAll();

    if (GetBlockIdx() != 0) {
        return;
    }

    // stage 2: 计算核间累加
    PRELOAD(4);
    postXInTensorGM.SetGlobalBuffer((__gm__ float*)savedWorkspaceAddr + GetBlockIdx() * td_->col);
    postDyInTensorGM.SetGlobalBuffer((__gm__ float*)savedWorkspaceAddr + td_->gammaBetaUsableBlocks * td_->col +
                                     GetBlockIdx() * td_->col);
    pdGammaOutTensorGMStag2.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)savedPdGammaAddr);
    pdBetaOutTensorGMStag2.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)savedPdBetaAddr);

    M = td_->gammaBetaUsableBlocks;
    Mtail = td_->gammaBetaMTailStg2;
    BasicBlockLoop = td_->gammaBetaMBasicBlockLoopStg2;
    MainFoldCount = td_->gammaBetaMMainFoldCountStg2;
    ResultCacheID = td_->gammaBetaMResultCacheIDStg2;

    int64_t totalRounds = td_->gammaBetaNloop + (td_->gammaBetaNtail > 0 ? 1 : 0);
    int64_t mfactor = BasicBlockLoop
                  ? td_->gammaBetaMfactorBlockAligned
                  : (td_->gammaBetaUsableBlocks == td_->gammaBetaMfactorBlockAligned ? td_->gammaBetaMfactorBlockAligned
                                                                                     : Mtail);
    int64_t loopCnt = BasicBlockLoop ? BasicBlockLoop : 1;
    for (int64_t round = 0; round < totalRounds; ++round) {
        int64_t ni = (round < td_->gammaBetaNloop) ? round : td_->gammaBetaNloop;
        int64_t nfactor = (round < td_->gammaBetaNloop) ? td_->gammaBetaNfactorBlockAligned : td_->gammaBetaNtail;

        PrologueStag2();

        for (int64_t i = 0; i < loopCnt; ++i) {
            PostProcessMainBlock(ni, i, mfactor, nfactor);
            if (BasicBlockLoop && ((i < MainFoldCount) || (i == MainFoldCount && Mtail > 0))) {
                PostProcessFoldBlock(ni, i, (i < MainFoldCount) ? td_->gammaBetaMfactorBlockAligned : Mtail, nfactor);
            }
            PRELOAD(2);
            ProcessSummation(ni, i, mfactor, td_->gammaBetaNfactorBlockAligned);
        }

        EpilogueStag2(ni * td_->gammaBetaNfactorBlockAligned, nfactor);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::PrologueStag1()
{
    if (td_->pdbetaIsRequire) {
        betaStag1 = outQueueSum.template AllocTensor<float>();
    }

    if (td_->pdgammaIsRequire) {
        gammaStag1 = outQueueSum.template AllocTensor<float>();
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::PrologueStag2()
{
    if (td_->pdbetaIsRequire) {
        betaStag2 = outQueueSum.template AllocTensor<PD_GAMMA_TYPE>();
    }

    if (td_->pdgammaIsRequire) {
        gammaStag2 = outQueueSum.template AllocTensor<PD_GAMMA_TYPE>();
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::EpilogueStag1(const int64_t offset,
                                                                                                    const int64_t extent)
{
    if (td_->pdbetaIsRequire) {
        CopyUB2UBWithCast<float>(betaStag1, cacheTensor0[ResultCacheID * td_->gammaBetaNfactorBlockAligned],
                                         extent);
        outQueueSum.EnQue(betaStag1);
        betaStag1 = outQueueSum.template DeQue<float>();
        CopyOut<float>(pdBetaOutTensorGMStag1[offset], betaStag1, extent);
        outQueueSum.FreeTensor(betaStag1);
    }

    if (td_->pdgammaIsRequire) {
        CopyUB2UBWithCast<float>(gammaStag1, cacheTensor1[ResultCacheID * td_->gammaBetaNfactorBlockAligned],
                                         extent);
        outQueueSum.EnQue(gammaStag1);
        gammaStag1 = outQueueSum.template DeQue<float>();
        CopyOut<float>(pdGammaOutTensorGMStag1[offset], gammaStag1, extent);
        outQueueSum.FreeTensor(gammaStag1);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::EpilogueStag2(const int64_t offset,
                                                                                                    const int64_t extent)
{
    if (td_->pdbetaIsRequire) {
        CopyUB2UBWithCast<PD_GAMMA_TYPE>(betaStag2, cacheTensor0[ResultCacheID * td_->gammaBetaNfactorBlockAligned],
                                         extent);
        outQueueSum.EnQue(betaStag2);
        betaStag2 = outQueueSum.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(pdBetaOutTensorGMStag2[offset], betaStag2, extent);
        outQueueSum.FreeTensor(betaStag2);
    }

    if (td_->pdgammaIsRequire) {
        CopyUB2UBWithCast<PD_GAMMA_TYPE>(gammaStag2, cacheTensor1[ResultCacheID * td_->gammaBetaNfactorBlockAligned],
                                         extent);
        outQueueSum.EnQue(gammaStag2);
        gammaStag2 = outQueueSum.template DeQue<PD_GAMMA_TYPE>();
        CopyOut<PD_GAMMA_TYPE>(pdGammaOutTensorGMStag2[offset], gammaStag2, extent);
        outQueueSum.FreeTensor(gammaStag2);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::ProcessMainBlock(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessMainBlock
    int64_t offset =
        ni * td_->gammaBetaNfactorBlockAligned + basicBlockIdx * td_->gammaBetaMfactorBlockAligned * td_->col;
    dyMain_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyMain_, dyInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyMain_);
        dyMain_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyMain_.ReinterpretCast<T>()[td_->gammaBetaNfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyMain_);
        dyMain_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyMain_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
    }

    if (td_->pdgammaIsRequire) {
        xMain_ = inQueueX.template AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            CopyIn(xMain_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor,
                   td_->gammaBetaNfactorBlockAligned, td_->col);
            inQueueX.EnQue(xMain_);
            xMain_ = inQueueX.template DeQue<float>();
        } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            LocalTensor<T> castTempTensor = xMain_.ReinterpretCast<T>()[td_->gammaBetaNfactorBlockAligned];
            CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactorBlockAligned,
                   td_->col);
            inQueueX.EnQue(xMain_);
            xMain_ = inQueueX.template DeQue<float>();
            CastToFp32From<T>(xMain_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
        }

        offset = basicBlockIdx * td_->gammaBetaMfactorBlockAligned;
        rstd_ = inQueueParam.template AllocTensor<float>();
        CopyIn(rstd_, rstdInTensorGM[offset], mfactor);
        inQueueParam.EnQue(rstd_);
        rstd_ = inQueueParam.template DeQue<float>();

        mean_ = inQueueParam.template AllocTensor<float>();
        CopyIn(mean_, meanInTensorGM[offset], mfactor);
        inQueueParam.EnQue(mean_);
        mean_ = inQueueParam.template DeQue<float>();

        ComputeGamma(xMain_, dyMain_, xMain_, rstd_, mean_, mfactor, td_->gammaBetaNfactorBlockAligned);
        inQueueParam.FreeTensor(rstd_);
        inQueueParam.FreeTensor(mean_);
        if (!td_->pdbetaIsRequire) {
            inQueueDy.FreeTensor(dyMain_);
        }
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::ProcessFoldBlock(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessFoldBlock
    int64_t offset = ni * td_->gammaBetaNfactorBlockAligned +
                     (basicBlockIdx + BasicBlockLoop) * td_->gammaBetaMfactorBlockAligned * td_->col;
    LocalTensor<float> dyFold_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyFold_, dyInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyFold_.ReinterpretCast<T>()[td_->gammaBetaNfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyFold_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
    }
    if (td_->pdbetaIsRequire) {
        VectorAdd(dyMain_, dyMain_, dyFold_, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
    }

    if (td_->pdgammaIsRequire) {
        LocalTensor<float> xFold_ = inQueueX.template AllocTensor<float>();
        if constexpr (IsSameType<T, float>::value) {
            CopyIn(xFold_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor,
                   td_->gammaBetaNfactorBlockAligned, td_->col);
            inQueueX.EnQue(xFold_);
            xFold_ = inQueueX.template DeQue<float>();
        } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
            LocalTensor<T> castTempTensor = xFold_.ReinterpretCast<T>()[td_->gammaBetaNfactorBlockAligned];
            CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->gammaBetaNfactorBlockAligned,
                   td_->col);
            inQueueX.EnQue(xFold_);
            xFold_ = inQueueX.template DeQue<float>();
            CastToFp32From<T>(xFold_, castTempTensor, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
        }

        offset = (basicBlockIdx + BasicBlockLoop) * td_->gammaBetaMfactorBlockAligned;
        LocalTensor<float> rstdFold_ = inQueueParam.template AllocTensor<float>();
        CopyIn(rstdFold_, rstdInTensorGM[offset], mfactor);
        inQueueParam.EnQue(rstdFold_);
        rstdFold_ = inQueueParam.template DeQue<float>();

        LocalTensor<float> meanFold_ = inQueueParam.template AllocTensor<float>();
        CopyIn(meanFold_, meanInTensorGM[offset], mfactor);
        inQueueParam.EnQue(meanFold_);
        meanFold_ = inQueueParam.template DeQue<float>();

        ComputeGamma(xFold_, dyFold_, xFold_, rstdFold_, meanFold_, mfactor, td_->gammaBetaNfactorBlockAligned);
        inQueueParam.FreeTensor(rstdFold_);
        inQueueParam.FreeTensor(meanFold_);
        inQueueDy.FreeTensor(dyFold_);
        VectorAdd(xMain_, xMain_, xFold_, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
        inQueueX.FreeTensor(xFold_);
    } else {
        inQueueDy.FreeTensor(dyFold_);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::ProcessSummation(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessSummation
    int64_t cacheID = GetCacheID(basicBlockIdx);
    uint32_t srcShape[2] = {static_cast<uint32_t>(mfactor), static_cast<uint32_t>(nfactor)};
    if (td_->pdbetaIsRequire) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tempTensor, dyMain_, srcShape, false);
        inQueueDy.FreeTensor(dyMain_);
        UpdateCache(cacheTensor0, tempTensor, cacheID, td_->gammaBetaNfactorBlockAligned, nfactor);
    }

    if (td_->pdgammaIsRequire) {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(tempTensor, xMain_, srcShape, false);
        inQueueX.FreeTensor(xMain_);
        UpdateCache(cacheTensor1, tempTensor, cacheID, td_->gammaBetaNfactorBlockAligned, nfactor);
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::ComputeGamma(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& rstdTensor, const LocalTensor<float>& meanTensor, const int64_t rowSize,
    const int64_t colSize)
{
    // ComputeGamma
    int64_t colLength = colSize * sizeof(float);
    uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
    uint16_t innerLoopTimes = CeilDiv(static_cast<int64_t>(colLength), static_cast<int64_t>(GetVRegSize()));
    uint32_t outerStride = td_->gammaBetaNfactorBlockAligned;
    uint32_t innerStride = static_cast<uint32_t>(GetVRegSize() / sizeof(float));
    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
            __local_mem__ float* x = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* dy = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* mean = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* rstd = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            uint32_t count = static_cast<uint32_t>(colSize);
            AscendC::MicroAPI::MaskReg pMask;
            pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                AscendC::MicroAPI::RegTensor<float> meanReg;
                AscendC::MicroAPI::RegTensor<float> rstdReg;
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, (__local_mem__ float*)mean + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg, (__local_mem__ float*)rstd + i);

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
            __local_mem__ float* rstd = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                AscendC::MicroAPI::RegTensor<float> meanReg;
                AscendC::MicroAPI::RegTensor<float> rstdReg;
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, (__local_mem__ float*)mean + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg, (__local_mem__ float*)rstd + i);

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

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::PostProcessMainBlock(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessMainBlock
    int64_t offset =
        ni * td_->gammaBetaNfactorBlockAligned + basicBlockIdx * td_->gammaBetaMfactorBlockAligned * td_->col;
    if (td_->pdbetaIsRequire) {
        dyMain_ = inQueueDy.template AllocTensor<float>();
        CopyIn(dyMain_, postDyInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyMain_);
        dyMain_ = inQueueDy.template DeQue<float>();
    }

    if (td_->pdgammaIsRequire) {
        xMain_ = inQueueX.template AllocTensor<float>();
        CopyIn(xMain_, postXInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueX.EnQue(xMain_);
        xMain_ = inQueueX.template DeQue<float>();
    }
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMGammaBeta<T, PD_GAMMA_TYPE>::PostProcessFoldBlock(
    const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // ProcessFoldBlock
    int64_t offset = ni * td_->gammaBetaNfactorBlockAligned +
                     (basicBlockIdx + BasicBlockLoop) * td_->gammaBetaMfactorBlockAligned * td_->col;
    if (td_->pdbetaIsRequire) {
        LocalTensor<float> dyFold_ = inQueueDy.template AllocTensor<float>();
        CopyIn(dyFold_, postDyInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
        VectorAdd(dyMain_, dyMain_, dyFold_, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
        inQueueDy.FreeTensor(dyFold_);
    }

    if (td_->pdgammaIsRequire) {
        LocalTensor<float> xFold_ = inQueueX.template AllocTensor<float>();
        CopyIn(xFold_, postXInTensorGM[offset], mfactor, nfactor, td_->gammaBetaNfactorBlockAligned, td_->col);
        inQueueX.EnQue(xFold_);
        xFold_ = inQueueX.template DeQue<float>();
        VectorAdd(xMain_, xMain_, xFold_, mfactor, nfactor, td_->gammaBetaNfactorBlockAligned);
        inQueueX.FreeTensor(xFold_);
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX, GM_ADDR workspace,
    const LayerNormGradV3TilingDataGroupedReduceBigM* tilingData, TPipe* pipeIn)
{
    constexpr static int64_t SINGLE_CORE_REDUCE_THRESHOLD = 64;
    td_ = tilingData;
    M = td_->row;
    N = td_->col;

    backwardMainBlockFactor = Arith::CeilDiv(M, GetBlockNum());
    backwardBlockDim = Arith::CeilDiv(M, backwardMainBlockFactor);

    if (GetBlockIdx() >= backwardBlockDim) {
        return;
    }

    MainBlockCount = FloorDiv(M, backwardMainBlockFactor);
    TailBlockCount = backwardBlockDim - MainBlockCount;
    TailBlockFactor = M - MainBlockCount * backwardMainBlockFactor;

    if (GetBlockIdx() < MainBlockCount) {
        Mloop = Arith::FloorDiv(backwardMainBlockFactor, backwardMfactor);
        Mtail = backwardMainBlockFactor - Mloop * backwardMfactor;
        CurrentBlockFactor = backwardMainBlockFactor;
    } else {
        Mloop = Arith::FloorDiv(TailBlockFactor, backwardMfactor);
        Mtail = TailBlockFactor - Mloop * backwardMfactor;
        CurrentBlockFactor = TailBlockFactor;
    }
    Nloop = Arith::FloorDiv(N, backwardNfactor);
    NTotalLoop = Arith::CeilDiv(N, backwardNfactor);
    Ntail = N - Nloop * backwardNfactor;
    BasicBlockLoop = FindNearestPower2(NTotalLoop);
    MainFoldCount = Nloop - BasicBlockLoop;

    NfactorBlockAligned =
        Aligned(static_cast<int64_t>(backwardNfactor * sizeof(float)), static_cast<int64_t>(GetUbBlockSize())) /
        sizeof(float);
    MfactorBlockAligned =
        Aligned(static_cast<int64_t>(backwardMfactor * sizeof(float)), static_cast<int64_t>(GetUbBlockSize())) /
        sizeof(float);

    if (BasicBlockLoop == 0) {
        CacheBufferCount = 1;
        ResultCacheID = 0;
    } else {
        CacheBufferCount = 64 - ScalarCountLeadingZero(BasicBlockLoop);
        ResultCacheID = GetCacheID(BasicBlockLoop - 1);
    }

    // Init GM Tensor
    int64_t colOffset = GetBlockIdx() * backwardMainBlockFactor * N;
    int64_t dyGmLen = CurrentBlockFactor * N;
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + colOffset, dyGmLen);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + colOffset, dyGmLen);
    pdXOutTensorGM.SetGlobalBuffer((__gm__ T*)pdX + colOffset, dyGmLen);

    int64_t rstdOffset = GetBlockIdx() * backwardMainBlockFactor;
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd + rstdOffset, CurrentBlockFactor);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + rstdOffset, CurrentBlockFactor);
    gammaInTensorGM.SetGlobalBuffer((__gm__ U*)gamma, N);

    pipe_ = pipeIn;
    int64_t dyBufSize = MfactorBlockAligned * NfactorBlockAligned * sizeof(float);
    pipe_->InitBuffer(inQueueDy, 3, dyBufSize);
    pipe_->InitBuffer(inQueueX, 3, dyBufSize);
    pipe_->InitBuffer(outQueueDx, 2, dyBufSize);
    pipe_->InitBuffer(reduceSumTempBuffer, dyBufSize);
    int64_t mFactorAlignedSize = MfactorBlockAligned * sizeof(float);
    pipe_->InitBuffer(inQueueParam, 4, mFactorAlignedSize);
    pipe_->InitBuffer(tempBuffer, mFactorAlignedSize);
    int64_t cacheBufSize = CacheBufferCount * MfactorBlockAligned * sizeof(float);
    pipe_->InitBuffer(cacheBuffer0, cacheBufSize);
    pipe_->InitBuffer(cacheBuffer1, cacheBufSize);
    pipe_->InitBuffer(inQueueGamma, 2, NfactorBlockAligned * sizeof(float));
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::Process()

{
    if (GetBlockIdx() >= backwardBlockDim) {
        return;
    }
    tempTensor = tempBuffer.Get<float>();
    cacheTensor0 = cacheBuffer0.Get<float>();
    cacheTensor1 = cacheBuffer1.Get<float>();
    reduceSumTempTensor = reduceSumTempBuffer.Get<float>();
    sum1Tensor = cacheTensor0[ResultCacheID * MfactorBlockAligned];
    sum2Tensor = cacheTensor1[ResultCacheID * MfactorBlockAligned];

    int64_t totalMRounds = Mloop + (Mtail > 0 ? 1 : 0);

    int64_t loopCnt = BasicBlockLoop ? BasicBlockLoop : 1;
    int64_t nfactor =
        BasicBlockLoop
            ? NfactorBlockAligned
            : (td_->col == NfactorBlockAligned ? NfactorBlockAligned : Ntail);

    for (int64_t mi = 0; mi < totalMRounds; ++mi) {
        int64_t mfactor = (mi < Mloop) ? MfactorBlockAligned : Mtail;

        Prologue(mi, mfactor);

        for (int64_t i = 0; i < loopCnt; ++i) {
            ProcessMainBlock(mi, i, mfactor, nfactor);

            if (BasicBlockLoop &&
                ((i < MainFoldCount) || (i == MainFoldCount && Ntail > 0))) {
                int64_t foldNfactor =
                    (i < MainFoldCount) ? NfactorBlockAligned : Ntail;
                ProcessFoldBlock(mi, i, mfactor, foldNfactor);
            }

            ProcessSummation(mi, i, mfactor, nfactor);
        }

        int64_t xTotalLoop = Nloop + (Ntail > 0 ? 1 : 0);
        for (int64_t ni = 0; ni < xTotalLoop; ++ni) {
            int64_t xNfactor = (ni < Nloop) ? NfactorBlockAligned : Ntail;
            ProcessX(mi, ni, mfactor, xNfactor);
        }
        Epilogue();
    }
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::Prologue(const int64_t mi, const int64_t mfactor)
{
    // Prologue
    int64_t offset = mi * backwardMfactor;
    mean_ = inQueueParam.template AllocTensor<float>();
    CopyIn(mean_, meanInTensorGM[offset], mfactor);
    inQueueParam.EnQue(mean_);
    mean_ = inQueueParam.template DeQue<float>();

    rstd_ = inQueueParam.template AllocTensor<float>();
    CopyIn(rstd_, rstdInTensorGM[offset], mfactor);
    inQueueParam.EnQue(rstd_);
    rstd_ = inQueueParam.template DeQue<float>();
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::ProcessMainBlock(
    const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // Process Main Block
    int64_t offset = mi * backwardMfactor * N + basicBlockIdx * backwardNfactor;
    sum1Main_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(sum1Main_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, NfactorBlockAligned, N);
        inQueueDy.EnQue(sum1Main_);
        sum1Main_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = sum1Main_.ReinterpretCast<T>()[NfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * NfactorBlockAligned, N);
        inQueueDy.EnQue(sum1Main_);
        sum1Main_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(sum1Main_, castTempTensor, mfactor, nfactor, NfactorBlockAligned);
    }

    offset = basicBlockIdx * backwardNfactor;
    LocalTensor<float> gammaMain_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gammaMain_.ReinterpretCast<U>(), gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> castTempTensor = gammaMain_.ReinterpretCast<U>()[NfactorBlockAligned];
        CopyIn(castTempTensor, gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gammaMain_, castTempTensor, nfactor);
    }
    NlastBroadcastMul(sum1Main_, sum1Main_, gammaMain_, mfactor, NfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaMain_);

    offset = mi * backwardMfactor * N + basicBlockIdx * backwardNfactor;
    sum2Main_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(sum2Main_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, NfactorBlockAligned, N);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = sum2Main_.ReinterpretCast<T>()[NfactorBlockAligned];
        CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * NfactorBlockAligned, N);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(sum2Main_, castTempTensor, mfactor, nfactor, NfactorBlockAligned);
    }
    Normalize(sum2Main_, sum2Main_, mean_, rstd_, mfactor, NfactorBlockAligned);
    VectorMul(sum2Main_, sum2Main_, sum1Main_, mfactor * NfactorBlockAligned);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::ProcessFoldBlock(
    const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // Process Fold Block
    int64_t offset = mi * backwardMfactor * N + (basicBlockIdx + BasicBlockLoop) * backwardNfactor;
    LocalTensor<float> dyFold_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dyFold_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, NfactorBlockAligned, N);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dyFold_.ReinterpretCast<T>()[NfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * NfactorBlockAligned, N);
        inQueueDy.EnQue(dyFold_);
        dyFold_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dyFold_, castTempTensor, mfactor, nfactor, NfactorBlockAligned);
    }

    offset = (basicBlockIdx + BasicBlockLoop) * backwardNfactor;
    LocalTensor<float> gammaFold_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gammaFold_.ReinterpretCast<U>(), gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaFold_);
        gammaFold_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> castTempTensor = gammaFold_.ReinterpretCast<U>()[NfactorBlockAligned];
        CopyIn(castTempTensor, gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaFold_);
        gammaFold_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gammaFold_, castTempTensor, nfactor);
    }

    NlastBroadcastMul(dyFold_, dyFold_, gammaFold_, mfactor, NfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaFold_);

    offset = mi * backwardMfactor * N + (basicBlockIdx + BasicBlockLoop) * backwardNfactor;
    LocalTensor<float> xFold_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(xFold_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, NfactorBlockAligned, N);
        inQueueX.EnQue(xFold_);
        xFold_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = xFold_.ReinterpretCast<T>()[NfactorBlockAligned];
        CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * NfactorBlockAligned, N);
        inQueueX.EnQue(xFold_);
        xFold_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(xFold_, castTempTensor, mfactor, nfactor, NfactorBlockAligned);
    }
    Normalize(xFold_, xFold_, mean_, rstd_, mfactor, NfactorBlockAligned);
    VectorMul(xFold_, xFold_, dyFold_, mfactor * NfactorBlockAligned);

    VectorAdd(sum1Main_, sum1Main_, dyFold_, mfactor, nfactor, NfactorBlockAligned);
    inQueueDy.FreeTensor(dyFold_);
    VectorAdd(sum2Main_, sum2Main_, xFold_, mfactor, nfactor, NfactorBlockAligned);
    inQueueX.FreeTensor(xFold_);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::ProcessSummation(
    const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
{
    // Process Summation
    int64_t cacheID = GetCacheID(basicBlockIdx);
    LastReduceSum(tempTensor, sum1Main_, reduceSumTempTensor, mfactor, nfactor, NfactorBlockAligned);
    inQueueDy.FreeTensor(sum1Main_);
    UpdateCache(cacheTensor0, tempTensor, cacheID, MfactorBlockAligned, mfactor);
    LastReduceSum(tempTensor, sum2Main_, reduceSumTempTensor, mfactor, nfactor, NfactorBlockAligned);
    inQueueX.FreeTensor(sum2Main_);
    UpdateCache(cacheTensor1, tempTensor, cacheID, MfactorBlockAligned, mfactor);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::ProcessX(
    const int64_t mi, const int64_t ni, const int64_t mfactor, const int64_t nfactor)
{
    // Process X
    int64_t offset = mi * backwardMfactor * N + ni * backwardNfactor;
    LocalTensor<float> x_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(x_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, NfactorBlockAligned, N);
        inQueueX.EnQue(x_);
        x_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = x_.ReinterpretCast<T>()[NfactorBlockAligned];
        CopyIn(castTempTensor, xInTensorGM[offset], mfactor, nfactor, 2 * NfactorBlockAligned, N);
        inQueueX.EnQue(x_);
        x_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(x_, castTempTensor, mfactor, nfactor, NfactorBlockAligned);
    }
    Normalize(x_, x_, mean_, rstd_, mfactor, NfactorBlockAligned);

    LocalTensor<float> dy_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dy_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, NfactorBlockAligned, N);
        inQueueDy.EnQue(dy_);
        dy_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> castTempTensor = dy_.ReinterpretCast<T>()[NfactorBlockAligned];
        CopyIn(castTempTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * NfactorBlockAligned, N);
        inQueueDy.EnQue(dy_);
        dy_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dy_, castTempTensor, mfactor, nfactor, NfactorBlockAligned);
    }

    LocalTensor<float> gamma_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gamma_.ReinterpretCast<U>(), gammaInTensorGM[ni * backwardNfactor], nfactor);
        inQueueGamma.EnQue(gamma_);
        gamma_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> castTempTensor = gamma_.ReinterpretCast<U>()[NfactorBlockAligned];
        CopyIn(castTempTensor, gammaInTensorGM[ni * backwardNfactor], nfactor);
        inQueueGamma.EnQue(gamma_);
        gamma_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gamma_, castTempTensor, nfactor);
    }

    LocalTensor<T> dx_ = outQueueDx.template AllocTensor<T>();
    ComputeDx(dx_, dy_, x_, gamma_, sum1Tensor, sum2Tensor, rstd_, mfactor, nfactor, NfactorBlockAligned);
    inQueueDy.FreeTensor(dy_);
    inQueueX.FreeTensor(x_);
    inQueueGamma.FreeTensor(gamma_);

    outQueueDx.EnQue(dx_);
    dx_ = outQueueDx.template DeQue<T>();
    CopyOut(pdXOutTensorGM[offset], dx_, mfactor, nfactor, N, NfactorBlockAligned);
    outQueueDx.FreeTensor(dx_);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::ComputeDx(
    const LocalTensor<T>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
    const LocalTensor<float>& gammaTensor, const LocalTensor<float>& sum1Tensor, const LocalTensor<float>& sum2Tensor,
    const LocalTensor<float>& rstdTensor, const int64_t rowSize, const int64_t colSize, const int64_t stride)
{
    // Compute Dx
    constexpr static uint32_t VL = GetVRegSize() / sizeof(float);
    uint16_t outerLoopTimes = rowSize;
    uint16_t innerLoopTimes =
        CeilDiv(static_cast<int64_t>(colSize * sizeof(float)), static_cast<int64_t>(GetVRegSize()));
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL;
    float floatN = static_cast<float>(N);
    float reciprocalN = static_cast<float>(1) / floatN;

    if (innerLoopTimes == 1) {
        __VEC_SCOPE__
        {
            __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
            __local_mem__ float* dy = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* x = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* gamma = (__local_mem__ float*)gammaTensor.GetPhyAddr();
            __local_mem__ float* sum1 = (__local_mem__ float*)sum1Tensor.GetPhyAddr();
            __local_mem__ float* sum2 = (__local_mem__ float*)sum2Tensor.GetPhyAddr();
            __local_mem__ float* rstd = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            uint32_t count;

            AscendC::MicroAPI::RegTensor<float> xReg, dyReg, dxReg;
            AscendC::MicroAPI::RegTensor<float> sum1Reg, sum2Reg, rstdReg;
            AscendC::MicroAPI::RegTensor<float> gammaReg;
            AscendC::MicroAPI::RegTensor<float> Reg0, Reg1, Reg2, Reg3, Reg4, Reg5;
            AscendC::MicroAPI::MaskReg pMask;
            count = static_cast<uint32_t>(colSize);
            pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum1Reg, (__local_mem__ float*)sum1 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum2Reg, (__local_mem__ float*)sum2 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg, (__local_mem__ float*)rstd + i);
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
            __local_mem__ float* rstd = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            uint32_t count;

            AscendC::MicroAPI::RegTensor<float> xReg, dyReg, dxReg;
            AscendC::MicroAPI::RegTensor<float> sum1Reg, sum2Reg, rstdReg;
            AscendC::MicroAPI::RegTensor<float> gammaReg;
            AscendC::MicroAPI::RegTensor<float> Reg0, Reg1, Reg2, Reg3, Reg4, Reg5;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                count = static_cast<uint32_t>(colSize);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum1Reg, (__local_mem__ float*)sum1 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sum2Reg, (__local_mem__ float*)sum2 + i);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg, (__local_mem__ float*)rstd + i);
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
__aicore__ inline void LayerNormGradV3GroupedReduceBigMBackward<T, U>::Epilogue()
{
    // Epilogue
    inQueueParam.FreeTensor(mean_);
    inQueueParam.FreeTensor(rstd_);
}

} // namespace LayerNormGradV3
#endif // LAYER_NORM_GRAD_V3_GROUPED_REDUCE_BIG_M_IMPL_