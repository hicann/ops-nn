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
 * \file layer_norm_grad_recompute_gamma_beta_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_RECOMPUTE_GAMMA_BETA_IMPL_
#define LAYER_NORM_GRAD_RECOMPUTE_GAMMA_BETA_IMPL_

#include "layer_norm_grad_recompute.h"
#include "layer_norm_grad_base.h"

namespace LayerNormGrad {
using namespace AscendC;
template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
    const LayerNormGradTilingDataRecompute* tilingData, TPipe* pipeIn)
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
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::Process()
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
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::Prologue()
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
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::Epilogue(const int64_t offset, const int64_t extent)
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
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::ProcessMainBlock(
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
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::ProcessFoldBlock(
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
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::ProcessSummation(
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
__aicore__ inline void LayerNormGradRecomputeGammaBeta<T, PD_GAMMA_TYPE>::ComputeGamma(
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
} // namespace LayerNormGrad
#endif
