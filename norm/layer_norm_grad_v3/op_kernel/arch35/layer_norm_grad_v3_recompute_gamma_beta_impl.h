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
 * \file layer_norm_grad_v3_recompute_gamma_beta_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_RECOMPUTE_GAMMA_BETA_IMPL_
#define LAYER_NORM_GRAD_V3_RECOMPUTE_GAMMA_BETA_IMPL_

#include "layer_norm_grad_v3_recompute.h"

namespace LayerNormGradV3 {
using namespace AscendC;
template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3RecomputeGammaBeta<T, PD_GAMMA_TYPE>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
    const LayerNormGradV3TilingDataRecompute* tilingData, TPipe* pipeIn)
{
    td_ = tilingData;

    if (GetBlockIdx() >= td_->gammaBetaBlockDim) {
        return;
    }

    if (GetBlockIdx() < (td_->gammaBetaBlockDim - 1)) {
        Ntail = td_->gammaBetaNtailMainBlock;
        NTotalloop = td_->gammaBetaNloopMainBlock;
    } else {
        Ntail = td_->gammaBetaNtailTailBlock;
        NTotalloop = td_->gammaBetaNloopTailBlock;
    }

    // Init GM Tensor
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + GetBlockIdx() * td_->gammaBetaMainBlockFactor);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * td_->gammaBetaMainBlockFactor);
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd, td_->row);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean, td_->row);
    pdGammaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)pdGamma + GetBlockIdx() * td_->gammaBetaMainBlockFactor);
    pdBetaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)pdBeta + GetBlockIdx() * td_->gammaBetaMainBlockFactor);

    // Init Pipe
    pipe_ = pipeIn;
    int64_t cacheBufSize = td_->gammaBetaCacheBufferCount * td_->gammaBetaNfactor * sizeof(float);
    int64_t dyBufSize = td_->gammaBetaMfactor * td_->gammaBetaNfactor * sizeof(float);
    pipe_->InitBuffer(inQueueDy, 3, dyBufSize);
    pipe_->InitBuffer(inQueueX, 3, dyBufSize);
    pipe_->InitBuffer(inQueueParam, 2, td_->gammaBetaMfactor * sizeof(float));
    pipe_->InitBuffer(outQueueSum, 2, td_->gammaBetaNfactor * sizeof(float));
    pipe_->InitBuffer(cacheBuffer0, cacheBufSize);
    pipe_->InitBuffer(cacheBuffer1, cacheBufSize);
    pipe_->InitBuffer(tempBuffer, td_->gammaBetaNfactor * sizeof(float));
}

template <typename T, typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradV3RecomputeGammaBeta<T, PD_GAMMA_TYPE>::Process()
{
    // Process
    if (GetBlockIdx() >= td_->gammaBetaBlockDim) {
        return;
    }
    tempTensor = tempBuffer.Get<float>();
    cacheTensor0 = cacheBuffer0.Get<float>();
    cacheTensor1 = cacheBuffer1.Get<float>();

    int64_t mfactorMain = td_->gammaBetaBasicBlockLoop ?
                              td_->gammaBetaMfactor :
                              (td_->row == td_->gammaBetaMfactor ? td_->gammaBetaMfactor : td_->gammaBetaMtail);
    int64_t loopCount = td_->gammaBetaBasicBlockLoop ? td_->gammaBetaBasicBlockLoop : 1;
    for (int64_t ni = 0; ni < NTotalloop; ++ni) {
        LayerNormGradV3Base::GammaBetaPrologueCommon<PD_GAMMA_TYPE>(td_, outQueueSum, beta_, gamma_);

        int64_t nfactor = (ni == NTotalloop - 1) ? Ntail : td_->gammaBetaNfactor;

        for (int64_t basicBlockIdx = 0; basicBlockIdx < loopCount; ++basicBlockIdx) {
            LayerNormGradV3Base::ProcessGammaBetaMainBlockCommon<T>(
                td_, ni, basicBlockIdx, mfactorMain, nfactor, dyMain_, xMain_, rstd_, mean_, inQueueDy, inQueueX,
                inQueueParam, dyInTensorGM, xInTensorGM, rstdInTensorGM, meanInTensorGM);

            if (td_->gammaBetaBasicBlockLoop > 0 &&
                ((basicBlockIdx < td_->gammaBetaMainFoldCount) ||
                 (basicBlockIdx == td_->gammaBetaMainFoldCount && td_->gammaBetaMtail > 0))) {
                const int64_t mfactorFold = (basicBlockIdx < td_->gammaBetaMainFoldCount) ? td_->gammaBetaMfactor :
                                                                                            td_->gammaBetaMtail;
                LayerNormGradV3Base::ProcessGammaBetaFoldBlockCommon<T>(
                    td_, ni, basicBlockIdx, mfactorFold, nfactor, dyMain_, xMain_, inQueueDy, inQueueX, inQueueParam,
                    dyInTensorGM, xInTensorGM, rstdInTensorGM, meanInTensorGM);
            }

            LayerNormGradV3Base::GammaBetaProcessSummationCommon(td_, basicBlockIdx, mfactorMain, nfactor, tempTensor,
                                                                 dyMain_, xMain_, cacheTensor0, cacheTensor1, inQueueDy,
                                                                 inQueueX);
        }

        LayerNormGradV3Base::GammaBetaEpilogueCommon<PD_GAMMA_TYPE>(td_, ni * td_->gammaBetaNfactor, nfactor,
                                                                    outQueueSum, cacheTensor0, cacheTensor1, beta_,
                                                                    gamma_, pdBetaOutTensorGM, pdGammaOutTensorGM);
    }
}

} // namespace LayerNormGradV3
#endif
