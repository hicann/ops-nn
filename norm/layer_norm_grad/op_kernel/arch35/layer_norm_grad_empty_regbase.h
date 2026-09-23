/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_grad_empty_regbase.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_EMPTY_REGBASE_
#define LAYER_NORM_GRAD_EMPTY_REGBASE_
#include "layer_norm_grad_api.h"
#include "layer_norm_grad_base.h"

namespace LayerNormGrad {
using namespace AscendC;

template <typename PD_GAMMA_TYPE>
class LayerNormGradEmptyRegBase : public LayerNormGradBase {
public:
    __aicore__ inline LayerNormGradEmptyRegBase() : LayerNormGradBase(){};
    __aicore__ inline void Init(GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
                                const LayerNormGradTilingDataEmptyRegBase* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FillZeroAndCopyOut(const int64_t ni, const int64_t count);

private:
    const LayerNormGradTilingDataEmptyRegBase* __restrict td_;

    constexpr static int64_t DOUBLE_BUFFER = 2;

    int64_t isTailCore = 0;
    int64_t nAlign = 0;
    int64_t currentNLoop = 0;
    int64_t currentNTail = 0;

    GlobalTensor<PD_GAMMA_TYPE> pdGammaOutTensorGM;
    GlobalTensor<PD_GAMMA_TYPE> pdBetaOutTensorGM;

    TPipe* pipe_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQueueGammaBeta;
};

template <typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradEmptyRegBase<PD_GAMMA_TYPE>::Init(
    GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace, const LayerNormGradTilingDataEmptyRegBase* tilingData,
    TPipe* pipeIn)
{
    td_ = tilingData;
    int64_t usedCoreNum = td_->usedCoreNum;
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= usedCoreNum) {
        return;
    }
    // col=0空tensor场景, 或dgamma/dbeta都不需要时无数据需要写出, 直接退出
    if (td_->col == 0 || (!td_->pdgammaIsRequire && !td_->pdbetaIsRequire)) {
        return;
    }

    // 主核每核nPerCore列, 尾核nTailCore列
    nAlign = td_->nAlign;
    isTailCore = (blockIdx == usedCoreNum - 1) ? 1 : 0;
    int64_t currentN = isTailCore ? td_->nTailCore : td_->nPerCore;
    currentNLoop = currentN / nAlign; // 整块迭代次数
    currentNTail = currentN % nAlign; // 不足一块的尾块大小

    int64_t gmOffset = blockIdx * td_->nPerCore;
    pdGammaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)pdGamma + gmOffset, currentN);
    pdBetaOutTensorGM.SetGlobalBuffer((__gm__ PD_GAMMA_TYPE*)pdBeta + gmOffset, currentN);

    pipe_ = pipeIn;
    pipe_->InitBuffer(outQueueGammaBeta, DOUBLE_BUFFER, nAlign * sizeof(PD_GAMMA_TYPE));
}

template <typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradEmptyRegBase<PD_GAMMA_TYPE>::Process()
{
    if (GetBlockIdx() >= td_->usedCoreNum) {
        return;
    }
    // col=0空tensor场景, 或dgamma/dbeta都不需要时直接退出
    if (td_->col == 0 || (!td_->pdgammaIsRequire && !td_->pdbetaIsRequire)) {
        return;
    }

    int64_t ni = 0;
    for (ni = 0; ni < currentNLoop; ++ni) {
        FillZeroAndCopyOut(ni, nAlign);
    }
    // 尾块单独处理
    if (currentNTail > 0) {
        FillZeroAndCopyOut(ni, currentNTail);
    }
}

template <typename PD_GAMMA_TYPE>
__aicore__ inline void LayerNormGradEmptyRegBase<PD_GAMMA_TYPE>::FillZeroAndCopyOut(const int64_t ni,
                                                                                    const int64_t count)
{
    int64_t offset = ni * nAlign; // 第ni块的N方向偏移
    LocalTensor<PD_GAMMA_TYPE> outTensor = outQueueGammaBeta.template AllocTensor<PD_GAMMA_TYPE>();

    // 步骤1: 高阶API Duplicate将buffer填0
    Duplicate<PD_GAMMA_TYPE>(outTensor, static_cast<PD_GAMMA_TYPE>(0), static_cast<uint32_t>(count));
    outQueueGammaBeta.EnQue(outTensor);
    outTensor = outQueueGammaBeta.template DeQue<PD_GAMMA_TYPE>();

    // 步骤2: 按需搬出dgamma/dbeta
    if (td_->pdgammaIsRequire) {
        CopyOut<PD_GAMMA_TYPE>(pdGammaOutTensorGM[offset], outTensor, count);
    }
    if (td_->pdbetaIsRequire) {
        CopyOut<PD_GAMMA_TYPE>(pdBetaOutTensorGM[offset], outTensor, count);
    }
    outQueueGammaBeta.FreeTensor(outTensor);
}

} // namespace LayerNormGrad
#endif // LAYER_NORM_GRAD_EMPTY_REGBASE_
