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
 * \file layer_norm_grad_recompute.h
 * \brief
 */
#ifndef LAYER_NORM_GRAD_RECOMPUTE
#define LAYER_NORM_GRAD_RECOMPUTE

#include "layer_norm_grad_base.h"

namespace LayerNormGrad {
template <typename T, typename PD_GAMMA_TYPE>
class LayerNormGradRecomputeGammaBeta : public LayerNormGradBase {
public:
    __aicore__ inline LayerNormGradRecomputeGammaBeta() : LayerNormGradBase(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
        const LayerNormGradTilingDataRecompute* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoTiling();
    __aicore__ inline void Prologue();
    __aicore__ inline void Epilogue(const int64_t offset, const int64_t extent);
    __aicore__ inline void ProcessMainBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ProcessFoldBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ProcessSummation(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);

private:
    __aicore__ inline void ComputeGamma(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
        const LocalTensor<float>& varTensor, const LocalTensor<float>& meanTensor, const int64_t rowSize,
        const int64_t colSize);

private:
    const LayerNormGradTilingDataRecompute* __restrict td_;
    int64_t Ntail = 0;
    int64_t NTotalloop = 0;

    // Global Tensor
    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> varInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<PD_GAMMA_TYPE> pdGammaOutTensorGM;
    GlobalTensor<PD_GAMMA_TYPE> pdBetaOutTensorGM;

    // Local Tensor
    LocalTensor<float> tempTensor;
    LocalTensor<float> cacheTensor0;
    LocalTensor<float> cacheTensor1;
    LocalTensor<PD_GAMMA_TYPE> beta_;
    LocalTensor<PD_GAMMA_TYPE> gamma_;

    LocalTensor<float> dyMain_;
    LocalTensor<float> xMain_;
    LocalTensor<float> var_;
    LocalTensor<float> mean_;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueDy;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueParam;
    TQue<QuePosition::VECOUT, 1> outQueueSum;
    TBuf<> cacheBuffer0;
    TBuf<> cacheBuffer1;
    TBuf<> tempBuffer;
}; // LayerNormGradRecomputeGammaBeta

template <typename T, typename U>
class LayerNormGradRecomputeBackward : public LayerNormGradBase {
public:
    __aicore__ inline LayerNormGradRecomputeBackward() : LayerNormGradBase(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX, GM_ADDR workspace,
        const LayerNormGradTilingDataRecompute* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Prologue(const int64_t mi, const int64_t mfactor);
    __aicore__ inline void ProcessMainBlock(
        const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ProcessFoldBlock(
        const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ProcessSummation(
        const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ProcessX(const int64_t mi, const int64_t ni, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ComputeDx(
        const LocalTensor<T>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
        const LocalTensor<float>& gammaTensor, const LocalTensor<float>& sum1Tensor,
        const LocalTensor<float>& sum2Tensor, const LocalTensor<float>& varTensor, const int64_t rowSize,
        const int64_t colSize, const int64_t stride);
    __aicore__ inline void Epilogue();

private:
    const LayerNormGradTilingDataRecompute* __restrict td_;
    
    int64_t Mloop = 0;
    int64_t Mtail = 0;

    // GM Tensor
    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> varInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;

    // Local Tensor
    LocalTensor<float> mean_;
    LocalTensor<float> var_;
    LocalTensor<float> sum1Tensor;
    LocalTensor<float> sum2Tensor;
    LocalTensor<float> sum1Main_;
    LocalTensor<float> sum2Main_;
    LocalTensor<float> tempTensor;
    LocalTensor<float> cacheTensor0;
    LocalTensor<float> cacheTensor1;
    LocalTensor<float> reduceSumTempTensor;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueDy;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueParam;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECOUT, 1> outQueueDx;
    TBuf<> reduceSumTempBuffer;
    TBuf<> tempBuffer;
    TBuf<> cacheBuffer0;
    TBuf<> cacheBuffer1;
}; // LayerNormGradRecomputeBackward
} // namespace LayerNormGrad

#endif
