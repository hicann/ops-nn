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
 * \file layer_norm_grad_v3_grouped_reduce.h
 * \brief
 */
#ifndef LAYER_NORM_GRAD_V3_GROUPED_REDUCE
#define LAYER_NORM_GRAD_V3_GROUPED_REDUCE

#include "layer_norm_grad_v3_base.h"
#include "layer_norm_grad_v3_api.h"

namespace LayerNormGradV3
{
template <typename T, typename PD_GAMMA_TYPE>
class LayerNormGradV3GroupedReduceBigMGammaBeta : public LayerNormGradV3Base
{
public:
    __aicore__ inline LayerNormGradV3GroupedReduceBigMGammaBeta() : LayerNormGradV3Base(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta,
                                GM_ADDR workspace, const LayerNormGradV3TilingDataGroupedReduceBigM* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoTiling();
    __aicore__ inline void PrologueStag1();
    __aicore__ inline void EpilogueStag1(const int64_t offset, const int64_t extent);
    __aicore__ inline void PrologueStag2();
    __aicore__ inline void EpilogueStag2(const int64_t offset, const int64_t extent);
    __aicore__ inline void ProcessMainBlock(const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor,
                                            const int64_t nfactor);
    __aicore__ inline void ProcessFoldBlock(const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor,
                                            const int64_t nfactor);
    __aicore__ inline void PostProcessMainBlock(const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor,
                                                const int64_t nfactor);
    __aicore__ inline void PostProcessFoldBlock(const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor,
                                                const int64_t nfactor);
    __aicore__ inline void ProcessSummation(const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor,
                                            const int64_t nfactor);

private:
    __aicore__ inline void ComputeGamma(const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor,
                                        const LocalTensor<float>& xTensor, const LocalTensor<float>& rstdTensor,
                                        const LocalTensor<float>& meanTensor, const int64_t rowSize,
                                        const int64_t colSize);

private:
    const LayerNormGradV3TilingDataGroupedReduceBigM* __restrict td_;

    int64_t M = 0;

    int64_t Mloop = 0;
    int64_t Mtail = 0;
    int64_t MTotalLoop = 0;

    int64_t BasicBlockLoop = 0;
    int64_t MainFoldCount = 0;
    int64_t CacheBufferCount = 0;
    int64_t ResultCacheID = 0;

    // Global Tensor
    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<float> pdGammaOutTensorGMStag1;
    GlobalTensor<float> pdBetaOutTensorGMStag1;
    GlobalTensor<PD_GAMMA_TYPE> pdGammaOutTensorGMStag2;
    GlobalTensor<PD_GAMMA_TYPE> pdBetaOutTensorGMStag2;
    GlobalTensor<float> postDyInTensorGM;
    GlobalTensor<float> postXInTensorGM;
    GM_ADDR savedPdGammaAddr = nullptr;
    GM_ADDR savedPdBetaAddr = nullptr;
    GM_ADDR savedWorkspaceAddr = nullptr;

    // Local Tensor
    LocalTensor<float> tempTensor;
    LocalTensor<float> cacheTensor0;
    LocalTensor<float> cacheTensor1;
    LocalTensor<float> betaStag1;
    LocalTensor<float> gammaStag1;
    LocalTensor<PD_GAMMA_TYPE> betaStag2;
    LocalTensor<PD_GAMMA_TYPE> gammaStag2;

    LocalTensor<float> dyMain_;
    LocalTensor<float> xMain_;
    LocalTensor<float> rstd_;
    LocalTensor<float> mean_;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueDy;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueParam;
    TQue<QuePosition::VECOUT, 1> outQueueSum;
    TBuf<> cacheBuffer0;
    TBuf<> cacheBuffer1;
    TBuf<> tempBuffer;
};  // LayerNormGradV3GroupedReduceBigMGammaBeta

template <typename T, typename U>
class LayerNormGradV3GroupedReduceBigMBackward : public LayerNormGradV3Base
{
public:
    __aicore__ inline LayerNormGradV3GroupedReduceBigMBackward() : LayerNormGradV3Base(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX,
                                GM_ADDR workspace, const LayerNormGradV3TilingDataGroupedReduceBigM* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Prologue(const int64_t mi, const int64_t mfactor);
    __aicore__ inline void ProcessMainBlock(const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor,
                                            const int64_t nfactor);
    __aicore__ inline void ProcessFoldBlock(const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor,
                                            const int64_t nfactor);
    __aicore__ inline void ProcessSummation(const int64_t mi, const int64_t basicBlockIdx, const int64_t mfactor,
                                            const int64_t nfactor);
    __aicore__ inline void ProcessX(const int64_t mi, const int64_t ni, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ComputeDx(const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor,
                                     const LocalTensor<float>& xTensor, const LocalTensor<float>& gammaTensor,
                                     const LocalTensor<float>& sum1Tensor, const LocalTensor<float>& sum2Tensor,
                                     const LocalTensor<float>& rstdTensor, const int64_t rowSize, const int64_t colSize,
                                     const int64_t stride);
    __aicore__ inline void Epilogue();

private:
    const LayerNormGradV3TilingDataGroupedReduceBigM* __restrict td_;

    constexpr static int64_t CORE_NUM = 64;

    int64_t startM_ = 0;
    int64_t mToProcess_ = 0;
    int64_t usableBlocks_ = 0;
    int64_t backwardMainBlockFactor = 64;
    int64_t backwardBlockDim = 0;
    constexpr static int64_t backwardMfactor = 64;
    constexpr static int64_t backwardNfactor = 64;

    int64_t M = 0;
    int64_t N = 0;
    int64_t MainBlockCount = 0;
    int64_t TailBlockCount = 0;
    int64_t TailBlockFactor = 0;
    int64_t CurrentBlockFactor = 0;

    int64_t BlockIdx = 0;
    int64_t Nfactor = 0;
    int64_t Mfactor = 0;

    int64_t Nloop = 0;
    int64_t Ntail = 0;
    int64_t Mloop = 0;
    int64_t Mtail = 0;

    int64_t NTotalLoop = 0;
    int64_t BasicBlockLoop = 0;
    int64_t MainFoldCount = 0;

    int64_t NfactorBlockAligned = 0;
    int64_t MfactorBlockAligned = 0;

    // Cache
    int64_t CacheBufferCount = 0;
    int64_t ResultCacheID = 0;

    // GM Tensor
    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;

    // Local Tensor
    LocalTensor<float> mean_;
    LocalTensor<float> rstd_;
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
};  // LayerNormGradV3GroupedReduceBigMBackward

template <typename T, typename PD_GAMMA_TYPE>
class LayerNormGradV3GroupedReduceBigNGammaBeta : public LayerNormGradV3Base {
public:
    __aicore__ inline LayerNormGradV3GroupedReduceBigNGammaBeta() : LayerNormGradV3Base(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR pdGamma, GM_ADDR pdBeta, GM_ADDR workspace,
        const LayerNormGradV3TilingDataGroupedReduceBigN* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoTiling();
    __aicore__ inline void Prologue();
    __aicore__ inline void Epilogue(const int64_t offset, const int64_t extent);
    __aicore__ inline void ProcessMainBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ProcessFoldBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void PostProcessMainBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void PostProcessFoldBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);
    __aicore__ inline void ProcessSummation(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor);

private:
    __aicore__ inline void ComputeGamma(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
        const LocalTensor<float>& rstdTensor, const LocalTensor<float>& meanTensor, const int64_t rowSize,
        const int64_t colSize);

private:
    const LayerNormGradV3TilingDataGroupedReduceBigN* __restrict td_;

    int64_t NTotalloop = 0;
    int64_t Ntail = 0;

    // Global Tensor
    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<PD_GAMMA_TYPE> pdGammaOutTensorGM;
    GlobalTensor<PD_GAMMA_TYPE> pdBetaOutTensorGM;
    GlobalTensor<float> postDyInTensorGM;
    GlobalTensor<float> postXInTensorGM;
    GM_ADDR savedPdGammaAddr = nullptr;
    GM_ADDR savedPdBetaAddr = nullptr;
    GM_ADDR savedWorkspaceAddr = nullptr;

    // Local Tensor
    LocalTensor<float> tempTensor;
    LocalTensor<float> cacheTensor0;
    LocalTensor<float> cacheTensor1;
    LocalTensor<PD_GAMMA_TYPE> beta_;
    LocalTensor<PD_GAMMA_TYPE> gamma_;

    LocalTensor<float> dyMain_;
    LocalTensor<float> xMain_;
    LocalTensor<float> rstd_;
    LocalTensor<float> mean_;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueDy;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueParam;
    TQue<QuePosition::VECOUT, 1> outQueueSum;
    TBuf<> cacheBuffer0;
    TBuf<> cacheBuffer1;
    TBuf<> tempBuffer;
}; // LayerNormGradV3GroupedReduceBigNGammaBeta

template <typename T, typename U>
class LayerNormGradV3GroupedReduceBigNBackward : public LayerNormGradV3Base {
public:
    __aicore__ inline LayerNormGradV3GroupedReduceBigNBackward() : LayerNormGradV3Base(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX, GM_ADDR workspace,
        const LayerNormGradV3TilingDataGroupedReduceBigN* tilingData, TPipe* pipeIn);
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
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& dyTensor, const LocalTensor<float>& xTensor,
        const LocalTensor<float>& gammaTensor, const LocalTensor<float>& sum1Tensor,
        const LocalTensor<float>& sum2Tensor, const LocalTensor<float>& rstdTensor, const int64_t rowSize,
        const int64_t colSize, const int64_t stride);
    __aicore__ inline void Epilogue();
    __aicore__ inline void PostPrologue(const int64_t mi, const int64_t mfactor);

private:
    const LayerNormGradV3TilingDataGroupedReduceBigN* __restrict td_;

    int64_t startN_ = 0;
    int64_t nToProcess_ = 0;

    int64_t Nloop = 0;
    int64_t Ntail = 0;
    int64_t BasicBlockLoop = 0;
    int64_t MainFoldCount = 0;

    // Cache
    int64_t CacheBufferCount = 0;
    int64_t ResultCacheID = 0;

    // GM Tensor
    GlobalTensor<T> dyInTensorGM;
    GlobalTensor<T> xInTensorGM;
    GlobalTensor<float> rstdInTensorGM;
    GlobalTensor<float> meanInTensorGM;
    GlobalTensor<U> gammaInTensorGM;
    GlobalTensor<T> pdXOutTensorGM;
    GlobalTensor<float> partialSum1GM, partialSum2GM, finalSumGM;

    // Local Tensor
    LocalTensor<float> mean_;
    LocalTensor<float> rstd_;
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
    TQue<QuePosition::VECOUT, 1> outTmpQueue0;
    TQue<QuePosition::VECOUT, 1> outTmpQueue1;
    TBuf<> reduceSumTempBuffer;
    TBuf<> tempBuffer;
    TBuf<> cacheBuffer0;
    TBuf<> cacheBuffer1;
}; // LayerNormGradV3GroupedReduceBigNBackward

}  // namespace LayerNormGradV3

#endif
