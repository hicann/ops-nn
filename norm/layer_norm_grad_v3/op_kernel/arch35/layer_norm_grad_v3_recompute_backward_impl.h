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
 * \file layer_norm_grad_v3_recompute_backward_impl.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_RECOMPUTE_BACKWARD_IMPL_
#define LAYER_NORM_GRAD_V3_RECOMPUTE_BACKWARD_IMPL_

#include "layer_norm_grad_v3_recompute.h"
#include "layer_norm_grad_v3_base.h"

namespace LayerNormGradV3 {
using namespace AscendC;

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::Init(
    GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR pdX, GM_ADDR workspace,
    const LayerNormGradV3TilingDataRecompute* tilingData, TPipe* pipeIn)
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
    int64_t rstdTensorSize = GetBlockIdx() * td_->backwardMainBlockFactor;
    dyInTensorGM.SetGlobalBuffer((__gm__ T*)dy + xInTensorSize);
    xInTensorGM.SetGlobalBuffer((__gm__ T*)x + xInTensorSize);
    rstdInTensorGM.SetGlobalBuffer((__gm__ float*)rstd + rstdTensorSize);
    meanInTensorGM.SetGlobalBuffer((__gm__ float*)mean + rstdTensorSize);
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
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::Process()
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
    int64_t nfactor = td_->backwardBasicBlockLoop ?
                          td_->backwardNfactor :
                          (td_->col == td_->backwardNfactor ? td_->backwardNfactor : td_->backwardNLoopTail);
    int64_t xTotalLoop = td_->backwardNLoopMain + (td_->backwardNLoopTail > 0 ? 1 : 0);
    for (int64_t mi = 0; mi < totalMRounds; ++mi) {
        int64_t mfactor = (mi < Mloop) ? td_->backwardMfactor : Mtail;
        Prologue(mi, mfactor);
        int64_t loopCnt = td_->backwardBasicBlockLoop > 0 ? td_->backwardBasicBlockLoop : 1;
        for (int64_t basicBlockIdx = 0; basicBlockIdx < loopCnt; ++basicBlockIdx) {
            ProcessMainBlock(mi, basicBlockIdx, mfactor, nfactor);
            if (td_->backwardBasicBlockLoop &&
                ((basicBlockIdx < td_->backwardMainFoldCount) ||
                 (basicBlockIdx == td_->backwardMainFoldCount && td_->backwardNLoopTail > 0))) {
                int64_t foldNfactor = (basicBlockIdx < td_->backwardMainFoldCount) ? td_->backwardNfactor :
                                                                                     td_->backwardNLoopTail;
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
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::Prologue(const int64_t mi, const int64_t mfactor)
{
    // Prologue
    int64_t offset = mi * td_->backwardMfactor;
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
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::ProcessMainBlock(const int64_t mi,
                                                                                const int64_t basicBlockIdx,
                                                                                const int64_t mfactor,
                                                                                const int64_t nfactor)
{
    // Process Main Block
    int64_t offset = mi * td_->backwardMfactor * td_->col + basicBlockIdx * td_->backwardNfactor;
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

    offset = basicBlockIdx * td_->backwardNfactor;
    LocalTensor<float> gammaMain_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gammaMain_.ReinterpretCast<U>(), gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> gammaMainCastTensor = gammaMain_.ReinterpretCast<U>()[td_->backwardNfactorBlockAligned];
        CopyIn(gammaMainCastTensor, gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaMain_);
        gammaMain_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gammaMain_, gammaMainCastTensor, nfactor);
    }
    // 计算
    NlastBroadcastMul(sum1Main_, sum1Main_, gammaMain_, mfactor, td_->backwardNfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaMain_);

    offset = mi * td_->backwardMfactor * td_->col + basicBlockIdx * td_->backwardNfactor;
    sum2Main_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(sum2Main_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> sum2CastTensor = sum2Main_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(sum2CastTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(sum2Main_);
        sum2Main_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(sum2Main_, sum2CastTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }
    // 计算
    Normalize(sum2Main_, sum2Main_, mean_, rstd_, mfactor, td_->backwardNfactorBlockAligned);
    // 计算
    VectorMul(sum2Main_, sum2Main_, sum1Main_, mfactor * td_->backwardNfactorBlockAligned);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::ProcessFoldBlock(const int64_t mi,
                                                                                const int64_t basicBlockIdx,
                                                                                const int64_t mfactor,
                                                                                const int64_t nfactor)
{
    // Process Fold Block
    int64_t offset = mi * td_->backwardMfactor * td_->col +
                     (basicBlockIdx + td_->backwardBasicBlockLoop) * td_->backwardNfactor;
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

    offset = (basicBlockIdx + td_->backwardBasicBlockLoop) * td_->backwardNfactor;
    LocalTensor<float> gammaFold_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gammaFold_.ReinterpretCast<U>(), gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaFold_);
        gammaFold_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> gammaFoldCastTensor = gammaFold_.ReinterpretCast<U>()[td_->backwardNfactorBlockAligned];
        CopyIn(gammaFoldCastTensor, gammaInTensorGM[offset], nfactor);
        inQueueGamma.EnQue(gammaFold_);
        gammaFold_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gammaFold_, gammaFoldCastTensor, nfactor);
    }
    // 进行计算
    PRELOAD(2);
    NlastBroadcastMul(dyFold_, dyFold_, gammaFold_, mfactor, td_->backwardNfactorBlockAligned);
    inQueueGamma.FreeTensor(gammaFold_);

    offset = mi * td_->backwardMfactor * td_->col +
             (basicBlockIdx + td_->backwardBasicBlockLoop) * td_->backwardNfactor;
    LocalTensor<float> xFold_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(xFold_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
        inQueueX.EnQue(xFold_);
        xFold_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> xFoldCastTensor = xFold_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(xFoldCastTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(xFold_);
        xFold_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(xFold_, xFoldCastTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }
    Normalize(xFold_, xFold_, mean_, rstd_, mfactor, td_->backwardNfactorBlockAligned);
    VectorMul(xFold_, xFold_, dyFold_, mfactor * td_->backwardNfactorBlockAligned);

    VectorAdd(sum1Main_, sum1Main_, dyFold_, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    inQueueDy.FreeTensor(dyFold_);
    VectorAdd(sum2Main_, sum2Main_, xFold_, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    inQueueX.FreeTensor(xFold_);
}

template <typename T, typename U>
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::ProcessSummation(const int64_t mi,
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
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::ProcessX(const int64_t mi, const int64_t ni,
                                                                        const int64_t mfactor, const int64_t nfactor)
{
    // Process X
    int64_t offset = mi * td_->backwardMfactor * td_->col + ni * td_->backwardNfactor;
    LocalTensor<float> x_ = inQueueX.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(x_.ReinterpretCast<T>(), xInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
        inQueueX.EnQue(x_);
        x_ = inQueueX.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> xCastTensor = x_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(xCastTensor, xInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueX.EnQue(x_);
        x_ = inQueueX.template DeQue<float>();
        CastToFp32From<T>(x_, xCastTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }
    Normalize(x_, x_, mean_, rstd_, mfactor, td_->backwardNfactorBlockAligned);

    LocalTensor<float> dy_ = inQueueDy.template AllocTensor<float>();
    if constexpr (IsSameType<T, float>::value) {
        CopyIn(dy_.ReinterpretCast<T>(), dyInTensorGM[offset], mfactor, nfactor, td_->backwardNfactorBlockAligned,
               td_->col);
        inQueueDy.EnQue(dy_);
        dy_ = inQueueDy.template DeQue<float>();
    } else if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
        LocalTensor<T> dyCastTensor = dy_.ReinterpretCast<T>()[td_->backwardNfactorBlockAligned];
        CopyIn(dyCastTensor, dyInTensorGM[offset], mfactor, nfactor, 2 * td_->backwardNfactorBlockAligned, td_->col);
        inQueueDy.EnQue(dy_);
        dy_ = inQueueDy.template DeQue<float>();
        CastToFp32From<T>(dy_, dyCastTensor, mfactor, nfactor, td_->backwardNfactorBlockAligned);
    }

    LocalTensor<float> gamma_ = inQueueGamma.template AllocTensor<float>();
    if constexpr (IsSameType<U, float>::value) {
        CopyIn(gamma_.ReinterpretCast<U>(), gammaInTensorGM[ni * td_->backwardNfactor], nfactor);
        inQueueGamma.EnQue(gamma_);
        gamma_ = inQueueGamma.template DeQue<float>();
    } else if constexpr (IsSameType<U, bfloat16_t>::value || IsSameType<U, half>::value) {
        LocalTensor<U> gammaCastTensor = gamma_.ReinterpretCast<U>()[td_->backwardNfactorBlockAligned];
        CopyIn(gammaCastTensor, gammaInTensorGM[ni * td_->backwardNfactor], nfactor);
        inQueueGamma.EnQue(gamma_);
        gamma_ = inQueueGamma.template DeQue<float>();
        CastToFp32From<U>(gamma_, gammaCastTensor, nfactor);
    }

    LocalTensor<T> dx_ = outQueueDx.template AllocTensor<T>();
    ComputeDxCommon<T>(dx_, dy_, x_, gamma_, sum1Tensor, sum2Tensor, rstd_, mfactor, nfactor,
                       td_->backwardNfactorBlockAligned, td_->col);
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
__aicore__ inline void LayerNormGradV3RecomputeBackward<T, U>::Epilogue()
{
    // Epilogue
    inQueueParam.FreeTensor(mean_);
    inQueueParam.FreeTensor(rstd_);
}
} // namespace LayerNormGradV3
#endif
