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
 * \file layer_norm_v3_welford_multi_params.h
 * \brief
 */

#ifndef LAYER_NORM_V3_WELFORD_MULTI_PARAMS_H
#define LAYER_NORM_V3_WELFORD_MULTI_PARAMS_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "layer_norm_v3_common.h"
#include "../../inc/platform.h"

namespace LayerNormV3 {
using namespace AscendC;

template <typename T, typename U, typename M, bool IsOutRstd>
class LayerNormV3WelfordMultiParams {
public:
    __aicore__ inline LayerNormV3WelfordMultiParams(const LayerNormV3TilingDataWelfordMultiParams* tilingData,
                                                    TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        td_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout)
    {
        if (GetBlockIdx() >= td_->numBlocks) {
            return;
        }

        currentBlockFactor = (GetBlockIdx() == td_->mainBlockCount) ? td_->tailBlockFactor : td_->mainBlockFactor;
        int64_t fmOffset = GetBlockIdx() * td_->mainBlockFactor * td_->N;
        int64_t paramOffset = GetBlockIdx() * td_->mainBlockFactor;
        int64_t fmSize = currentBlockFactor * td_->N;
        int64_t paramSize = currentBlockFactor;

        xGm.SetGlobalBuffer((__gm__ T*)x + fmOffset, fmSize);
        yGm.SetGlobalBuffer((__gm__ T*)y + fmOffset, fmSize);
        meanGm.SetGlobalBuffer((__gm__ M*)mean + paramOffset, paramSize);
        lastoutGm.SetGlobalBuffer((__gm__ M*)lastout + paramOffset, paramSize);
        if (td_->nullptrGamma == 0) {
            gammaGm.SetGlobalBuffer((__gm__ U*)gamma, td_->paramsToNormSize * td_->N);
        }
        if (td_->nullptrBeta == 0) {
            betaGm.SetGlobalBuffer((__gm__ U*)beta, td_->paramsToNormSize * td_->N);
        }

        pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, td_->tileLength * sizeof(T));
        pipe_->InitBuffer(outQueueY, DOUBLE_BUFFER, td_->tileLength * sizeof(T));
        pipe_->InitBuffer(outQueueMean, DOUBLE_BUFFER, AGGREGATION_COUNT * sizeof(float));
        pipe_->InitBuffer(outQueueLastout, DOUBLE_BUFFER, AGGREGATION_COUNT * sizeof(float));
        pipe_->InitBuffer(welfordTempBuffer, td_->welfordTempSize);
        if (td_->nullptrGamma == 0) {
            pipe_->InitBuffer(inQueueGamma, DOUBLE_BUFFER, td_->tileLength * sizeof(U));
        }
        if (td_->nullptrBeta == 0) {
            pipe_->InitBuffer(inQueueBeta, DOUBLE_BUFFER, td_->tileLength * sizeof(U));
        }
        pipe_->InitBuffer(apiTempBuffer, td_->apiTempBufferSize);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= td_->numBlocks) {
            return;
        }

        ResetCache();
        paramAddr = 0;
        mean_ = welfordTempBuffer.Get<float>();
        variance_ = mean_[td_->tileLength];
        if constexpr (IsOutRstd) {
            varianceTensor = variance_[td_->tileLength];
        } else {
            rstdTensor = variance_[td_->tileLength];
        }
        shared_ = apiTempBuffer.Get<uint8_t>();

        for (int64_t i = 0; i < currentBlockFactor; ++i) {
            // Welford Algorithm
            welfordCount = 0;
            WelfordInitialize(mean_, variance_, td_->tileLength);
            for (int64_t welfordUpdateCount = 0; welfordUpdateCount < td_->welfordUpdateTimes; welfordUpdateCount++) {
                int64_t offset = i * td_->N + welfordUpdateCount * td_->tileLength;
                ProcessWelfordUpdate(inQueueX, xGm, offset, td_->tileLength, td_->tileLength, welfordCount, mean_,
                                     variance_, shared_);
            }
            if (td_->welfordUpdateTail > 0) {
                int64_t offset = i * td_->N + td_->welfordUpdateTimes * td_->tileLength;
                ProcessWelfordUpdate(inQueueX, xGm, offset, td_->welfordUpdateTail, td_->tileLength, welfordCount,
                                     mean_, variance_, shared_);
            }

            WelfordFinalizePara para;
            para.rnLength = welfordCount;
            para.abLength = td_->tileLength;
            para.headCount = welfordCount;
            para.headCountLength = td_->welfordUpdateTail;
            para.tailCount = welfordCount - (td_->welfordUpdateTail > 0);
            para.tailCountLength = td_->tileLength - td_->welfordUpdateTail;
            para.abRec = 1.0f / static_cast<float>(td_->tileLength);
            para.rRec = 1.0f / static_cast<float>(td_->N);
            if constexpr (IsOutRstd) {
                WelfordFinalize<true>(meanTensor[cacheCount], varianceTensor[cacheCount], mean_, variance_, shared_,
                                      para);
            } else {
                WelfordFinalize<true>(meanTensor[cacheCount], lastoutTensor[cacheCount], mean_, variance_, shared_,
                                      para);
            }

            // Normalize — compute gammaBase for current row
            int64_t globalRowIdx = GetBlockIdx() * td_->mainBlockFactor + i;
            int64_t gammaBase = (globalRowIdx % td_->paramsToNormSize) * td_->N;

            for (int64_t welfordUpdateCount = 0; welfordUpdateCount < td_->welfordUpdateTimes; welfordUpdateCount++) {
                int64_t fmOffset = i * td_->N + welfordUpdateCount * td_->tileLength;
                int64_t paramOffset = welfordUpdateCount * td_->tileLength;
                ProcessNormalize(fmOffset, paramOffset, td_->tileLength, gammaBase);
            }
            if (td_->welfordUpdateTail > 0) {
                int64_t fmOffset = i * td_->N + td_->welfordUpdateTimes * td_->tileLength;
                int64_t paramOffset = td_->welfordUpdateTimes * td_->tileLength;
                ProcessNormalize(fmOffset, paramOffset, td_->welfordUpdateTail, gammaBase);
            }
            cacheCount++;
            // check cache buffer
            if (cacheCount >= AGGREGATION_COUNT) {
                RefreshCache<M>(cacheCount, paramAddr, meanTensor, lastoutTensor, outQueueMean, outQueueLastout, meanGm,
                                lastoutGm);
                ResetCache();
            }
        }

        // refresh cache
        if (cacheCount > 0) {
            RefreshCache<M>(cacheCount, paramAddr, meanTensor, lastoutTensor, outQueueMean, outQueueLastout, meanGm,
                            lastoutGm);
        }
    }

private:
    __aicore__ inline void ResetCache()
    {
        cacheCount = 0;
        meanTensor = outQueueMean.template AllocTensor<float>();
        lastoutTensor = outQueueLastout.template AllocTensor<float>();
    }

    __aicore__ inline void ProcessNormalize(const int64_t fmOffset, const int64_t paramOffset, const int64_t elemCnt,
                                            const int64_t gammaBase)
    {
        xTensor = inQueueX.template AllocTensor<T>();
        CopyIn(xTensor, xGm[fmOffset], elemCnt);
        inQueueX.EnQue(xTensor);
        xTensor = inQueueX.template DeQue<T>();

        if (td_->nullptrGamma == 0) {
            gammaTensor = inQueueGamma.template AllocTensor<U>();
            CopyIn(gammaTensor, gammaGm[gammaBase + paramOffset], elemCnt);
            inQueueGamma.EnQue(gammaTensor);
            gammaTensor = inQueueGamma.template DeQue<U>();
        }
        if (td_->nullptrBeta == 0) {
            betaTensor = inQueueBeta.template AllocTensor<U>();
            CopyIn(betaTensor, betaGm[gammaBase + paramOffset], elemCnt);
            inQueueBeta.EnQue(betaTensor);
            betaTensor = inQueueBeta.template DeQue<U>();
        }

        yTensor = outQueueY.template AllocTensor<T>();

        NormalizePara para;
        para.aLength = 1;
        para.rLength = elemCnt;
        para.rLengthWithPadding = elemCnt;
        if constexpr (IsOutRstd) {
            Normalize<U, T, false>(yTensor, lastoutTensor[cacheCount], meanTensor[cacheCount],
                                   varianceTensor[cacheCount], xTensor, gammaTensor, betaTensor, shared_, td_->epsilon,
                                   para);
        } else {
            Normalize<U, T, false>(yTensor, rstdTensor[cacheCount], meanTensor[cacheCount], lastoutTensor[cacheCount],
                                   xTensor, gammaTensor, betaTensor, shared_, td_->epsilon, para);
        }

        inQueueX.FreeTensor(xTensor);
        if (td_->nullptrGamma == 0) {
            inQueueGamma.FreeTensor(gammaTensor);
        }
        if (td_->nullptrBeta == 0) {
            inQueueBeta.FreeTensor(betaTensor);
        }

        outQueueY.EnQue(yTensor);
        yTensor = outQueueY.template DeQue<T>();
        CopyOut(yGm[fmOffset], yTensor, elemCnt);
        outQueueY.FreeTensor(yTensor);
    }

private:
    TPipe* pipe_;
    const LayerNormV3TilingDataWelfordMultiParams* __restrict td_;

    constexpr static int64_t DOUBLE_BUFFER = 2;
    constexpr static int64_t AGGREGATION_COUNT = 256;

    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECIN, 1> inQueueBeta;

    TQue<QuePosition::VECOUT, 1> outQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueMean;
    TQue<QuePosition::VECOUT, 1> outQueueLastout;

    TBuf<TPosition::VECCALC> welfordTempBuffer;
    TBuf<TPosition::VECCALC> apiTempBuffer;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<M> meanGm;
    GlobalTensor<M> lastoutGm;
    GlobalTensor<U> gammaGm;
    GlobalTensor<U> betaGm;

    LocalTensor<T> xTensor;
    LocalTensor<T> yTensor;
    LocalTensor<U> gammaTensor;
    LocalTensor<U> betaTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> lastoutTensor;
    LocalTensor<float> varianceTensor;
    LocalTensor<float> rstdTensor;
    LocalTensor<float> mean_;
    LocalTensor<float> variance_;
    LocalTensor<uint8_t> shared_;

    int64_t currentBlockFactor = 0;
    int64_t cacheCount = 0;
    int64_t paramAddr = 0;
    int64_t welfordCount = 0;
};

} // namespace LayerNormV3

#endif // LAYER_NORM_V3_WELFORD_MULTI_PARAMS_H
