/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_norm_v4_welford.h
 * \brief
 */
#ifndef LAYER_NORM_V4_WELFORD_H
#define LAYER_NORM_V4_WELFORD_H

#include "layer_norm_v4_welford_base.h"

namespace LayerNormV4 {
template <typename T, typename U, typename M>
class LayerNormV4RegbaseWelford : public LayerNormV4WelfordBase<T, U, M, LayerNormV4TilingDataWelford> {
    using Base = LayerNormV4WelfordBase<T, U, M, LayerNormV4TilingDataWelford>;

public:
    __aicore__ inline LayerNormV4RegbaseWelford(const LayerNormV4TilingDataWelford* tilingData, TPipe* pipeIn)
        : Base(tilingData, pipeIn)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd)
    {
        if (GetBlockIdx() >= this->td_->numBlocks) {
            return;
        }
        currentBlockFactor = (GetBlockIdx() == this->td_->mainBlockCount) ? this->td_->tailBlockFactor :
                                                                            this->td_->mainBlockFactor;
        int64_t fmOffset = GetBlockIdx() * this->td_->mainBlockFactor * this->td_->N;
        int64_t paramOffset = GetBlockIdx() * this->td_->mainBlockFactor;
        int64_t fmSize = currentBlockFactor * this->td_->N;
        int64_t paramSize = currentBlockFactor;

        this->xGm.SetGlobalBuffer((__gm__ T*)x + fmOffset, fmSize);
        this->yGm.SetGlobalBuffer((__gm__ T*)y + fmOffset, fmSize);
        this->meanGm.SetGlobalBuffer((__gm__ M*)mean + paramOffset, paramSize);
        this->rstdGm.SetGlobalBuffer((__gm__ M*)rstd + paramOffset, paramSize);
        this->InitCommon(gamma, beta);
    }

    __aicore__ inline void Process()
    {
        // Process
        if (GetBlockIdx() >= this->td_->numBlocks) {
            return;
        }

        this->ResetCache();
        this->paramAddr = 0;
        this->mean_ = this->welfordTempBuffer.template Get<float>();
        this->variance_ = this->mean_[this->td_->tileLength];
        this->varianceTensor = this->variance_[this->td_->tileLength];
        this->shared_ = this->apiTempBuffer.template Get<uint8_t>();

        for (int64_t i = 0; i < currentBlockFactor; ++i) {
            // Welford Algorithm
            this->WelfordInitialize(this->mean_, this->variance_, this->td_->tileLength);
            for (int64_t welfordUpdateCount = 0; welfordUpdateCount < this->td_->welfordUpdateTimes;
                 welfordUpdateCount++) {
                int64_t offset = i * this->td_->N + welfordUpdateCount * this->td_->tileLength;
                this->ProcessWelfordUpdate(offset, this->td_->tileLength);
            }
            if (this->td_->welfordUpdateTail > 0) {
                int64_t offset = i * this->td_->N + this->td_->welfordUpdateTimes * this->td_->tileLength;
                this->ProcessWelfordUpdate(offset, this->td_->welfordUpdateTail);
            }

            WelfordFinalizePara para;
            para.rnLength = this->welfordCount;
            para.abLength = this->td_->tileLength;
            para.headCount = this->welfordCount;
            para.headCountLength = this->td_->welfordUpdateTail;
            para.tailCount = this->welfordCount - (this->td_->welfordUpdateTail > 0);
            para.tailCountLength = this->td_->tileLength - this->td_->welfordUpdateTail;
            para.abRec = 1.0f / static_cast<float>(this->td_->tileLength);
            para.rRec = 1.0f / static_cast<float>(this->td_->N);
            WelfordFinalize<true>(this->meanTensor[this->cacheCount], this->varianceTensor[this->cacheCount],
                                  this->mean_, this->variance_, this->shared_, para);

            // Normalize
            for (int64_t welfordUpdateCount = 0; welfordUpdateCount < this->td_->welfordUpdateTimes;
                 welfordUpdateCount++) {
                int64_t fmOffset = i * this->td_->N + welfordUpdateCount * this->td_->tileLength;
                int64_t paramOffset = welfordUpdateCount * this->td_->tileLength;
                this->ProcessNormalize(fmOffset, paramOffset, this->td_->tileLength);
            }
            if (this->td_->welfordUpdateTail > 0) {
                int64_t fmOffset = i * this->td_->N + this->td_->welfordUpdateTimes * this->td_->tileLength;
                int64_t paramOffset = this->td_->welfordUpdateTimes * this->td_->tileLength;
                this->ProcessNormalize(fmOffset, paramOffset, this->td_->welfordUpdateTail);
            }

            this->cacheCount++;
            // check cache buffer
            if (this->cacheCount >= this->AGGREGATION_COUNT) {
                this->RefreshCache();
                this->ResetCache();
            }
        }

        // refresh cache
        if (this->cacheCount > 0) {
            this->RefreshCache();
        }
    }

private:
    int64_t currentBlockFactor = 0;
};

} // namespace LayerNormV4

#endif // LAYER_NORM_V4_WELFORD_H
