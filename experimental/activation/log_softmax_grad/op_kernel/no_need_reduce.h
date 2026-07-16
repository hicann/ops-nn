/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file no_need_reduce.h
 * \brief
 */
#ifndef __NO_NEED_REDUCE_H__
#define __NO_NEED_REDUCE_H__

#include "log_softmax_grad_base.h"

namespace NsLogSoftmaxGrad {

template <typename T, int BUF_NUM = 2, bool IS_CONTIGUOUS = true>
class NoNeedReduce : public LogSoftmaxGradBase<T, BUF_NUM> {
public:
    __aicore__ inline NoNeedReduce() = default;

    __aicore__ inline void Init(LogSoftmaxGradTilingData& tiling, TPipe* pipePtr)
    {
        this->BaseInit(tiling, pipePtr);
        auto singleBufSize = this->singleBufElems_ * sizeof(float);
        this->pipePtr_->InitBuffer(this->inQueDy_, BUF_NUM, singleBufSize);
        this->pipePtr_->InitBuffer(this->inQueX_, BUF_NUM, singleBufSize);
        this->pipePtr_->InitBuffer(this->outQueZ_, BUF_NUM, singleBufSize);
    }

    __aicore__ inline void Process(GM_ADDR dy, GM_ADDR x, GM_ADDR z)
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        uint64_t blockNum = GetBlockNum();
        uint64_t blockIdx = GetBlockIdx();
        uint64_t elemsFloorAlign = this->totalElems_ / this->singleBufElems_ * this->singleBufElems_;
        uint64_t elemsRemained = this->totalElems_ - elemsFloorAlign;
        uint64_t jumpStride = blockNum * this->singleBufElems_;
        this->dyGM_.SetGlobalBuffer((__gm__ T*)dy, this->totalElems_);
        this->xGM_.SetGlobalBuffer((__gm__ T*)x, this->totalElems_);
        this->zGM_.SetGlobalBuffer((__gm__ T*)z, this->totalElems_);
        uint64_t offset = blockIdx * this->singleBufElems_;
        for (; offset < elemsFloorAlign; offset += jumpStride) {
            this->template CopyInDy<IS_CONTIGUOUS>(offset, this->singleBufElems_);
            this->template CopyInX<IS_CONTIGUOUS>(offset, this->singleBufElems_);
            this->Compute(this->singleBufElems_);
            this->template CopyOutZ<IS_CONTIGUOUS>(offset, this->singleBufElems_);
        }

        if (elemsRemained && offset < this->totalElems_) {
            this->template CopyInDy<IS_CONTIGUOUS>(offset, elemsRemained);
            this->template CopyInX<IS_CONTIGUOUS>(offset, elemsRemained);
            this->Compute(elemsRemained);
            this->template CopyOutZ<IS_CONTIGUOUS>(offset, elemsRemained);
        }
    }

    __aicore__ inline void Compute(uint64_t count)
    {
        LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
        LocalTensor<float> x = this->inQueX_.template DeQue<float>();
        LocalTensor<float> z = this->outQueZ_.template AllocTensor<float>();
        Exp(x, x, count);
        PipeBarrier<PIPE_V>();
        Mul(x, dy, x, count);
        PipeBarrier<PIPE_V>();
        Sub(z, dy, x, count);
        this->outQueZ_.EnQue(z);
        this->inQueX_.FreeTensor(x);
        this->inQueDy_.FreeTensor(dy);
    }
};

} // namespace NsLogSoftmaxGrad
#endif // __NO_NEED_REDUCE_H__
