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
 * \file reduce_mid.h
 * \brief
 */
#ifndef __REDUCE_MID_H__
#define __REDUCE_MID_H__

#include "log_softmax_grad_base.h"

namespace NsLogSoftmaxGrad {

template <typename T, bool IS_SMALL = false, bool IS_CONTIGUOUS = false, int BUF_NUM = 2>
class ReduceMid : public LogSoftmaxGradBase<T, BUF_NUM> {
public:
    __aicore__ inline ReduceMid() = default;

    __aicore__ inline void Init(LogSoftmaxGradTilingData& tiling, TPipe* pipePtr)
    {
        this->BaseInit(tiling, pipePtr);
        auto singleBufSize = this->singleBufElems_ * sizeof(float);
        this->pipePtr_->InitBuffer(this->inQueDy_, BUF_NUM, singleBufSize);
        this->pipePtr_->InitBuffer(this->inQueX_, BUF_NUM, singleBufSize);
        this->pipePtr_->InitBuffer(this->outQueZ_, BUF_NUM, singleBufSize);
        if constexpr (IS_SMALL == false) {
            this->pipePtr_->InitBuffer(tempBuf_, singleBufSize);
            t1_ = tempBuf_.Get<float>();
        }

        dim2Bytes_ = this->mergedDim2_ * sizeof(T);
        if constexpr (IS_CONTIGUOUS == false) {
            this->inParams_.dstStride = DEFAULT_DATA_COPY_STRIDE;
            this->outParams_.srcStride = DEFAULT_DATA_COPY_STRIDE;
        }
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();
    }

    __aicore__ inline void Process(GM_ADDR dy, GM_ADDR x, GM_ADDR z)
    {
        if (this->dim0Tile_ == 0) {
            LoopDim0((__gm__ T*)dy, (__gm__ T*)x, (__gm__ T*)z);
        } else {
            TileDim0((__gm__ T*)dy, (__gm__ T*)x, (__gm__ T*)z);
        }
    }

private:
    __aicore__ inline void TileDim0(__gm__ T* dyAddrBase, __gm__ T* xAddrBase, __gm__ T* zAddrBase)
    {
        if constexpr (IS_CONTIGUOUS == false) {
            this->inParams_.blockLen = this->mergedDim2_ * sizeof(T);
            this->inParams_.srcStride = 0;
            this->inParams_.blockCount = this->dim0Tile_ * this->mergedDim1_;
            this->outParams_.blockLen = this->inParams_.blockLen;
            this->outParams_.dstStride = this->inParams_.srcStride;
            this->outParams_.blockCount = this->inParams_.blockCount;
        }

        uint64_t tempStride = this->dim0Tile_ * this->mergedDim1_ * this->mergedDim2_;
        for (uint64_t i = blockIdx_; i < this->dim0LoopTime_; i += blockNum_) {
            auto offset = i * tempStride;
            this->dyGM_.SetGlobalBuffer(dyAddrBase + offset);
            this->xGM_.SetGlobalBuffer(xAddrBase + offset);
            this->zGM_.SetGlobalBuffer(zAddrBase + offset);
            ProcSmallDim1(0, this->dim0Tile_, this->mergedDim2_);
        }
        if (this->dim0Remained_ && blockIdx_ == this->dim0LoopTime_ % blockNum_) {
            auto offset = this->dim0LoopTime_ * tempStride;
            this->dyGM_.SetGlobalBuffer(dyAddrBase + offset);
            this->xGM_.SetGlobalBuffer(xAddrBase + offset);
            this->zGM_.SetGlobalBuffer(zAddrBase + offset);
            ProcSmallDim1(0, this->dim0Remained_, this->mergedDim2_);
        }
    }

    __aicore__ inline void LoopDim0(__gm__ T* dyAddrBase, __gm__ T* xAddrBase, __gm__ T* zAddrBase)
    {
        if constexpr (IS_CONTIGUOUS == false) {
            this->inParams_.blockLen = this->dim2Tile_ * sizeof(T);
            this->inParams_.srcStride = dim2Bytes_ - this->inParams_.blockLen;
            this->outParams_.blockLen = this->inParams_.blockLen;
            this->outParams_.dstStride = this->inParams_.srcStride;
            if constexpr (IS_SMALL) {
                this->inParams_.blockCount = this->mergedDim1_;
                this->outParams_.blockCount = this->inParams_.blockCount;
            }
        }

        uint64_t dim0Stride = this->mergedDim1_ * this->mergedDim2_;
        uint64_t totalTasks = this->mergedDim0_ * this->dim2LoopTime_;
        uint64_t taskIdx = blockIdx_;
        for (; taskIdx < totalTasks; taskIdx += blockNum_) {
            uint64_t dim0Idx = taskIdx / this->dim2LoopTime_;
            uint64_t dim2Idx = taskIdx - dim0Idx * this->dim2LoopTime_;
            uint64_t dim0Offset = dim0Idx * dim0Stride;
            uint64_t dim2Offset = dim2Idx * this->dim2Tile_;
            this->dyGM_.SetGlobalBuffer(dyAddrBase + dim0Offset);
            this->xGM_.SetGlobalBuffer(xAddrBase + dim0Offset);
            this->zGM_.SetGlobalBuffer(zAddrBase + dim0Offset);
            if constexpr (IS_SMALL) {
                ProcSmallDim1(dim2Offset, this->dim2Tile_);
            } else {
                ProcLargeDim1(dim2Offset, this->dim2Tile_);
            }
        }
        if (this->dim2Remained_) {
            ProcessDim2Remained(dyAddrBase, xAddrBase, zAddrBase, totalTasks, taskIdx);
        }
    }

    __aicore__ inline void ProcessDim2Remained(__gm__ T* dyAddrBase, __gm__ T* xAddrBase, __gm__ T* zAddrBase,
                                               uint64_t totalTasks, uint64_t taskIdx)
    {
        this->inParams_.blockLen = this->dim2Remained_ * sizeof(T);
        this->inParams_.srcStride = dim2Bytes_ - this->inParams_.blockLen;
        this->outParams_.blockLen = this->inParams_.blockLen;
        this->outParams_.dstStride = this->inParams_.srcStride;
        uint64_t dim2Offset = this->mergedDim2_ - this->dim2Remained_;
        uint64_t dim2RemainedAlign = AlignUp(this->dim2Remained_, BLOCK_SIZE / sizeof(T));
        this->dim1Tile_ = GetMin(this->singleBufElems_ / dim2RemainedAlign, this->mergedDim1_);
        this->dim1LoopTime_ = this->mergedDim1_ / this->dim1Tile_;
        this->dim1Remained_ = this->mergedDim1_ - this->dim1LoopTime_ * this->dim1Tile_;
        bool isSmall = this->dim1Tile_ == this->mergedDim1_;
        if (isSmall) {
            this->inParams_.blockCount = this->mergedDim1_;
            this->outParams_.blockCount = this->inParams_.blockCount;
        }
        uint64_t newTotalTasks = totalTasks + this->mergedDim0_;
        uint64_t dim0Stride = this->mergedDim1_ * this->mergedDim2_;
        for (; taskIdx < newTotalTasks; taskIdx += blockNum_) {
            uint64_t dim0Offset = (taskIdx - totalTasks) * dim0Stride;
            this->dyGM_.SetGlobalBuffer(dyAddrBase + dim0Offset);
            this->xGM_.SetGlobalBuffer(xAddrBase + dim0Offset);
            this->zGM_.SetGlobalBuffer(zAddrBase + dim0Offset);
            if (isSmall) {
                ProcSmallDim1(dim2Offset, this->dim2Remained_);
            } else {
                ProcLargeDim1(dim2Offset, this->dim2Remained_);
            }
        }
    }

    __aicore__ inline void ProcSmallDim1(uint64_t dim2Offset, uint64_t w)
    {
        uint64_t wAlign = w;
        if constexpr (IS_CONTIGUOUS == false) {
            constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
            wAlign = AlignUp(w, elemsPerBlock);
        }
        uint64_t copyNum = this->mergedDim1_ * wAlign;
        this->template CopyInDy<IS_CONTIGUOUS>(dim2Offset, copyNum);
        this->template CopyInX<IS_CONTIGUOUS>(dim2Offset, copyNum);
        ComputeSmall(this->mergedDim1_, wAlign);
        this->template CopyOutZ<IS_CONTIGUOUS>(dim2Offset, copyNum);
    }

    __aicore__ inline void ProcSmallDim1(uint64_t dim2Offset, uint64_t n, uint64_t w)
    {
        uint64_t wAlign = w;
        if constexpr (IS_CONTIGUOUS == false) {
            constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
            wAlign = AlignUp(w, elemsPerBlock);
        }
        uint64_t copyNum = n * this->mergedDim1_ * wAlign;
        this->template CopyInDy<IS_CONTIGUOUS>(dim2Offset, copyNum);
        this->template CopyInX<IS_CONTIGUOUS>(dim2Offset, copyNum);
        ComputeSmall(n, this->mergedDim1_, wAlign);
        this->template CopyOutZ<IS_CONTIGUOUS>(dim2Offset, copyNum);
    }

    __aicore__ inline void ComputeSmall(uint64_t h, uint64_t wAlign)
    {
        uint64_t calcElems = h * wAlign;
        LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
        LocalTensor<float> x = this->inQueX_.template DeQue<float>();
        LocalTensor<float> z = this->outQueZ_.template AllocTensor<float>();
        DataCopy(z, dy, calcElems);
        Exp(x, x, calcElems);
        BinaryAddReduceDy(z, h, wAlign);
        PipeBarrier<PIPE_V>();
        RectMul(x, z, h, wAlign);
        PipeBarrier<PIPE_V>();
        Sub(z, dy, x, calcElems);
        this->outQueZ_.EnQue(z);
        this->inQueX_.FreeTensor(x);
        this->inQueDy_.FreeTensor(dy);
    }

    __aicore__ inline void ComputeSmall(uint64_t n, uint64_t h, uint64_t wAlign)
    {
        uint64_t calcElems = n * h * wAlign;
        LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
        LocalTensor<float> x = this->inQueX_.template DeQue<float>();
        LocalTensor<float> z = this->outQueZ_.template AllocTensor<float>();
        DataCopy(z, dy, calcElems);
        Exp(x, x, calcElems);
        BinaryAddReduceDy(z, n, h, wAlign);
        PipeBarrier<PIPE_V>();
        for (uint64_t i = 0; i < n; i++) {
            auto offset = i * h * wAlign;
            auto tempDst = x[offset];
            auto temoSrc = z[offset];
            RectMul(tempDst, temoSrc, h, wAlign);
        }
        PipeBarrier<PIPE_V>();
        Sub(z, dy, x, calcElems);
        this->outQueZ_.EnQue(z);
        this->inQueX_.FreeTensor(x);
        this->inQueDy_.FreeTensor(dy);
    }

    __aicore__ inline void ProcLargeDim1(uint64_t dim2Offset, uint64_t w)
    {
        uint64_t wAlign = w;
        if constexpr (IS_CONTIGUOUS == false) {
            constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
            wAlign = AlignUp(w, elemsPerBlock);
        }
        uint64_t calcElems = this->dim1Tile_ * wAlign;
        Duplicate(t1_, float(0.0), calcElems);
        uint64_t tempStride = this->dim1Tile_ * this->mergedDim2_;
        AccumulateDy(dim2Offset, wAlign, calcElems, tempStride);
        ComputeAndCopyOut(dim2Offset, wAlign, calcElems, tempStride);
    }

    __aicore__ inline void AccumulateDy(uint64_t dim2Offset, uint64_t wAlign, uint64_t calcElems, uint64_t tempStride)
    {
        if constexpr (IS_CONTIGUOUS == false) {
            this->inParams_.blockCount = this->dim1Tile_;
        }
        uint64_t dim1Offset = dim2Offset;
        for (uint64_t i = 0; i < this->dim1LoopTime_; i++) {
            this->template CopyInDy<IS_CONTIGUOUS>(dim1Offset, calcElems);
            LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
            Add(t1_, t1_, dy, calcElems);
            this->inQueDy_.FreeTensor(dy);
            dim1Offset += tempStride;
        }
        if (this->dim1Remained_) {
            uint64_t remainedElems = this->dim1Remained_ * wAlign;
            if constexpr (IS_CONTIGUOUS == false) {
                this->inParams_.blockCount = this->dim1Remained_;
            }
            this->template CopyInDy<IS_CONTIGUOUS>(dim1Offset, remainedElems);
            LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
            Add(t1_, t1_, dy, remainedElems);
            this->inQueDy_.FreeTensor(dy);
        }
    }

    __aicore__ inline void ComputeAndCopyOut(uint64_t dim2Offset, uint64_t wAlign, uint64_t calcElems,
                                             uint64_t tempStride)
    {
        if constexpr (IS_CONTIGUOUS == false) {
            this->inParams_.blockCount = this->dim1Tile_;
            this->outParams_.blockCount = this->inParams_.blockCount;
        }
        uint64_t dim1Offset = dim2Offset;
        this->template CopyInDy<IS_CONTIGUOUS>(dim1Offset, calcElems);
        this->template CopyInX<IS_CONTIGUOUS>(dim1Offset, calcElems);
        ComputeLarge<true>(this->dim1Tile_, wAlign);
        this->template CopyOutZ<IS_CONTIGUOUS>(dim1Offset, calcElems);
        dim1Offset += tempStride;
        for (uint64_t i = 1; i < this->dim1LoopTime_; i++) {
            this->template CopyInDy<IS_CONTIGUOUS>(dim1Offset, calcElems);
            this->template CopyInX<IS_CONTIGUOUS>(dim1Offset, calcElems);
            ComputeLarge(this->dim1Tile_, wAlign);
            this->template CopyOutZ<IS_CONTIGUOUS>(dim1Offset, calcElems);
            dim1Offset += tempStride;
        }
        if (this->dim1Remained_) {
            if constexpr (IS_CONTIGUOUS == false) {
                this->inParams_.blockCount = this->dim1Remained_;
                this->outParams_.blockCount = this->inParams_.blockCount;
            }
            uint64_t remainedElems = this->dim1Remained_ * wAlign;
            this->template CopyInDy<IS_CONTIGUOUS>(dim1Offset, remainedElems);
            this->template CopyInX<IS_CONTIGUOUS>(dim1Offset, remainedElems);
            ComputeLarge(this->dim1Remained_, wAlign);
            this->template CopyOutZ<IS_CONTIGUOUS>(dim1Offset, remainedElems);
        }
    }

    template <bool IS_FIRST = false>
    __aicore__ inline void ComputeLarge(uint64_t h, uint64_t wAlign)
    {
        if constexpr (IS_FIRST) {
            BinaryAddReduceDy(t1_, this->dim1Tile_, wAlign);
        }
        uint64_t calcElems = h * wAlign;
        LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
        LocalTensor<float> x = this->inQueX_.template DeQue<float>();
        LocalTensor<float> z = this->outQueZ_.template AllocTensor<float>();
        Exp(x, x, calcElems);
        PipeBarrier<PIPE_V>();
        RectMul(x, t1_, h, wAlign);
        PipeBarrier<PIPE_V>();
        Sub(z, dy, x, calcElems);
        this->outQueZ_.EnQue(z);
        this->inQueX_.FreeTensor(x);
        this->inQueDy_.FreeTensor(dy);
    }

    __aicore__ inline void BinaryAddReduceDy(LocalTensor<float>& dst, uint64_t h, uint64_t wAlign)
    {
        uint64_t totalNum = h;
        while (totalNum > 1) {
            uint64_t halfNum = (totalNum + 1) >> 1;
            uint64_t calcElems = (totalNum - halfNum) * wAlign;
            PipeBarrier<PIPE_V>();
            Add(dst, dst, dst[halfNum * wAlign], calcElems);
            totalNum = halfNum;
        }
    }

    __aicore__ inline void BinaryAddReduceDy(LocalTensor<float>& dst, uint64_t n, uint64_t h, uint64_t wAlign)
    {
        uint64_t totalNum = h;
        while (totalNum > 1) {
            uint64_t halfNum = (totalNum + 1) >> 1;
            uint64_t calcElems = (totalNum - halfNum) * wAlign;
            PipeBarrier<PIPE_V>();
            for (uint64_t i = 0; i < n; i++) {
                LocalTensor<float> temp = dst[i * h * wAlign];
                Add(temp, temp, temp[halfNum * wAlign], calcElems);
            }
            totalNum = halfNum;
        }
    }

    __aicore__ inline void RectMul(LocalTensor<float>& a1, LocalTensor<float>& a2, uint64_t h, uint64_t wAlign)
    {
        if (wAlign <= MAX_WIDTH) {
            uint64_t wLoopTime = wAlign / FP32_ELEMS_PER_REPEAT;
            uint64_t wRemained = wAlign - wLoopTime * FP32_ELEMS_PER_REPEAT;
            uint64_t hLoopTime = h / MAX_REPEAT_TIME;
            uint64_t hRemained = h - hLoopTime * MAX_REPEAT_TIME;
            uint8_t wAlignBlkCnt = static_cast<uint8_t>(wAlign / FP32_ELEMS_PER_BLOCK);
            uint64_t wOffset = 0;
            for (uint64_t i = 0; i < wLoopTime; i++) {
                uint64_t hOffset = wOffset;
                for (uint64_t j = 0; j < hLoopTime; j++) {
                    Mul(a1[hOffset], a1[hOffset], a2[wOffset], FP32_ELEMS_PER_REPEAT, MAX_REPEAT_TIME,
                        {1, 1, 1, wAlignBlkCnt, wAlignBlkCnt, 0});
                    hOffset += MAX_REPEAT_TIME * wAlign;
                }
                if (hRemained) {
                    Mul(a1[hOffset], a1[hOffset], a2[wOffset], FP32_ELEMS_PER_REPEAT, hRemained,
                        {1, 1, 1, wAlignBlkCnt, wAlignBlkCnt, 0});
                }
                wOffset += FP32_ELEMS_PER_REPEAT;
            }
            if (wRemained) {
                uint64_t hOffset = wOffset;
                for (uint64_t j = 0; j < hLoopTime; j++) {
                    Mul(a1[hOffset], a1[hOffset], a2[wOffset], wRemained, MAX_REPEAT_TIME,
                        {1, 1, 1, wAlignBlkCnt, wAlignBlkCnt, 0});
                    hOffset += MAX_REPEAT_TIME * wAlign;
                }
                if (hRemained) {
                    Mul(a1[hOffset], a1[hOffset], a2[wOffset], wRemained, hRemained,
                        {1, 1, 1, wAlignBlkCnt, wAlignBlkCnt, 0});
                }
            }
        } else {
            int64_t offset = 0;
            for (uint64_t i = 0; i < h; i++) {
                Mul(a1[offset], a1[offset], a2, wAlign);
                offset += wAlign;
            }
        }
    }

private:
    TBuf<QuePosition::VECCALC> tempBuf_;
    LocalTensor<float> t1_;
    uint64_t dim2Bytes_;
    uint64_t blockNum_;
    uint64_t blockIdx_;
};

} // namespace NsLogSoftmaxGrad
#endif // __REDUCE_MID_H__
