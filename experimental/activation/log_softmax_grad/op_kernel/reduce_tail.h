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
 * \file reduce_tail.h
 * \brief
 */
#ifndef __REDUCE_TAIL_H__
#define __REDUCE_TAIL_H__

#include "log_softmax_grad_base.h"

namespace NsLogSoftmaxGrad {

constexpr uint32_t REDUCE_MAX_REPEAT_TIME = 248;

template <typename T, bool IS_SMALL = false, bool IS_CONTIGUOUS = false, int BUF_NUM = 2>
class ReduceTail : public LogSoftmaxGradBase<T, BUF_NUM> {
public:
    __aicore__ inline ReduceTail() = default;

    __aicore__ inline void Init(LogSoftmaxGradTilingData& tiling, TPipe* pipePtr)
    {
        this->BaseInit(tiling, pipePtr);
        uint64_t singleBufSize = this->singleBufElems_ * sizeof(float);
        this->pipePtr_->InitBuffer(this->inQueDy_, BUF_NUM, singleBufSize);
        this->pipePtr_->InitBuffer(this->inQueX_, BUF_NUM, singleBufSize);
        this->pipePtr_->InitBuffer(this->outQueZ_, BUF_NUM, singleBufSize);
        this->pipePtr_->InitBuffer(tempBuf_, singleBufSize);
        t1_ = tempBuf_.Get<float>();

        if constexpr (IS_CONTIGUOUS == false) {
            this->inParams_.dstStride = DEFAULT_DATA_COPY_STRIDE;
            this->outParams_.srcStride = DEFAULT_DATA_COPY_STRIDE;
        }
    }

    __aicore__ inline void Process(GM_ADDR dy, GM_ADDR x, GM_ADDR z)
    {
        uint64_t blockNum = GetBlockNum();
        uint64_t blockIdx = GetBlockIdx();
        __gm__ T* dyAddrOffset = (__gm__ T*)dy;
        __gm__ T* xAddrOffset = (__gm__ T*)x;
        __gm__ T* zAddrOffset = (__gm__ T*)z;

        if constexpr (IS_CONTIGUOUS == false) {
            this->inParams_.blockCount = this->dim1Tile_;
            this->outParams_.blockCount = this->inParams_.blockCount;
            if constexpr (IS_SMALL) {
                this->inParams_.blockLen = this->dim2Tile_ * sizeof(T);
                this->inParams_.srcStride = 0;
                this->outParams_.blockLen = this->inParams_.blockLen;
                this->outParams_.dstStride = this->inParams_.srcStride;
            }
        }

        uint64_t dim1Stride = this->dim1Tile_ * this->mergedDim2_;
        for (uint64_t i = blockIdx; i < this->dim1LoopTime_; i += blockNum) {
            auto offset = i * dim1Stride;
            this->dyGM_.SetGlobalBuffer(dyAddrOffset + offset);
            this->xGM_.SetGlobalBuffer(xAddrOffset + offset);
            this->zGM_.SetGlobalBuffer(zAddrOffset + offset);
            if constexpr (IS_SMALL) {
                ProcSmallDim2(this->dim1Tile_);
            } else {
                ProcLargeDim2();
            }
        }
        if (this->dim1Remained_ && blockIdx == this->dim1LoopTime_ % blockNum) {
            if constexpr (IS_CONTIGUOUS == false) {
                this->inParams_.blockCount = this->dim1Remained_;
                this->outParams_.blockCount = this->inParams_.blockCount;
            }
            auto offset = this->dim1LoopTime_ * dim1Stride;
            this->dyGM_.SetGlobalBuffer(dyAddrOffset + offset);
            this->xGM_.SetGlobalBuffer(xAddrOffset + offset);
            this->zGM_.SetGlobalBuffer(zAddrOffset + offset);
            if constexpr (IS_SMALL) {
                ProcSmallDim2(this->dim1Remained_);
            } else {
                ProcLargeDim2();
            }
        }
    }

private:
    __aicore__ inline void ProcSmallDim2(uint64_t h)
    {
        constexpr uint64_t elemsPerBlock = BLOCK_SIZE / sizeof(T);
        uint64_t wAlign = 0;
        if constexpr (IS_CONTIGUOUS) {
            wAlign = this->dim2Tile_;
        } else {
            wAlign = AlignUp(this->dim2Tile_, elemsPerBlock);
        }
        uint64_t copyNum = h * wAlign;
        this->template CopyInX<IS_CONTIGUOUS>(0, copyNum);
        this->template CopyInDy<IS_CONTIGUOUS>(0, copyNum);
        if (IS_CONTIGUOUS && this->dim2Tile_ % FP32_ELEMS_PER_BLOCK) {
            ComputeSmallUseTrans(h, this->dim2Tile_);
        } else {
            ComputeSmall(h, this->dim2Tile_, wAlign);
        }
        this->template CopyOutZ<IS_CONTIGUOUS>(0, copyNum);
    }

    __aicore__ inline void ComputeSmallUseTrans(uint64_t h, uint64_t w)
    {
        uint64_t calcElems = h * w;
        LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
        LocalTensor<float> x = this->inQueX_.template DeQue<float>();
        LocalTensor<float> z = this->outQueZ_.template AllocTensor<float>();
        Exp(x, x, calcElems);
        TransAndBinaryAdd(z, dy, t1_, h, w);
        PipeBarrier<PIPE_V>();
        Mul(x, x, z, calcElems);
        PipeBarrier<PIPE_V>();
        Sub(z, dy, x, calcElems);
        this->outQueZ_.EnQue(z);
        this->inQueX_.FreeTensor(x);
        this->inQueDy_.FreeTensor(dy);
    }

    __aicore__ inline void ComputeSmall(uint64_t h, uint64_t w, uint64_t wAlign)
    {
        uint64_t calcElems = h * wAlign;
        LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
        LocalTensor<float> x = this->inQueX_.template DeQue<float>();
        LocalTensor<float> z = this->outQueZ_.template AllocTensor<float>();
        DataCopy(z, dy, calcElems);
        Exp(x, x, calcElems);
        BinaryAddReduceDy(t1_, z, h, w, wAlign);
        PipeBarrier<PIPE_V>();
        BroadcastSum(z, t1_, h);
        PipeBarrier<PIPE_V>();
        RectMul(x, z, h, wAlign);
        PipeBarrier<PIPE_V>();
        Sub(z, dy, x, calcElems);
        this->outQueZ_.EnQue(z);
        this->inQueX_.FreeTensor(x);
        this->inQueDy_.FreeTensor(dy);
    }

    __aicore__ inline void ProcLargeDim2()
    {
        Duplicate(t1_, float(0.0), this->singleBufElems_);
        uint64_t dim2Offset = 0;
        for (uint64_t i = 0; i < this->dim2LoopTime_; i++) {
            this->template CopyInDy<IS_CONTIGUOUS>(dim2Offset, this->singleBufElems_);
            LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
            Add(t1_, t1_, dy, this->singleBufElems_);
            this->inQueDy_.FreeTensor(dy);
            dim2Offset += this->singleBufElems_;
        }
        if (this->dim2Remained_) {
            this->template CopyInDy<IS_CONTIGUOUS>(dim2Offset, this->dim2Remained_);
            LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
            Add(t1_, t1_, dy, this->dim2Remained_);
            this->inQueDy_.FreeTensor(dy);
        }

        dim2Offset = 0;
        this->template CopyInDy<IS_CONTIGUOUS>(dim2Offset, this->singleBufElems_);
        this->template CopyInX<IS_CONTIGUOUS>(dim2Offset, this->singleBufElems_);
        ComputeLarge<true>(this->singleBufElems_);
        this->template CopyOutZ<IS_CONTIGUOUS>(dim2Offset, this->singleBufElems_);
        dim2Offset += this->singleBufElems_;
        for (uint64_t i = 1; i < this->dim2LoopTime_; i++) {
            this->template CopyInDy<IS_CONTIGUOUS>(dim2Offset, this->singleBufElems_);
            this->template CopyInX<IS_CONTIGUOUS>(dim2Offset, this->singleBufElems_);
            ComputeLarge(this->singleBufElems_);
            this->template CopyOutZ<IS_CONTIGUOUS>(dim2Offset, this->singleBufElems_);
            dim2Offset += this->singleBufElems_;
        }
        if (this->dim2Remained_) {
            this->template CopyInDy<IS_CONTIGUOUS>(dim2Offset, this->dim2Remained_);
            this->template CopyInX<IS_CONTIGUOUS>(dim2Offset, this->dim2Remained_);
            ComputeLarge(this->dim2Remained_);
            this->template CopyOutZ<IS_CONTIGUOUS>(dim2Offset, this->dim2Remained_);
        }
    }

    template <bool IS_FIRST = false>
    __aicore__ inline void ComputeLarge(uint64_t w)
    {
        LocalTensor<float> z = this->outQueZ_.template AllocTensor<float>();
        if constexpr (IS_FIRST) {
            BinaryAddReduceDy(t1_, t1_, this->singleBufElems_);
            PipeBarrier<PIPE_V>();
            Brcb(z, t1_, 1, {1, 0});
            PipeBarrier<PIPE_V>();
            DataCopy(t1_, z, FP32_ELEMS_PER_BLOCK);
            PipeBarrier<PIPE_V>();
        }
        LocalTensor<float> dy = this->inQueDy_.template DeQue<float>();
        LocalTensor<float> x = this->inQueX_.template DeQue<float>();
        Exp(x, x, w);
        PipeBarrier<PIPE_V>();
        RectMul(x, t1_, w);
        PipeBarrier<PIPE_V>();
        Sub(z, dy, x, w);
        this->outQueZ_.EnQue(z);
        this->inQueX_.FreeTensor(x);
        this->inQueDy_.FreeTensor(dy);
    }

    __aicore__ inline void TransAndBinaryAdd(LocalTensor<float>& dst, LocalTensor<float>& src, LocalTensor<float>& temp,
                                             uint64_t h, uint64_t w)
    {
        constexpr uint64_t transLen = 16;
        constexpr uint64_t transHalfLen = transLen / 2;
        uint64_t newH = DivCeil(h, FP32_ELEMS_PER_BLOCK);
        uint64_t newW = w * FP32_ELEMS_PER_BLOCK;
        uint64_t newHAlign = AlignUp(newH, transLen);
        TransDataTo5HDParams transParams;
        transParams.repeatTimes = static_cast<uint8_t>(w);
        transParams.srcRepStride = 1;
        transParams.dstRepStride = newHAlign;
        uint64_t srcLocalList[transLen];
        uint64_t dstLocalList[transLen];
        for (uint64_t hOffset = 0; hOffset < newHAlign; hOffset += transLen) {
            auto srcOffset = hOffset * newW;
            for (uint64_t i = 0; i < transLen; i++) {
                srcLocalList[i] = src.GetPhyAddr(srcOffset);
                srcOffset += newW;
            }
            auto dstOffset = hOffset;
            for (uint64_t i = 0; i < transLen; i += 2) {
                dstLocalList[i] = dst.GetPhyAddr(dstOffset);
                dstLocalList[i + 1] = dst.GetPhyAddr(dstOffset + transHalfLen);
                dstOffset += newHAlign;
            }
            TransDataTo5HD<float>(dstLocalList, srcLocalList, transParams);
        }
        BinaryAddReduceAndCopy(dst, temp, w, newHAlign);
        transParams.repeatTimes = static_cast<uint8_t>(w);
        transParams.srcRepStride = newHAlign;
        transParams.dstRepStride = 1;
        PipeBarrier<PIPE_V>();
        for (uint64_t hOffset = 0; hOffset < newHAlign; hOffset += transLen) {
            auto srcOffset = hOffset;
            for (uint64_t i = 0; i < transHalfLen; i++) {
                srcLocalList[i] = temp.GetPhyAddr(srcOffset);
                srcLocalList[i + transHalfLen] = temp.GetPhyAddr(srcOffset + transHalfLen);
                srcOffset += newHAlign;
            }
            auto dstOffset = hOffset * newW;
            for (uint64_t i = 0; i < transLen; i += 2) {
                dstLocalList[i] = dst.GetPhyAddr(dstOffset);
                dstLocalList[i + 1] = dst.GetPhyAddr(dstOffset + transHalfLen * newW);
                dstOffset += newW;
            }
            TransDataTo5HD<float>(dstLocalList, srcLocalList, transParams);
        }
    }

    __aicore__ inline void BinaryAddReduceAndCopy(LocalTensor<float>& dst, LocalTensor<float>& temp, uint64_t w,
                                                  uint64_t newHAlign)
    {
        uint64_t totalNum = w;
        while (totalNum > 1) {
            uint64_t halfNum = (totalNum + 1) >> 1;
            uint64_t calcElems = (totalNum - halfNum) * newHAlign;
            PipeBarrier<PIPE_V>();
            for (uint64_t i = 0; i < FP32_ELEMS_PER_BLOCK; i++) {
                auto baseOffset = i * w * newHAlign;
                Add(dst[baseOffset], dst[baseOffset], dst[baseOffset + halfNum * newHAlign], calcElems);
            }
            totalNum = halfNum;
        }
        PipeBarrier<PIPE_V>();
        for (uint64_t hOffset = 0; hOffset < newHAlign; hOffset += FP32_ELEMS_PER_REPEAT) {
            uint64_t maskLen = GetMin(newHAlign - hOffset, FP32_ELEMS_PER_REPEAT);
            for (uint64_t i = 0; i < FP32_ELEMS_PER_BLOCK; i++) {
                auto baseOffset = hOffset + i * w * newHAlign;
                Copy(temp[baseOffset], dst[baseOffset], maskLen, w,
                     {1, 1, static_cast<uint16_t>(newHAlign / FP32_ELEMS_PER_BLOCK), 0});
            }
        }
    }

    __aicore__ inline void BinaryAddReduceDy(LocalTensor<float>& result, LocalTensor<float>& input, uint64_t w)
    {
        constexpr uint64_t elemsDoubleRepeat = 2 * FP32_ELEMS_PER_REPEAT;
        uint64_t totalNum = w;
        while (totalNum > FP32_ELEMS_PER_REPEAT) {
            uint64_t halfNum = DivCeil(totalNum, elemsDoubleRepeat) * FP32_ELEMS_PER_REPEAT;
            PipeBarrier<PIPE_V>();
            Add(input, input, input[halfNum], totalNum - halfNum);
            totalNum = halfNum;
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(result, input, totalNum, 1, 1, 1, 0);
    }

    __aicore__ inline void BinaryAddReduceDy(LocalTensor<float>& result, LocalTensor<float>& input, uint64_t h,
                                             uint64_t w, uint64_t wAlign)
    {
        constexpr uint64_t elemsDoubleRepeat = 2 * FP32_ELEMS_PER_REPEAT;
        uint64_t hLoopTime = h / REDUCE_MAX_REPEAT_TIME;
        uint64_t hRemained = h - hLoopTime * REDUCE_MAX_REPEAT_TIME;
        uint64_t hRepeatStride = REDUCE_MAX_REPEAT_TIME * wAlign;
        uint8_t wBlkCount = static_cast<uint8_t>(wAlign / FP32_ELEMS_PER_BLOCK);
        uint64_t totalNum = w;
        totalNum = ReduceLargeWidth(input, totalNum, h, wAlign, elemsDoubleRepeat);
        if (totalNum > FP32_ELEMS_PER_REPEAT) {
            uint64_t offset = 0;
            PipeBarrier<PIPE_V>();
            for (uint64_t i = 0; i < hLoopTime; i++) {
                LocalTensor<float> temp = input[offset];
                Add(temp, temp, temp[FP32_ELEMS_PER_REPEAT], totalNum - FP32_ELEMS_PER_REPEAT, REDUCE_MAX_REPEAT_TIME,
                    {1, 1, 1, wBlkCount, wBlkCount, wBlkCount});
                offset += hRepeatStride;
            }
            if (hRemained) {
                LocalTensor<float> temp = input[offset];
                Add(temp, temp, temp[FP32_ELEMS_PER_REPEAT], totalNum - FP32_ELEMS_PER_REPEAT, hRemained,
                    {1, 1, 1, wBlkCount, wBlkCount, wBlkCount});
            }
            totalNum = FP32_ELEMS_PER_REPEAT;
        }
        ReduceSmallWidth(result, input, totalNum, hLoopTime, hRemained, hRepeatStride, wBlkCount);
    }

    __aicore__ inline uint64_t ReduceLargeWidth(LocalTensor<float>& input, uint64_t totalNum, uint64_t h,
                                                uint64_t wAlign, uint64_t elemsDoubleRepeat)
    {
        while (totalNum > elemsDoubleRepeat) {
            uint64_t halfNum = DivCeil(totalNum, elemsDoubleRepeat) * FP32_ELEMS_PER_REPEAT;
            uint64_t offset = 0;
            PipeBarrier<PIPE_V>();
            for (uint64_t i = 0; i < h; i++) {
                LocalTensor<float> temp = input[offset];
                Add(temp, temp, temp[halfNum], totalNum - halfNum);
                offset += wAlign;
            }
            totalNum = halfNum;
        }
        return totalNum;
    }

    __aicore__ inline void ReduceSmallWidth(LocalTensor<float>& result, LocalTensor<float>& input, uint64_t totalNum,
                                            uint64_t hLoopTime, uint64_t hRemained, uint64_t hRepeatStride,
                                            uint8_t wBlkCount)
    {
        uint64_t hOffset = 0;
        uint64_t offset = 0;
        PipeBarrier<PIPE_V>();
        for (uint64_t i = 0; i < hLoopTime; i++) {
            LocalTensor<float> temp = input[offset];
            WholeReduceSum(result[hOffset], temp, totalNum, REDUCE_MAX_REPEAT_TIME, 1, 1, wBlkCount);
            offset += hRepeatStride;
            hOffset += REDUCE_MAX_REPEAT_TIME;
        }
        if (hRemained) {
            LocalTensor<float> temp = input[offset];
            WholeReduceSum(result[hOffset], temp, totalNum, hRemained, 1, 1, wBlkCount);
        }
    }

    __aicore__ inline void BroadcastSum(LocalTensor<float>& dst, LocalTensor<float>& src, uint64_t h)
    {
        constexpr uint64_t dstStride = MAX_REPEAT_TIME * FP32_ELEMS_PER_REPEAT;
        constexpr uint64_t srcStride = MAX_REPEAT_TIME * FP32_ELEMS_PER_BLOCK;
        uint64_t hBlkCount = DivCeil(h, FP32_ELEMS_PER_BLOCK);
        uint64_t loopTime = hBlkCount / MAX_REPEAT_TIME;
        uint64_t remained = hBlkCount - loopTime * MAX_REPEAT_TIME;
        uint64_t srcOffset = 0;
        uint64_t dstOffset = 0;
        for (uint64_t i = 0; i < loopTime; i++) {
            Brcb(dst[dstOffset], src[srcOffset], MAX_REPEAT_TIME, {1, 8});
            dstOffset += dstStride;
            srcOffset += srcStride;
        }
        if (remained) {
            Brcb(dst[dstOffset], src[srcOffset], remained, {1, 8});
        }
    }

    __aicore__ inline void RectMul(LocalTensor<float>& a1, LocalTensor<float>& a2, uint64_t w)
    {
        constexpr uint64_t stride = MAX_REPEAT_TIME * FP32_ELEMS_PER_REPEAT;
        uint64_t wLoopTime = w / stride;
        uint64_t wTail = w - wLoopTime * stride;
        uint64_t wTailRepeatTime = wTail / FP32_ELEMS_PER_REPEAT;
        uint64_t wTailRemained = wTail - wTailRepeatTime * FP32_ELEMS_PER_REPEAT;
        uint64_t wOffset = 0;
        for (uint64_t i = 0; i < wLoopTime; i++) {
            Mul(a1[wOffset], a1[wOffset], a2, FP32_ELEMS_PER_REPEAT, MAX_REPEAT_TIME, {1, 1, 0, 8, 8, 0});
            wOffset += stride;
        }
        if (wTailRepeatTime) {
            Mul(a1[wOffset], a1[wOffset], a2, FP32_ELEMS_PER_REPEAT, wTailRepeatTime, {1, 1, 0, 8, 8, 0});
            wOffset += wTailRepeatTime * FP32_ELEMS_PER_REPEAT;
        }
        if (wTailRemained) {
            Mul(a1[wOffset], a1[wOffset], a2, wTailRemained, 1, {1, 1, 0, 0, 0, 0});
        }
    }

    __aicore__ inline void RectMul(LocalTensor<float>& a1, LocalTensor<float>& a2, uint64_t h, uint64_t wAlign)
    {
        if (wAlign <= MAX_WIDTH) {
            RectMulSmallWidth(a1, a2, h, wAlign);
        } else {
            RectMulLargeWidth(a1, a2, h, wAlign);
        }
    }

    __aicore__ inline void RectMulSmallWidth(LocalTensor<float>& a1, LocalTensor<float>& a2, uint64_t h,
                                             uint64_t wAlign)
    {
        uint64_t wLoopTime = wAlign / FP32_ELEMS_PER_REPEAT;
        uint64_t wRemained = wAlign - wLoopTime * FP32_ELEMS_PER_REPEAT;
        uint64_t hLoopTime = h / MAX_REPEAT_TIME;
        uint64_t hRemained = h - hLoopTime * MAX_REPEAT_TIME;
        uint8_t wBlkCount = static_cast<uint8_t>(wAlign / FP32_ELEMS_PER_BLOCK);
        uint64_t wOffset = 0;
        for (uint64_t i = 0; i < wLoopTime; i++) {
            uint64_t hOffset = wOffset;
            uint64_t sumOffset = 0;
            for (uint64_t j = 0; j < hLoopTime; j++) {
                Mul(a1[hOffset], a1[hOffset], a2[sumOffset], FP32_ELEMS_PER_REPEAT, MAX_REPEAT_TIME,
                    {1, 1, 0, wBlkCount, wBlkCount, 1});
                hOffset += MAX_REPEAT_TIME * wAlign;
                sumOffset += MAX_REPEAT_TIME * FP32_ELEMS_PER_BLOCK;
            }
            if (hRemained) {
                Mul(a1[hOffset], a1[hOffset], a2[sumOffset], FP32_ELEMS_PER_REPEAT, hRemained,
                    {1, 1, 0, wBlkCount, wBlkCount, 1});
            }
            wOffset += FP32_ELEMS_PER_REPEAT;
        }

        if (wRemained) {
            uint64_t hOffset = wOffset;
            uint64_t sumOffset = 0;
            for (uint64_t j = 0; j < hLoopTime; j++) {
                Mul(a1[hOffset], a1[hOffset], a2[sumOffset], wRemained, MAX_REPEAT_TIME,
                    {1, 1, 0, wBlkCount, wBlkCount, 1});
                hOffset += MAX_REPEAT_TIME * wAlign;
                sumOffset += MAX_REPEAT_TIME * FP32_ELEMS_PER_BLOCK;
            }
            if (hRemained) {
                Mul(a1[hOffset], a1[hOffset], a2[sumOffset], wRemained, hRemained, {1, 1, 0, wBlkCount, wBlkCount, 1});
            }
        }
    }

    __aicore__ inline void RectMulLargeWidth(LocalTensor<float>& a1, LocalTensor<float>& a2, uint64_t h,
                                             uint64_t wAlign)
    {
        constexpr uint64_t stride = MAX_REPEAT_TIME * FP32_ELEMS_PER_REPEAT;
        uint64_t wLoopTime = wAlign / stride;
        uint64_t wTail = wAlign - wLoopTime * stride;
        uint64_t wTailRepeatTime = wTail / FP32_ELEMS_PER_REPEAT;
        uint64_t wTailRemained = wTail - wTailRepeatTime * FP32_ELEMS_PER_REPEAT;
        for (uint64_t i = 0; i < h; i++) {
            uint64_t sumOffset = 0;
            uint64_t wOffset = 0;
            for (uint64_t j = 0; j < wLoopTime; j++) {
                Mul(a1[wOffset], a1[wOffset], a2[sumOffset], FP32_ELEMS_PER_REPEAT, MAX_REPEAT_TIME,
                    {1, 1, 0, 8, 8, 0});
                wOffset += stride;
            }
            if (wTailRepeatTime) {
                Mul(a1[wOffset], a1[wOffset], a2[sumOffset], FP32_ELEMS_PER_REPEAT, wTailRepeatTime,
                    {1, 1, 0, 8, 8, 0});
                wOffset += wTailRepeatTime * FP32_ELEMS_PER_REPEAT;
            }
            if (wTailRemained) {
                Mul(a1[wOffset], a1[wOffset], a2[sumOffset], wTailRemained, 1, {1, 1, 0, 0, 0, 0});
            }
            sumOffset += FP32_ELEMS_PER_BLOCK;
        }
    }

    __aicore__ inline void RectAdd(LocalTensor<float>& a1, LocalTensor<float>& a2, uint64_t h, uint64_t w,
                                   uint64_t wAlign)
    {
        if (wAlign <= MAX_WIDTH) {
            uint64_t wLoopTime = w / FP32_ELEMS_PER_REPEAT;
            uint64_t wRemained = w - wLoopTime * FP32_ELEMS_PER_REPEAT;
            uint64_t hLoopTime = h / MAX_REPEAT_TIME;
            uint64_t hRemained = h - hLoopTime * MAX_REPEAT_TIME;
            uint8_t wBlkCount = static_cast<uint8_t>(wAlign / FP32_ELEMS_PER_BLOCK);
            uint64_t wOffset = 0;
            for (uint64_t i = 0; i < wLoopTime; i++) {
                uint64_t hOffset = wOffset;
                for (uint64_t j = 0; j < hLoopTime; j++) {
                    Add(a1[hOffset], a1[hOffset], a2[hOffset], FP32_ELEMS_PER_REPEAT, MAX_REPEAT_TIME,
                        {1, 1, 1, wBlkCount, wBlkCount, wBlkCount});
                    hOffset += MAX_REPEAT_TIME * wAlign;
                }
                if (hRemained) {
                    Add(a1[hOffset], a1[hOffset], a2[hOffset], FP32_ELEMS_PER_REPEAT, hRemained,
                        {1, 1, 1, wBlkCount, wBlkCount, wBlkCount});
                }
                wOffset += FP32_ELEMS_PER_REPEAT;
            }

            if (wRemained) {
                uint64_t hOffset = wOffset;
                for (uint64_t j = 0; j < hLoopTime; j++) {
                    Add(a1[hOffset], a1[hOffset], a2[hOffset], wRemained, MAX_REPEAT_TIME,
                        {1, 1, 1, wBlkCount, wBlkCount, wBlkCount});
                    hOffset += MAX_REPEAT_TIME * wAlign;
                }
                if (hRemained) {
                    Add(a1[hOffset], a1[hOffset], a2[hOffset], wRemained, hRemained,
                        {1, 1, 1, wBlkCount, wBlkCount, wBlkCount});
                }
            }
        } else {
            int64_t offset = 0;
            for (uint64_t i = 0; i < h; i++) {
                Add(a1[offset], a1[offset], a2[offset], w);
                offset += wAlign;
            }
        }
    }

private:
    TBuf<QuePosition::VECCALC> tempBuf_;
    LocalTensor<float> t1_;
};

} // namespace NsLogSoftmaxGrad
#endif // __REDUCE_TAIL_H__
