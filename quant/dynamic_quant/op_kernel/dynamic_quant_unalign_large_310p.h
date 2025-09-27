/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file dynamic_quant_unalign_large_310p.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UNALIGN_LARGE_310P_H
#define DYNAMIC_QUANT_UNALIGN_LARGE_310P_H

#include "kernel_operator.h"
#include "dynamic_quant_align_310p.h"

namespace DynamicQuantNDOpt {

#if __CCE_AICORE__ < 220
constexpr uint32_t BLOCK_SIZE = 32;

class DynamicQuantUnalignLarge310P : public DynamicQuantAlign310p
{
public:
    __aicore__ inline DynamicQuantUnalignLarge310P()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, const DynamicQuantTilingData* __restrict tilingData)
    {
        InitParams(tilingData, offset);
        DynamicQuantAlign310p::InitBuffer(x, y, scale, offset);
        InitQueue();
    }

    __aicore__ inline void Process()
    {
        ProcessSymmetrical();
        ProcessSymmetricalTail();
    }

    __aicore__ inline void InitParams(const DynamicQuantTilingData* __restrict tilingData, GM_ADDR offset)
    {
        DynamicQuantAlign310p::InitParams(tilingData, offset);
        innerLoopEle_ = tilingData->innerLoopEle;
        innerLoopTimes_ = tilingData->innerLoopTimes;
        innerLoopTail_ = tilingData->innerLoopTail;
        this->lenCopyRow_ = this->innerLoopEle_ * this->numCopyRow_;
    }

    __aicore__ inline void InitQueue()
    {
        pipe_.InitBuffer(inQueue_, DYNAMIC_QUANT_BUFFER_NUM_TWO, lenCopyRow_ * DYNAMIC_QUANT_SIZE_OF_HALF);
        pipe_.InitBuffer(outQueue_, DYNAMIC_QUANT_BUFFER_NUM_ONE, lenCopyRow_ * DYNAMIC_QUANT_SIZE_OF_INT8);
        pipe_.InitBuffer(scaleQueue_, DYNAMIC_QUANT_BUFFER_NUM_ONE, numCopyRow_ * DYNAMIC_QUANT_SIZE_OF_FLOAT);
        pipe_.InitBuffer(fp16Buf_, innerLoopEle_ * DYNAMIC_QUANT_SIZE_OF_HALF);
        pipe_.InitBuffer(fp32Buf_, innerLoopEle_ * DYNAMIC_QUANT_SIZE_OF_FLOAT);
        if (asymmetric_) {
            pipe_.InitBuffer(offsetQueue_, DYNAMIC_QUANT_BUFFER_NUM_ONE, numCopyRow_ * DYNAMIC_QUANT_SIZE_OF_FLOAT);
        }
    }

private:
    __aicore__ inline void ComputeGetMaxValue(uint32_t innerLoopIndex, uint32_t calRow, uint32_t calCol)
    {
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<half> temp = fp16Buf_.Get<half>();
        AscendC::LocalTensor<half> inLocal = inQueue_.DeQue<half>();
        AscendC::LocalTensor<float> scaleLocal;
        uint32_t idx;
        float maxValue;
        if (innerLoopIndex == 0) {
            scaleLocal = scaleQueue_.AllocTensor<float>();
        } else {
            scaleLocal = scaleQueue_.DeQue<float>();
        }
        for (uint32_t i = 0; i < calRow; i++) {
            idx = i * calCol;
            Abs(temp, inLocal[idx], calCol);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceMax(temp, temp, temp, calCol, false);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            maxValue = static_cast<float>(temp.GetValue(0));
            if (innerLoopIndex == 0) {
                scaleLocal.SetValue(i, maxValue);
            } else {
                maxValue = (scaleLocal.GetValue(i) > maxValue) ? scaleLocal.GetValue(i) : maxValue;
                scaleLocal.SetValue(i, maxValue);
            }
        }
        scaleQueue_.EnQue<float>(scaleLocal);
        inQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void ComputeInnerTail(uint32_t calRow, uint32_t calCol)
    {
        // 尾块只需要搬运一次，直接量化生成结果
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
        AscendC::LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> temp = fp16Buf_.Get<half>();
        AscendC::LocalTensor<half> inLocal = inQueue_.DeQue<half>();

        uint32_t idx;
        float maxValue;
        float scale;
        for (uint32_t i = 0; i < calRow; i++) {
            idx = i * calCol;
            float rowMax = scaleLocal.GetValue(i);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            Abs(temp, inLocal[idx], calCol);
            AscendC::PipeBarrier<PIPE_V>();
            ReduceMax(temp, temp, temp, calCol, false);
            maxValue = static_cast<float>(temp.GetValue(0));
            maxValue = (rowMax > maxValue) ? rowMax : maxValue;
            if (maxValue <= DYNAMIC_QUANT_EPSINON) {
                scale = 0.0f;
            } else {
                scale = DYNAMIC_QUANT_MAX_VALUE / maxValue;
            }
            scaleLocal.SetValue(i, maxValue);
            Muls(inLocal[idx], inLocal[idx], static_cast<half>(scale), calCol);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outLocal[idx], inLocal[idx], AscendC::RoundMode::CAST_ROUND, calCol);
        }
        // enque the output tensor to VECOUT queue
        outQueue_.EnQue<int8_t>(outLocal);
        scaleQueue_.EnQue<float>(scaleLocal);
        inQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void QuantCompute(uint32_t calRow, uint32_t calCol)
    {
        // 尾块只需要搬运一次，直接量化生成结果
        // deque input tensors from VECIN queue
        AscendC::LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> temp = fp16Buf_.Get<half>();
        AscendC::LocalTensor<half> inLocal = inQueue_.DeQue<half>();
        AscendC::LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();

        uint32_t idx;
        float scale;
        for (uint32_t i = 0; i < calRow; i++) {
            idx = i * calCol;
            float rowMax = scaleLocal.GetValue(i);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
            if (rowMax <= DYNAMIC_QUANT_EPSINON || rowMax == 0.0f) {
                scale = 0.0f;
            } else {
                scale = DYNAMIC_QUANT_MAX_VALUE / rowMax;
            }
            Muls(inLocal[idx], inLocal[idx], static_cast<half>(scale), calCol);
            AscendC::PipeBarrier<PIPE_V>();
            Cast(outLocal[idx], inLocal[idx], AscendC::RoundMode::CAST_ROUND, calCol);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }
        // enque the output tensor to VECOUT queue
        outQueue_.EnQue<int8_t>(outLocal);
        inQueue_.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyIn(uint32_t CopyinOffset, AscendC::DataCopyParams& splitCopyinParams)
    {
        // alloc tensor from queue memory
        AscendC::LocalTensor<half> inLocal = inQueue_.AllocTensor<half>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(inLocal, inGm_[CopyinOffset], splitCopyinParams);
        // enque input tensors to VECIN queue
        inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void CopyOut(uint32_t CopyinOffset, AscendC::DataCopyParams& splitCopyoutParams)
    {
        // deque output tensor from VECOUT queue
        AscendC::LocalTensor<int8_t> outLocal = outQueue_.DeQue<int8_t>();
        // copy progress_th tile from local tensor to global tensor
        DataCopy(outGm_[CopyinOffset], outLocal, splitCopyoutParams);
        // free output tensor for reuse
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyScalesOut(uint32_t progress, uint32_t calRow)
    {
        // deque scale tensor from VECOUT queue
        AscendC::LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
        // copy progress_th tile from local tensor to global tensor
        Muls(scaleLocal, scaleLocal, static_cast<float>(1.0f / DYNAMIC_QUANT_MAX_VALUE), calRow);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(scaleGm_[progress * this->numCopyRow_], scaleLocal, this->numCopyRow_);
        // free scale tensor for reuse
        scaleQueue_.FreeTensor(scaleLocal);
    }

    __aicore__ inline void Compute(uint32_t offsetBase, uint32_t loopIndex, uint32_t calRow, uint32_t calCol)
    {
        uint32_t innerOffsetBase = 0;
        uint32_t srcOffset = 0;
        AscendC::DataCopyParams splitCopyinParams;
        AscendC::DataCopyParams splitCopyoutParams;
        // 计算每一块的最大值，按列切分
        for (uint32_t innerLoopIndex = 0; innerLoopIndex < this->innerLoopTimes_; innerLoopIndex++) {
            innerOffsetBase = offsetBase + innerLoopIndex * calCol;
            // 拷贝数据，一次拷贝8行数据数据起始地址innerOffsetBase，间隔sizeH_ - calCol，数据量calCol，
            splitCopyinParams = {
                static_cast<uint16_t>(calRow), static_cast<uint16_t>(calCol * sizeof(half) / BLOCK_SIZE),
                static_cast<uint16_t>((this->sizeH_ - calCol) * sizeof(half) / BLOCK_SIZE), 0};
            CopyIn(innerOffsetBase, splitCopyinParams);
            ComputeGetMaxValue(innerLoopIndex, calRow, calCol);
        }

        // 如果核内切的有尾块
        if (innerLoopTail_ > 0) {
            innerOffsetBase = offsetBase + this->innerLoopTimes_ * calCol;
            // 直接进行计算，可以最后一次尾块的搬运
            splitCopyinParams = {
                static_cast<uint16_t>(calRow), static_cast<uint16_t>(innerLoopTail_ * sizeof(half) / BLOCK_SIZE),
                static_cast<uint16_t>((this->sizeH_ - innerLoopTail_) * sizeof(half) / BLOCK_SIZE), 0};
            splitCopyoutParams = {
                static_cast<uint16_t>(calRow), static_cast<uint16_t>(innerLoopTail_ * sizeof(int8_t) / BLOCK_SIZE), 0,
                static_cast<uint16_t>((this->sizeH_ - innerLoopTail_) * sizeof(int8_t) / BLOCK_SIZE)};
            CopyIn(innerOffsetBase, splitCopyinParams);
            ComputeInnerTail(calRow, this->innerLoopTail_);
            CopyOut(innerOffsetBase, splitCopyoutParams);
        }

        // 量化计算
        for (uint32_t innerLoopIndex = 0; innerLoopIndex < this->innerLoopTimes_; innerLoopIndex++) {
            innerOffsetBase = offsetBase + innerLoopIndex * calCol;
            // 拷贝数据，一次拷贝8行数据数据起始地址innerOffsetBase，间隔sizeH_ - calCol，数据量calCol，
            splitCopyinParams = {
                static_cast<uint16_t>(calRow), static_cast<uint16_t>(calCol * sizeof(half) / BLOCK_SIZE),
                static_cast<uint16_t>((this->sizeH_ - calCol) * sizeof(half) / BLOCK_SIZE), 0};
            splitCopyoutParams = {
                static_cast<uint16_t>(calRow), static_cast<uint16_t>(calCol * sizeof(int8_t) / BLOCK_SIZE), 0,
                static_cast<uint16_t>((this->sizeH_ - calCol) * sizeof(int8_t) / BLOCK_SIZE)};
            CopyIn(innerOffsetBase, splitCopyinParams);
            QuantCompute(calRow, calCol);
            CopyOut(innerOffsetBase, splitCopyoutParams);
        }
        CopyScalesOut(loopIndex, calRow);
    }

    __aicore__ inline void ProcessSymmetrical()
    {
        uint32_t offsetBase = 0;
        // 循环每一批次，每8行一批次
        for (uint32_t i = 0; i < this->loopCount_; i++) {
            offsetBase = i * this->numCopyRow_ * this->sizeH_;
            Compute(offsetBase, i, this->numCopyRow_, innerLoopEle_);
        }
    }

    __aicore__ inline void ProcessSymmetricalTail()
    {
        uint32_t offsetBase = 0;
        // 计算尾巴
        if (isTailLoop_) {
            offsetBase = this->loopCount_ * this->numCopyRow_ * this->sizeH_;
            Compute(offsetBase, this->loopCount_, this->numLastTailRow_, innerLoopEle_);
        }
    }

public:
    uint32_t innerLoopEle_;   // 列方向切分的大小
    uint32_t innerLoopTimes_; // 列方向切分批次
    uint32_t innerLoopTail_;  // 每批列方向尾块
};
#endif
} // namespace DynamicQuantNDOpt
#endif // DYNAMIC_QUANT_UNALIGN_LARGE_310P_H