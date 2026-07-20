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
 * \file masked_scatter.h
 * \brief kernel implementation of masked_scatter, translated from TBE TIK version
 */

#ifndef __MASKED_SCATTER_H__
#define __MASKED_SCATTER_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "masked_scatter_tiling_data.h"
#include "masked_scatter_tiling_key.h"

namespace NsMaskedScatter {

using namespace AscendC;

constexpr int64_t BYTES_PER_BLOCK = 32;
constexpr int64_t MAX_VEC_PROCESS_NUM = 64;
constexpr int64_t TASK_ALIGN = 4096;
constexpr int64_t COUNT_REDUCE_LEN = TASK_ALIGN;

constexpr int64_t DTYPE_BYTES_FLOAT16 = 2;
constexpr int64_t DTYPE_BYTES_FLOAT32 = 4;
constexpr int64_t DTYPE_BYTES_INT32 = 4;
constexpr int64_t DTYPE_BYTES_INT16 = 2;
constexpr int64_t DTYPE_BYTES_INT8 = 1;
constexpr int64_t DTYPE_BYTES_UINT8 = 1;
constexpr int64_t DTYPE_BYTES_BOOL = 1;
constexpr int64_t DTYPE_BYTES_BF16 = 2;

template <typename DTYPE>
class MaskedScatter {
public:
    __aicore__ inline MaskedScatter() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, const MaskedScatterTilingData* t);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b);
    __aicore__ inline int64_t AlignDiv(int64_t a, int64_t b);
    __aicore__ inline int64_t GetTaskBlockSize();
    __aicore__ inline int64_t CalcUpdatesStart(int64_t numElemMask);
    __aicore__ inline int64_t CountMaskNonZero(AscendC::LocalTensor<uint8_t>& maskUb, int64_t dataLen,
                                               int64_t alignedLen);
    __aicore__ inline int64_t ComputeUnrolled(int64_t inputOffset, int64_t updatesStart);
    __aicore__ inline void ScatterOne(AscendC::LocalTensor<DTYPE>& yUb, AscendC::LocalTensor<int8_t>& maskUb,
                                      AscendC::LocalTensor<DTYPE>& updatesUb, int64_t outputOffset,
                                      int64_t updatesBackOffset, int64_t numRemainUpdates, int64_t& updatesOffsetUb);
    __aicore__ inline int64_t ScatterUpdates(AscendC::LocalTensor<DTYPE>& yUb, AscendC::LocalTensor<int8_t>& maskUb,
                                             int64_t numElemPerInput, int64_t updatesStart, int64_t numRemainUpdates);
    __aicore__ inline int64_t Compute(int64_t inputOffset, int64_t updatesStart);
    __aicore__ inline int64_t GetDtypeBytesSize();

    AscendC::GlobalTensor<DTYPE> xGm_;
    AscendC::GlobalTensor<int8_t> maskGm_;
    AscendC::GlobalTensor<uint8_t> maskGmUint8_;
    AscendC::GlobalTensor<DTYPE> updatesGm_;
    AscendC::GlobalTensor<DTYPE> yGm_;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueMask_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueUpdates_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> boolBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> fp16Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> fp32Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sumBuf_;

    int64_t numElemX_;
    int64_t numElemMask_;
    int64_t numElemUpdates_;
    int64_t tilingCoreNum_;
    int64_t alignedElemPerCore_;
    int64_t dtypeBytesSize_;
    int64_t numElemPerBlock_;
    int64_t logiTaskNum_;
    int64_t logiTaskNumPerAicore_;
    int64_t logiTaskTail_;
};

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::GetDtypeBytesSize()
{
    return sizeof(DTYPE);
}

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::CeilDiv(int64_t a, int64_t b)
{
    return ((a + b - 1) / b);
}

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::AlignDiv(int64_t a, int64_t b)
{
    if (b != 0) {
        return ((a + b - 1) / b) * b;
    }
    return 0;
}

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::GetTaskBlockSize()
{
    int64_t blockSizeAligned = MAX_VEC_PROCESS_NUM;
    if (tilingCoreNum_ != 0) {
        int64_t blockSize = numElemX_ / tilingCoreNum_;
        blockSizeAligned = AlignDiv(blockSize, MAX_VEC_PROCESS_NUM);
    }
    if (numElemX_ >= tilingCoreNum_ * TASK_ALIGN) {
        blockSizeAligned = TASK_ALIGN;
    }
    if (numElemX_ <= MAX_VEC_PROCESS_NUM) {
        blockSizeAligned = MAX_VEC_PROCESS_NUM;
    }
    return blockSizeAligned;
}

template <typename DTYPE>
__aicore__ inline void MaskedScatter<DTYPE>::Init(GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y,
                                                  const MaskedScatterTilingData* t)
{
    xGm_.SetGlobalBuffer((__gm__ DTYPE*)x);
    maskGm_.SetGlobalBuffer((__gm__ int8_t*)mask);
    maskGmUint8_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    updatesGm_.SetGlobalBuffer((__gm__ DTYPE*)updates);
    yGm_.SetGlobalBuffer((__gm__ DTYPE*)y);

    numElemX_ = t->numElemX;
    numElemMask_ = t->numElemMask;
    numElemUpdates_ = t->numElemUpdates;
    tilingCoreNum_ = t->tilingCoreNum;

    dtypeBytesSize_ = GetDtypeBytesSize();
    numElemPerBlock_ = BYTES_PER_BLOCK / dtypeBytesSize_;

    alignedElemPerCore_ = GetTaskBlockSize();
    logiTaskNum_ = CeilDiv(numElemX_, alignedElemPerCore_);
    logiTaskNumPerAicore_ = logiTaskNum_ / tilingCoreNum_;
    logiTaskTail_ = logiTaskNum_ % tilingCoreNum_;

    pipe_.InitBuffer(inQueueX_, 1, TASK_ALIGN * dtypeBytesSize_);
    pipe_.InitBuffer(inQueueMask_, 1, TASK_ALIGN * DTYPE_BYTES_BOOL);
    pipe_.InitBuffer(inQueueUpdates_, 1, TASK_ALIGN * dtypeBytesSize_);
    pipe_.InitBuffer(outQueueY_, 1, TASK_ALIGN * dtypeBytesSize_);
    pipe_.InitBuffer(boolBuf_, TASK_ALIGN * DTYPE_BYTES_BOOL);
    pipe_.InitBuffer(fp16Buf_, TASK_ALIGN * DTYPE_BYTES_FLOAT16);
    pipe_.InitBuffer(fp32Buf_, TASK_ALIGN * DTYPE_BYTES_FLOAT32);
    pipe_.InitBuffer(sumBuf_, TASK_ALIGN * DTYPE_BYTES_FLOAT32);
}

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::CountMaskNonZero(AscendC::LocalTensor<uint8_t>& maskUb, int64_t dataLen,
                                                                 int64_t alignedLen)
{
    int64_t reduceAlignedLen = AlignDiv(dataLen, MAX_VEC_PROCESS_NUM);
    if (reduceAlignedLen > dataLen) {
        for (int64_t i = dataLen; i < reduceAlignedLen; i++) {
            maskUb.SetValue(i, 0);
        }
    }

    AscendC::LocalTensor<half> fp16Ub = fp16Buf_.Get<half>();
    AscendC::Cast(fp16Ub, maskUb, RoundMode::CAST_NONE, static_cast<uint32_t>(reduceAlignedLen));
    PipeBarrier<PIPE_V>();

    AscendC::LocalTensor<float> fp32Ub = fp32Buf_.Get<float>();
    AscendC::Duplicate(fp32Ub, 0.0f, static_cast<uint32_t>(reduceAlignedLen));
    PipeBarrier<PIPE_V>();
    AscendC::Cast(fp32Ub, fp16Ub, RoundMode::CAST_NONE, static_cast<uint32_t>(reduceAlignedLen));
    PipeBarrier<PIPE_V>();

    AscendC::Abs(fp32Ub, fp32Ub, static_cast<int32_t>(reduceAlignedLen));
    PipeBarrier<PIPE_V>();
    AscendC::Mins(fp32Ub, fp32Ub, 1.0f, static_cast<uint32_t>(reduceAlignedLen));
    PipeBarrier<PIPE_V>();

    AscendC::LocalTensor<float> sumUb = sumBuf_.Get<float>();
    int64_t repeatTimes = CeilDiv(reduceAlignedLen, MAX_VEC_PROCESS_NUM);
    AscendC::WholeReduceSum<float>(sumUb, fp32Ub, MAX_VEC_PROCESS_NUM, static_cast<int32_t>(repeatTimes), 1, 1,
                                   AscendC::DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float>(sumUb, sumUb, static_cast<int32_t>(repeatTimes), 1, 1, 1,
                                   AscendC::DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);

    float sumResFp32 = sumUb.GetValue(0);
    return static_cast<int64_t>(static_cast<int32_t>(sumResFp32 + 0.5f));
}

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::CalcUpdatesStart(int64_t numElemMask)
{
    int64_t sumCount = 0;

    int64_t dataMoveLength = COUNT_REDUCE_LEN;
    if (numElemMask < COUNT_REDUCE_LEN && numElemMask > 0) {
        dataMoveLength = numElemMask;
    }
    int64_t iters = CeilDiv(numElemMask, dataMoveLength);

    int64_t burst = CeilDiv(dataMoveLength * DTYPE_BYTES_BOOL, BYTES_PER_BLOCK);
    int64_t remainLength = numElemMask - (iters - 1) * dataMoveLength;

    for (int64_t ubIdx = 0; ubIdx < iters; ubIdx++) {
        int64_t offset = CeilDiv(dataMoveLength * DTYPE_BYTES_BOOL, BYTES_PER_BLOCK) * BYTES_PER_BLOCK * ubIdx;
        int64_t curBurst = burst;
        int64_t curDataLen = dataMoveLength;

        if (ubIdx == iters - 1 && remainLength > 0) {
            curBurst = CeilDiv(remainLength * DTYPE_BYTES_BOOL, BYTES_PER_BLOCK);
            curDataLen = remainLength;
        }

        AscendC::LocalTensor<uint8_t> maskUb = boolBuf_.Get<uint8_t>();
        AscendC::DataCopyExtParams maskParams{1, static_cast<uint32_t>(curBurst * BYTES_PER_BLOCK), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> maskPadParams{true, 0, 0, 0};
        AscendC::DataCopyPad(maskUb, maskGmUint8_[offset], maskParams, maskPadParams);
        event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);

        int64_t alignedDataLen = curBurst * BYTES_PER_BLOCK / DTYPE_BYTES_BOOL;
        sumCount += CountMaskNonZero(maskUb, curDataLen, alignedDataLen);
    }

    return sumCount;
}

template <typename DTYPE>
__aicore__ inline void MaskedScatter<DTYPE>::ScatterOne(AscendC::LocalTensor<DTYPE>& yUb,
                                                        AscendC::LocalTensor<int8_t>& maskUb,
                                                        AscendC::LocalTensor<DTYPE>& updatesUb, int64_t outputOffset,
                                                        int64_t updatesBackOffset, int64_t numRemainUpdates,
                                                        int64_t& updatesOffsetUb)
{
    if (maskUb.GetValue(outputOffset) != 0 && updatesOffsetUb < numRemainUpdates) {
        yUb.SetValue(outputOffset, updatesUb.GetValue(updatesOffsetUb + updatesBackOffset));
        updatesOffsetUb++;
    }
}

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::ScatterUpdates(AscendC::LocalTensor<DTYPE>& yUb,
                                                               AscendC::LocalTensor<int8_t>& maskUb,
                                                               int64_t numElemPerInput, int64_t updatesStart,
                                                               int64_t numRemainUpdates)
{
    if (numRemainUpdates <= 0) {
        AscendC::PipeBarrier<PIPE_MTE2>();
        return 0;
    }

    AscendC::LocalTensor<DTYPE> updatesUb = inQueueUpdates_.AllocTensor<DTYPE>();
    int64_t burstUpdates = CeilDiv(numRemainUpdates, numElemPerBlock_);
    int64_t updatesBackOffset = 0;
    int64_t alignedNumElemUpdates = AlignDiv(numElemUpdates_, numElemPerBlock_);
    if (updatesStart + burstUpdates * numElemPerBlock_ > alignedNumElemUpdates) {
        updatesBackOffset = updatesStart + burstUpdates * numElemPerBlock_ - alignedNumElemUpdates;
    }

    uint32_t updatesBytes = static_cast<uint32_t>(AlignDiv(numRemainUpdates * dtypeBytesSize_, BYTES_PER_BLOCK));
    AscendC::DataCopyExtParams updatesParams{1, updatesBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<DTYPE> updatesPadParams{true, 0, 0, 0};
    AscendC::DataCopyPad(updatesUb, updatesGm_[updatesStart - updatesBackOffset], updatesParams, updatesPadParams);
    inQueueUpdates_.EnQue(updatesUb);
    updatesUb = inQueueUpdates_.DeQue<DTYPE>();
    AscendC::PipeBarrier<PIPE_MTE2>();

    constexpr int64_t UNROLL_NUM = 8;
    int64_t updatesOffsetUb = 0;
    int64_t offset = 0;
    for (; offset + UNROLL_NUM <= numElemPerInput && updatesOffsetUb < numRemainUpdates; offset += UNROLL_NUM) {
        ScatterOne(yUb, maskUb, updatesUb, offset, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
        ScatterOne(yUb, maskUb, updatesUb, offset + 1, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
        ScatterOne(yUb, maskUb, updatesUb, offset + 2, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
        ScatterOne(yUb, maskUb, updatesUb, offset + 3, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
        ScatterOne(yUb, maskUb, updatesUb, offset + 4, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
        ScatterOne(yUb, maskUb, updatesUb, offset + 5, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
        ScatterOne(yUb, maskUb, updatesUb, offset + 6, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
        ScatterOne(yUb, maskUb, updatesUb, offset + 7, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
    }
    for (; offset < numElemPerInput; offset++) {
        ScatterOne(yUb, maskUb, updatesUb, offset, updatesBackOffset, numRemainUpdates, updatesOffsetUb);
    }
    inQueueUpdates_.FreeTensor(updatesUb);
    return updatesOffsetUb;
}

template <typename DTYPE>
__aicore__ inline int64_t MaskedScatter<DTYPE>::ComputeUnrolled(int64_t inputOffset, int64_t updatesStart)
{
    int64_t numElemPerInput = alignedElemPerCore_;
    int64_t burstMask = alignedElemPerCore_ / BYTES_PER_BLOCK;
    if (inputOffset + alignedElemPerCore_ > numElemX_) {
        numElemPerInput = numElemX_ - inputOffset;
        burstMask = CeilDiv(numElemPerInput, BYTES_PER_BLOCK);
    }
    int64_t burstX = CeilDiv(numElemPerInput, numElemPerBlock_);

    int64_t numRemainUpdates = numElemUpdates_ - updatesStart;
    if (numRemainUpdates > numElemPerInput) {
        numRemainUpdates = numElemPerInput;
    }

    AscendC::LocalTensor<DTYPE> yUb = outQueueY_.AllocTensor<DTYPE>();
    AscendC::DataCopyExtParams xParams{1, static_cast<uint32_t>(burstX * numElemPerBlock_ * dtypeBytesSize_), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DTYPE> xPadParams{true, 0, 0, 0};
    AscendC::DataCopyPad(yUb, xGm_[inputOffset], xParams, xPadParams);
    outQueueY_.EnQue(yUb);
    yUb = outQueueY_.DeQue<DTYPE>();

    AscendC::LocalTensor<int8_t> maskUb = inQueueMask_.AllocTensor<int8_t>();
    AscendC::DataCopyExtParams maskParams{1, static_cast<uint32_t>(burstMask * BYTES_PER_BLOCK), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int8_t> maskPadParams{false, 0, 0, false};
    AscendC::DataCopyPad(maskUb, maskGm_[inputOffset], maskParams, maskPadParams);
    inQueueMask_.EnQue(maskUb);
    maskUb = inQueueMask_.DeQue<int8_t>();

    updatesStart += ScatterUpdates(yUb, maskUb, numElemPerInput, updatesStart, numRemainUpdates);

    AscendC::PipeBarrier<PIPE_ALL>();

    outQueueY_.EnQue(yUb);
    yUb = outQueueY_.DeQue<DTYPE>();
    AscendC::DataCopyExtParams yParams{1, static_cast<uint32_t>(burstX * numElemPerBlock_ * dtypeBytesSize_), 0, 0, 0};
    AscendC::DataCopyPad(yGm_[inputOffset], yUb, yParams);
    event_t eventMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
    outQueueY_.FreeTensor(yUb);
    inQueueMask_.FreeTensor(maskUb);

    return updatesStart;
}

template <>
__aicore__ inline int64_t MaskedScatter<uint8_t>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <>
__aicore__ inline int64_t MaskedScatter<int32_t>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <>
__aicore__ inline int64_t MaskedScatter<int16_t>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <>
__aicore__ inline int64_t MaskedScatter<int8_t>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <>
__aicore__ inline int64_t MaskedScatter<float>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <>
__aicore__ inline int64_t MaskedScatter<half>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <>
__aicore__ inline int64_t MaskedScatter<bool>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <>
__aicore__ inline int64_t MaskedScatter<bfloat16_t>::Compute(int64_t inputOffset, int64_t updatesStart)
{
    return ComputeUnrolled(inputOffset, updatesStart);
}

template <typename DTYPE>
__aicore__ inline void MaskedScatter<DTYPE>::Process()
{
    int64_t aicoreIdx = GetBlockIdx();

    int64_t coreTaskNum = logiTaskNumPerAicore_;
    if (aicoreIdx < logiTaskTail_) {
        coreTaskNum++;
    }

    int64_t preCoreTaskNum = (logiTaskNumPerAicore_ + 1) * aicoreIdx;
    if (aicoreIdx > logiTaskTail_) {
        preCoreTaskNum = (logiTaskNumPerAicore_ + 1) * logiTaskTail_ +
                         (aicoreIdx - logiTaskTail_) * logiTaskNumPerAicore_;
    }

    int64_t taskUpdatesStart = 0;
    if (coreTaskNum > 0) {
        taskUpdatesStart = CalcUpdatesStart(preCoreTaskNum * alignedElemPerCore_);
    }
    for (int64_t taskIdx = 0; taskIdx < coreTaskNum; taskIdx++) {
        taskUpdatesStart = Compute((preCoreTaskNum + taskIdx) * alignedElemPerCore_, taskUpdatesStart);
    }
}

} // namespace NsMaskedScatter

#endif // __MASKED_SCATTER_H__
