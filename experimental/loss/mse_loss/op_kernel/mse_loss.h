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
 * \file mse_loss.h
 * \brief MseLoss 算子 kernel 类定义
 */

#ifndef MSELOSS_H
#define MSELOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mse_loss_tiling_data.h"
#include "mse_loss_tiling_key.h"
#include <type_traits>

namespace NsMseLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t VECTOR_REG_BYTES = 256;
constexpr int64_t UB_BLOCK_BYTES = static_cast<int64_t>(GetDataBlockSizeInBytes());
constexpr int64_t FLOAT_256B_ALIGN_ELEM = VECTOR_REG_BYTES / static_cast<int64_t>(sizeof(float));

template <typename T>
class MseLoss {
    using IO_T = T;
    static constexpr bool NEED_UPCAST = !std::is_same<IO_T, float>::value;

public:
    __aicore__ inline MseLoss(){};

    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace,
                                const MseLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void ComputeNone(int64_t currentNum);
    __aicore__ inline void Accumulate(int64_t currentNum, LocalTensor<float>& partialLocal);
    __aicore__ inline void WriteReduceResult(LocalTensor<float>& resultLocal);
    __aicore__ inline void CopyOutReduce(LocalTensor<float>& partialLocal);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> predictQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> labelQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TQue<QuePosition::VECIN, 1> downloadQueue;
    TBuf<QuePosition::VECCALC> predictFloatBuf;
    TBuf<QuePosition::VECCALC> labelFloatBuf;
    TBuf<QuePosition::VECCALC> tmpFloatBuf;
    TBuf<QuePosition::VECCALC> partialFloatBuf;

    GlobalTensor<IO_T> predictGM;
    GlobalTensor<IO_T> labelGM;
    GlobalTensor<IO_T> outputGM;
    GlobalTensor<float> workspaceGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t reduction_ = 2;
    int64_t blockNum_ = 1;
    int64_t workspaceFloatsPerCore_ = 1;
    float meanScale_ = 1.0f;
};

template <typename T>
__aicore__ inline void MseLoss<T>::Init(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace,
                                        const MseLossTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = (tilingData->ubFactor < FLOAT_256B_ALIGN_ELEM) ? FLOAT_256B_ALIGN_ELEM : tilingData->ubFactor;
    reduction_ = tilingData->reduction;
    blockNum_ = tilingData->blockNum;
    workspaceFloatsPerCore_ = tilingData->workspaceFloatsPerCore;
    meanScale_ = tilingData->meanScale;

    int64_t offset = tilingData->blockFactor * GetBlockIdx();
    predictGM.SetGlobalBuffer((__gm__ IO_T*)predict + offset, blockLength_);
    labelGM.SetGlobalBuffer((__gm__ IO_T*)label + offset, blockLength_);
    if (reduction_ == 0) {
        outputGM.SetGlobalBuffer((__gm__ IO_T*)y + offset, blockLength_);
    } else {
        outputGM.SetGlobalBuffer((__gm__ IO_T*)y, 1);
        workspaceGM.SetGlobalBuffer((__gm__ float*)workspace, blockNum_ * workspaceFloatsPerCore_);
    }

    pipe.InitBuffer(predictQueue, BUFFER_NUM, ubLength_ * sizeof(IO_T));
    pipe.InitBuffer(labelQueue, BUFFER_NUM, ubLength_ * sizeof(IO_T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(IO_T));
    pipe.InitBuffer(downloadQueue, 1, blockNum_ * workspaceFloatsPerCore_ * sizeof(float));
    pipe.InitBuffer(tmpFloatBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(partialFloatBuf, workspaceFloatsPerCore_ * sizeof(float));
    if constexpr (NEED_UPCAST) {
        pipe.InitBuffer(predictFloatBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(labelFloatBuf, ubLength_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void MseLoss<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<IO_T> predictLocal = predictQueue.template AllocTensor<IO_T>();
    LocalTensor<IO_T> labelLocal = labelQueue.template AllocTensor<IO_T>();
    constexpr int64_t ioAlignElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(IO_T));
    int64_t copyAlignedNum = (currentNum + ioAlignElem - 1) / ioAlignElem * ioAlignElem;
    uint8_t rightPadding = static_cast<uint8_t>(copyAlignedNum - currentNum);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(IO_T)), 0, 0, 0};
    DataCopyPadExtParams<IO_T> padParams{true, 0, rightPadding, static_cast<IO_T>(0)};
    DataCopyPad(predictLocal, predictGM[progress * ubLength_], copyParams, padParams);
    DataCopyPad(labelLocal, labelGM[progress * ubLength_], copyParams, padParams);
    predictQueue.EnQue(predictLocal);
    labelQueue.EnQue(labelLocal);
}

template <typename T>
__aicore__ inline void MseLoss<T>::ComputeNone(int64_t currentNum)
{
    LocalTensor<IO_T> predictLocal = predictQueue.template DeQue<IO_T>();
    LocalTensor<IO_T> labelLocal = labelQueue.template DeQue<IO_T>();
    LocalTensor<IO_T> outputLocal = outputQueue.template AllocTensor<IO_T>();

    if constexpr (NEED_UPCAST) {
        LocalTensor<float> predictFloat = predictFloatBuf.Get<float>();
        LocalTensor<float> labelFloat = labelFloatBuf.Get<float>();
        Cast(predictFloat, predictLocal, RoundMode::CAST_NONE, currentNum);
        Cast(labelFloat, labelLocal, RoundMode::CAST_NONE, currentNum);
        PipeBarrier<PIPE_V>();
        Sub(predictFloat, predictFloat, labelFloat, currentNum);
        PipeBarrier<PIPE_V>();
        Mul(labelFloat, predictFloat, predictFloat, currentNum);
        PipeBarrier<PIPE_V>();
        Cast(outputLocal, labelFloat, RoundMode::CAST_RINT, currentNum);
    } else {
        LocalTensor<float> tmpFloat = tmpFloatBuf.Get<float>();
        Sub(tmpFloat, predictLocal, labelLocal, currentNum);
        PipeBarrier<PIPE_V>();
        Mul(outputLocal, tmpFloat, tmpFloat, currentNum);
    }

    outputQueue.template EnQue<IO_T>(outputLocal);
    predictQueue.FreeTensor(predictLocal);
    labelQueue.FreeTensor(labelLocal);
}

template <typename T>
__aicore__ inline void MseLoss<T>::Accumulate(int64_t currentNum, LocalTensor<float>& partialLocal)
{
    LocalTensor<IO_T> predictLocal = predictQueue.template DeQue<IO_T>();
    LocalTensor<IO_T> labelLocal = labelQueue.template DeQue<IO_T>();
    LocalTensor<float> tmpFloat = tmpFloatBuf.Get<float>();

    if constexpr (NEED_UPCAST) {
        LocalTensor<float> predictFloat = predictFloatBuf.Get<float>();
        LocalTensor<float> labelFloat = labelFloatBuf.Get<float>();
        Cast(predictFloat, predictLocal, RoundMode::CAST_NONE, currentNum);
        Cast(labelFloat, labelLocal, RoundMode::CAST_NONE, currentNum);
        PipeBarrier<PIPE_V>();
        Sub(tmpFloat, predictFloat, labelFloat, currentNum);
    } else {
        Sub(tmpFloat, predictLocal, labelLocal, currentNum);
    }
    PipeBarrier<PIPE_V>();
    Mul(tmpFloat, tmpFloat, tmpFloat, currentNum);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpFloat, tmpFloat, tmpFloat, currentNum);
    PipeBarrier<PIPE_V>();
    Add(partialLocal, partialLocal, tmpFloat, 1);

    predictQueue.FreeTensor(predictLocal);
    labelQueue.FreeTensor(labelLocal);
}

template <typename T>
__aicore__ inline void MseLoss<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<IO_T> outputLocal = outputQueue.template DeQue<IO_T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(IO_T)), 0, 0, 0};
    DataCopyPad(outputGM[progress * ubLength_], outputLocal, copyParams);
    outputQueue.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void MseLoss<T>::CopyOutReduce(LocalTensor<float>& partialLocal)
{
    if (blockNum_ == 1) {
        if (reduction_ == 2) {
            Muls(partialLocal, partialLocal, meanScale_, 1);
            PipeBarrier<PIPE_V>();
        }
        WriteReduceResult(partialLocal);
        return;
    }

    PipeBarrier<PIPE_ALL>();
    DataCopy(workspaceGM[GetBlockIdx() * workspaceFloatsPerCore_], partialLocal, workspaceFloatsPerCore_);
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    if (GetBlockIdx() != 0) {
        return;
    }

    LocalTensor<float> mergeLocal = downloadQueue.template AllocTensor<float>();
    DataCopy(mergeLocal, workspaceGM, blockNum_ * workspaceFloatsPerCore_);
    downloadQueue.template EnQue<float>(mergeLocal);
    mergeLocal = downloadQueue.template DeQue<float>();
    ReduceSum<float>(mergeLocal, mergeLocal, mergeLocal, blockNum_ * workspaceFloatsPerCore_);
    PipeBarrier<PIPE_V>();
    if (reduction_ == 2) {
        Muls(mergeLocal, mergeLocal, meanScale_, 1);
        PipeBarrier<PIPE_V>();
    }

    WriteReduceResult(mergeLocal);
    downloadQueue.FreeTensor(mergeLocal);
}

template <typename T>
__aicore__ inline void MseLoss<T>::WriteReduceResult(LocalTensor<float>& resultLocal)
{
    if constexpr (NEED_UPCAST) {
        LocalTensor<IO_T> outLocal = outputQueue.template AllocTensor<IO_T>();
        Cast(outLocal, resultLocal, RoundMode::CAST_RINT, 1);
        outputQueue.template EnQue<IO_T>(outLocal);
    } else {
        LocalTensor<IO_T> outLocal = outputQueue.template AllocTensor<IO_T>();
        Adds(outLocal, resultLocal, 0.0f, 1);
        outputQueue.template EnQue<IO_T>(outLocal);
    }
    LocalTensor<IO_T> outLocal = outputQueue.template DeQue<IO_T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(IO_T)), 0, 0, 0};
    DataCopyPad(outputGM, outLocal, copyParams);
    outputQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void MseLoss<T>::Process()
{
    if (reduction_ == 0) {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
            CopyIn(i, currentNum);
            ComputeNone(currentNum);
            CopyOut(i, currentNum);
        }
        return;
    }

    LocalTensor<float> partialLocal = partialFloatBuf.Get<float>();
    Duplicate(partialLocal, 0.0f, workspaceFloatsPerCore_);
    if (blockLength_ > 0) {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - i * ubLength_) : ubLength_;
            CopyIn(i, currentNum);
            Accumulate(currentNum, partialLocal);
        }
    }
    CopyOutReduce(partialLocal);
}

} // namespace NsMseLoss
#endif // MSELOSS_H
