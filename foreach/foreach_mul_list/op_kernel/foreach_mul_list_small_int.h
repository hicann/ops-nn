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
 * \file foreach_mul_list_small_int.h
 * \brief
 */
#ifndef FOREACH_MUL_LIST_SMALL_INT_H
#define FOREACH_MUL_LIST_SMALL_INT_H

#include "kernel_operator.h"

namespace ForeachMulList {
using namespace AscendC;

constexpr int32_t SMALL_INT_BUFFER_NUM = 2;

template <typename T>
class ForeachMulListSmallInt {
public:
    __aicore__ inline ForeachMulListSmallInt(){};
    __aicore__ inline void Init(GM_ADDR inputs1, GM_ADDR inputs2, GM_ADDR outputs, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void ComputeSmallInt(const LocalTensor<T>& inputLocal1, const LocalTensor<T>& inputLocal2,
                                           const LocalTensor<T>& outputLocal, int64_t dataCount);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, SMALL_INT_BUFFER_NUM> inputQueue1;
    TQue<QuePosition::VECIN, SMALL_INT_BUFFER_NUM> inputQueue2;
    TQue<QuePosition::VECOUT, SMALL_INT_BUFFER_NUM> outputQueue;

    GlobalTensor<T> inputGm1;
    GlobalTensor<T> inputGm2;
    GlobalTensor<T> outputGm;

    GM_ADDR inputPtr1 = nullptr;
    GM_ADDR inputPtr2 = nullptr;
    GM_ADDR outputPtr = nullptr;

    int64_t blockIdx = 0;
    uint32_t maxDataCount = 0;

    uint64_t inputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;
};

template <typename T>
__aicore__ inline void ForeachMulListSmallInt<T>::Init(GM_ADDR inputs1, GM_ADDR inputs2, GM_ADDR outputs,
                                                       GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    inputPtr1 = inputs1;
    inputPtr2 = inputs2;
    outputPtr = outputs;
    ParseTilingData(tilingData);

    pipe.InitBuffer(inputQueue1, SMALL_INT_BUFFER_NUM, inputsTensorUbSize);
    pipe.InitBuffer(inputQueue2, SMALL_INT_BUFFER_NUM, inputsTensorUbSize);
    pipe.InitBuffer(outputQueue, SMALL_INT_BUFFER_NUM, inputsTensorUbSize);

    maxDataCount = inputsTensorUbSize / sizeof(T);
}

template <typename T>
__aicore__ inline void ForeachMulListSmallInt<T>::Process()
{
    for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = tensorDataCountList[i] - 1;
        if (i == tensorStart) {
            cursorStart = tensorStartOffset;
        }
        if (i == tensorEnd) {
            cursorEnd = tensorEndOffset;
        }

        int64_t dataCount = cursorEnd - cursorStart + 1;
        inputGm1.SetGlobalBuffer(GetTensorAddr(i, inputPtr1) + cursorStart);
        inputGm2.SetGlobalBuffer(GetTensorAddr(i, inputPtr2) + cursorStart);
        outputGm.SetGlobalBuffer(GetTensorAddr(i, outputPtr) + cursorStart);
        SingleTensorProcess(dataCount);
    }
}

template <typename T>
__aicore__ inline void ForeachMulListSmallInt<T>::ParseTilingData(const ForeachCommonTilingData* tilingData)
{
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachMulListSmallInt<T>::SingleTensorProcess(int64_t dataCount)
{
    uint32_t copyTimes = dataCount / maxDataCount;
    uint32_t copyTimesRemainder = dataCount % maxDataCount;
    uint32_t tempDataCount = maxDataCount;

    if (copyTimesRemainder > 0) {
        copyTimes++;
    }

    for (uint32_t i = 0; i < copyTimes; i++) {
        bool isRemainder = false;
        if (i == copyTimes - 1 && copyTimesRemainder > 0) {
            isRemainder = true;
            tempDataCount = copyTimesRemainder;
        }
        CopyIn(i, tempDataCount, isRemainder);
        Compute(i, tempDataCount, isRemainder);
    }
}

template <typename T>
__aicore__ inline void ForeachMulListSmallInt<T>::CopyIn(uint32_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> inputLocal1 = inputQueue1.template AllocTensor<T>();
    LocalTensor<T> inputLocal2 = inputQueue2.template AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal1, inputGm1[1ULL * index * maxDataCount], copyParams, padParams);
        DataCopyPad(inputLocal2, inputGm2[1ULL * index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(inputLocal1, inputGm1[1ULL * index * maxDataCount], dataCount);
        DataCopy(inputLocal2, inputGm2[1ULL * index * maxDataCount], dataCount);
    }
    inputQueue1.EnQue(inputLocal1);
    inputQueue2.EnQue(inputLocal2);
}

template <typename T>
__aicore__ inline void ForeachMulListSmallInt<T>::Compute(uint32_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> inputLocal1 = inputQueue1.template DeQue<T>();
    LocalTensor<T> inputLocal2 = inputQueue2.template DeQue<T>();
    LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();

    ComputeSmallInt(inputLocal1, inputLocal2, outputLocal, dataCount);

    inputQueue1.FreeTensor(inputLocal1);
    inputQueue2.FreeTensor(inputLocal2);
    outputQueue.template EnQue<T>(outputLocal);

    LocalTensor<T> retLocal = outputQueue.template DeQue<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm[1ULL * index * maxDataCount], retLocal, copyParams);
    } else {
        DataCopy(outputGm[1ULL * index * maxDataCount], retLocal, dataCount);
    }
    outputQueue.FreeTensor(retLocal);
}

template <typename T>
__aicore__ inline void ForeachMulListSmallInt<T>::ComputeSmallInt(const LocalTensor<T>& inputLocal1,
                                                                  const LocalTensor<T>& inputLocal2,
                                                                  const LocalTensor<T>& outputLocal, int64_t dataCount)
{
    if constexpr (IsSameType<T, int16_t>::value) {
        Mul(outputLocal, inputLocal1, inputLocal2, dataCount);
    } else {
        uint32_t packedDataCount = (static_cast<uint32_t>(dataCount) + 1U) / 2U;
        LocalTensor<int16_t> inputPacked1 = inputLocal1.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> inputPacked2 = inputLocal2.template ReinterpretCast<int16_t>();
        LocalTensor<int16_t> outputPacked = outputLocal.template ReinterpretCast<int16_t>();

        // The low byte of a packed int16 product is already the product of the two low bytes.
        Mul(outputPacked, inputPacked1, inputPacked2, packedDataCount);
        PipeBarrier<PIPE_V>();
        ShiftRight(inputPacked1, inputPacked1, static_cast<int16_t>(8), packedDataCount);
        PipeBarrier<PIPE_V>();
        ShiftRight(inputPacked2, inputPacked2, static_cast<int16_t>(8), packedDataCount);
        PipeBarrier<PIPE_V>();
        Mul(inputPacked1, inputPacked1, inputPacked2, packedDataCount);
        PipeBarrier<PIPE_V>();
        ShiftLeft(inputPacked1, inputPacked1, static_cast<int16_t>(8), packedDataCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint16_t> outputUnsigned = outputPacked.template ReinterpretCast<uint16_t>();
        ShiftLeft(outputUnsigned, outputUnsigned, static_cast<uint16_t>(8), packedDataCount);
        PipeBarrier<PIPE_V>();
        ShiftRight(outputUnsigned, outputUnsigned, static_cast<uint16_t>(8), packedDataCount);
        PipeBarrier<PIPE_V>();
        Or(outputPacked, outputPacked, inputPacked1, packedDataCount);
    }
}

template <typename T>
__aicore__ inline __gm__ T* ForeachMulListSmallInt<T>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtrAddr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtrAddr + index));
}

constexpr uint32_t SMALL_INT_TINY_TENSOR_UB_SIZE = 64;

template <typename T>
class ForeachMulListTinyScalar {
public:
    __aicore__ inline ForeachMulListTinyScalar(){};
    __aicore__ inline void Init(GM_ADDR inputs1, GM_ADDR inputs2, GM_ADDR outputs, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

private:
    GlobalTensor<T> inputGm1;
    GlobalTensor<T> inputGm2;
    GlobalTensor<T> outputGm;

    GM_ADDR inputPtr1 = nullptr;
    GM_ADDR inputPtr2 = nullptr;
    GM_ADDR outputPtr = nullptr;
    event_t eventMte2ToS = static_cast<event_t>(0);
    event_t eventSToMte3 = static_cast<event_t>(0);

    uint16_t tensorIndex = 0;
    int64_t dataCount = 0;
};

template <typename T>
__aicore__ inline void ForeachMulListTinyScalar<T>::Init(GM_ADDR inputs1, GM_ADDR inputs2, GM_ADDR outputs,
                                                         GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
{
    uint32_t blockIdx = GetBlockIdx();
    inputPtr1 = inputs1;
    inputPtr2 = inputs2;
    outputPtr = outputs;
    tensorIndex = static_cast<uint16_t>(blockIdx);
    dataCount = tilingData->tensorDataCountList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachMulListTinyScalar<T>::Process()
{
    inputGm1.SetGlobalBuffer(GetTensorAddr(tensorIndex, inputPtr1));
    inputGm2.SetGlobalBuffer(GetTensorAddr(tensorIndex, inputPtr2));
    outputGm.SetGlobalBuffer(GetTensorAddr(tensorIndex, outputPtr));
    Compute(dataCount);
}

template <typename T>
__aicore__ inline void ForeachMulListTinyScalar<T>::Compute(int64_t dataCount)
{
    constexpr uint32_t tensorUbElements = SMALL_INT_TINY_TENSOR_UB_SIZE / sizeof(T);
    LocalTensor<T> inputLocal1(TPosition::VECCALC, 0, tensorUbElements);
    LocalTensor<T> inputLocal2(TPosition::VECCALC, SMALL_INT_TINY_TENSOR_UB_SIZE, tensorUbElements);
    LocalTensor<T> outputLocal(TPosition::VECCALC, SMALL_INT_TINY_TENSOR_UB_SIZE * 2, tensorUbElements);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal1, inputGm1, copyParams, padParams);
    DataCopyPad(inputLocal2, inputGm2, copyParams, padParams);
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS);

    uint32_t groupCount = (static_cast<uint32_t>(dataCount) + sizeof(uint32_t) - 1U) / sizeof(uint32_t);
    LocalTensor<uint32_t> inputPacked1 = inputLocal1.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> inputPacked2 = inputLocal2.template ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> outputPacked = outputLocal.template ReinterpretCast<uint32_t>();
    for (uint32_t i = 0; i < groupCount; ++i) {
        uint32_t input1 = inputPacked1.GetValue(i);
        uint32_t input2 = inputPacked2.GetValue(i);
        uint32_t product0 = (input1 & 0xFFU) * (input2 & 0xFFU);
        uint32_t product1 = ((input1 >> 8) & 0xFFU) * ((input2 >> 8) & 0xFFU);
        uint32_t product2 = ((input1 >> 16) & 0xFFU) * ((input2 >> 16) & 0xFFU);
        uint32_t product3 = ((input1 >> 24) & 0xFFU) * ((input2 >> 24) & 0xFFU);
        uint32_t result = (product0 & 0xFFU) | ((product1 & 0xFFU) << 8) | ((product2 & 0xFFU) << 16) |
                          ((product3 & 0xFFU) << 24);
        outputPacked.SetValue(i, result);
    }

    SetFlag<HardEvent::S_MTE3>(eventSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
    DataCopyPad(outputGm, outputLocal, copyParams);
}

template <typename T>
__aicore__ inline __gm__ T* ForeachMulListTinyScalar<T>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtrAddr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtrAddr + index));
}

} // namespace ForeachMulList

#endif // FOREACH_MUL_LIST_SMALL_INT_H
