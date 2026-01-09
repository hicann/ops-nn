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
 * \file foreach_pow_list.h
 * \brief
 */

#ifndef FOREACH_POW_LIST_H
#define FOREACH_POW_LIST_H

#include <type_traits>
#include "kernel_operator.h"

namespace ForeachPow {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t POW_TENSOR_TENSOR_BF16_BUFFER_NUM = 3;
constexpr int32_t POW_TENSOR_TENSOR_HALF_BUFFER_NUM = 12;
constexpr int32_t POW_TENSOR_TENSOR_FLOAT_BUFFER_NUM = 3;
constexpr int32_t POW_TENSOR_TENSOR_INT32_BUFFER_NUM = 5;

constexpr uint8_t COPY_SPACE_MULTIPLE = 9;

constexpr uint8_t INPUT_PARAMETER_COUNT = 3;
template <typename T>
class InnerComputer
{
public:
    __aicore__ inline void Compute(
        LocalTensor<T>& inLocal_1, LocalTensor<T>& inLocal_2, LocalTensor<T>& outLocal,
        LocalTensor<float>& float32Tensor, LocalTensor<uint8_t>& bufferTensor, uint32_t maxCastDataCount,
        int64_t dataCount)
    {
        Power<T, false>(outLocal, inLocal_1, inLocal_2, bufferTensor, dataCount);
    }
};

template <>
class InnerComputer<bfloat16_t>
{
public:
    __aicore__ inline void Compute(
        LocalTensor<bfloat16_t>& inLocal_1, LocalTensor<bfloat16_t>& inLocal_2, LocalTensor<bfloat16_t>& outLocal,
        LocalTensor<float>& float32Tensor, LocalTensor<uint8_t>& bufferTensor, uint32_t maxCastDataCount,
        int64_t dataCount)
    {
        uint32_t castTimes = 0;
        uint32_t castTimesRemainder = 0;
        if (maxCastDataCount == 0) {
            castTimes = -1;
            castTimesRemainder = -1;
        } else {
            castTimes = dataCount / maxCastDataCount;
            castTimesRemainder = dataCount % maxCastDataCount;
        }

        for (uint32_t i = 0; i < castTimes; i++) {
            ComputerPerCast(
                inLocal_1, inLocal_2, outLocal, float32Tensor, bufferTensor, maxCastDataCount, i, maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputerPerCast(
                inLocal_1, inLocal_2, outLocal, float32Tensor, bufferTensor, maxCastDataCount, castTimes,
                castTimesRemainder);
        }
    }

private:
    __aicore__ inline void ComputerPerCast(
        LocalTensor<bfloat16_t>& inLocal_1, LocalTensor<bfloat16_t>& inLocal_2, LocalTensor<bfloat16_t>& outLocal,
        LocalTensor<float>& float32Tensor, LocalTensor<uint8_t>& bufferTensor, uint32_t maxCastDataCount,
        uint16_t index, int64_t dataCount)
    {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Power<float, false>(
            float32Tensor[BUFFER_NUM * maxCastDataCount], float32Tensor, float32Tensor[maxCastDataCount], bufferTensor,
            dataCount);
        PipeBarrier<PIPE_V>();
        Cast(
            outLocal[index * maxCastDataCount], float32Tensor[BUFFER_NUM * maxCastDataCount], RoundMode::CAST_RINT,
            dataCount);
        PipeBarrier<PIPE_V>();
    }
};

template <typename T>
class ForeachPowList
{
public:
    __aicore__ inline ForeachPowList(){};
    __aicore__ inline void Init(
        GM_ADDR self, GM_ADDR exponent, GM_ADDR outputs, GM_ADDR workspace, const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>

    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };

    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(
        int64_t dataCount, LocalTensor<float>& float32Tensor, LocalTensor<uint8_t>& bufferTensor);
    __aicore__ inline void CopyIn(uint32_t offset, int64_t dataCount, bool isRemainder);
    __aicore__ inline void Compute(
        uint32_t offset, int64_t dataCount, LocalTensor<float>& float32Tensor, LocalTensor<uint8_t>& bufferTensor);
    __aicore__ inline void CopyOut(uint32_t offset, int64_t dataCount, bool isRemainder);
    __aicore__ inline __gm__ T* GetTensorAddr(GM_ADDR TensorsPtr, uint16_t index, int64_t offset);
    __aicore__ inline __gm__ uint64_t GetTmpBufferUBSize();

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_1;
    TQue<QuePosition::VECIN, BUFFER_NUM> InQueue_2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> OutQueue;
    TBuf<QuePosition::VECCALC> tmpBuffer;

    GlobalTensor<T> inTensorsGM_1;
    GlobalTensor<T> inTensorsGM_2;
    GlobalTensor<T> outTensorsGM;

    GM_ADDR inTensorsPtr_1 = nullptr;
    GM_ADDR inTensorsPtr_2 = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    LocalTensor<uint8_t> BufferTensorUB;
    int64_t blockIdx = 0;

    uint32_t maxDataCount = {0};
    // tiling params
    uint64_t inputsTensorUbSize = 0;

    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = {0};
    uint16_t tensorEnd = {0};
    int64_t tensorStartOffset = {0};
    int64_t tensorEndOffset = {0};

    TQue<QuePosition::VECIN, 1> float32Queue;
    TQue<QuePosition::VECIN, 1> bufferQueue;
    uint32_t maxCastDataCount = {0};
    uint64_t reserved4CastingUbSize = 0;
};

template <typename T>
__aicore__ inline void ForeachPowList<T>::Init(
    GM_ADDR self, GM_ADDR exponent, GM_ADDR outputs, GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
{
    blockIdx = GetBlockIdx();

    inTensorsPtr_1 = self;
    inTensorsPtr_2 = exponent;
    outTensorsPtr = outputs;

    ParseTilingData(tilingData);
    if (std::is_same<T, bfloat16_t>::value) {
        uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        pipe.InitBuffer(InQueue_1, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(InQueue_2, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(OutQueue, BUFFER_NUM, totalTensorUbSize);
        maxDataCount = totalTensorUbSize / sizeof(T);
        pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT);
        LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
        float32Queue.EnQue(float32Tensor);
        pipe.InitBuffer(bufferQueue, 1, inputsTensorUbSize * GetTmpBufferUBSize());
        LocalTensor<uint8_t> bufferTensor = bufferQueue.AllocTensor<uint8_t>();
        bufferQueue.EnQue(bufferTensor);
        maxCastDataCount = inputsTensorUbSize / sizeof(float);
    } else {
        pipe.InitBuffer(InQueue_1, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(InQueue_2, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(OutQueue, BUFFER_NUM, inputsTensorUbSize);
        maxDataCount = inputsTensorUbSize / sizeof(T);
        pipe.InitBuffer(bufferQueue, 1, inputsTensorUbSize * GetTmpBufferUBSize());
        LocalTensor<uint8_t> bufferTensor = bufferQueue.AllocTensor<uint8_t>();
        bufferQueue.EnQue(bufferTensor);
    }
}

template <typename T>
__aicore__ inline void ForeachPowList<T>::Process()
{
    LocalTensor<uint8_t> bufferTensor = bufferQueue.DeQue<uint8_t>();
    LocalTensor<float> float32Tensor;
    if (std::is_same<T, bfloat16_t>::value) {
        float32Tensor = float32Queue.DeQue<float>();
    }
    for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t dataCount = tensorDataCountList[i];
        if (i == tensorStart) {
            cursorStart = tensorStartOffset;
            dataCount = dataCount - cursorStart;
        }
        inTensorsGM_1.SetGlobalBuffer(GetTensorAddr(inTensorsPtr_1, i, cursorStart));
        inTensorsGM_2.SetGlobalBuffer(GetTensorAddr(inTensorsPtr_2, i, cursorStart));
        outTensorsGM.SetGlobalBuffer(GetTensorAddr(outTensorsPtr, i, cursorStart));

        if (i == tensorEnd) {
            dataCount = tensorEndOffset + 1 - cursorStart;
        }
        SingleTensorProcess(dataCount, float32Tensor, bufferTensor);
    }
    bufferQueue.FreeTensor(bufferTensor);
    if (std::is_same<T, bfloat16_t>::value) {
        float32Queue.FreeTensor(float32Tensor);
    }
}

template <typename T>
__aicore__ inline void ForeachPowList<T>::SingleTensorProcess(
    int64_t dataCount, LocalTensor<float>& float32Tensor, LocalTensor<uint8_t>& bufferTensor)
{
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / maxDataCount;
    uint32_t copyTimesRemainder = dataCount % maxDataCount;

    for (uint32_t i = 0; i < copyTimes; i++) {
        uint32_t offset = i * maxDataCount;
        CopyIn(offset, maxDataCount, false);
        Compute(offset, maxDataCount, float32Tensor, bufferTensor);
        CopyOut(offset, maxDataCount, false);
    }

    if (copyTimesRemainder > 0) {
        CopyIn(copyTimes * maxDataCount, copyTimesRemainder, true);
        Compute(copyTimes * maxDataCount, copyTimesRemainder, float32Tensor, bufferTensor);
        CopyOut(copyTimes * maxDataCount, copyTimesRemainder, true);
    }
}

template <typename T>
__aicore__ inline void ForeachPowList<T>::ParseTilingData(const ForeachCommonTilingData* tilingData)
{
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachPowList<T>::CopyIn(uint32_t offset, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> inLocal_1 = InQueue_1.AllocTensor<T>();
    LocalTensor<T> inLocal_2 = InQueue_2.AllocTensor<T>();

    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inLocal_1, inTensorsGM_1[1ULL * offset], copyParams, padParams);
        DataCopyPad(inLocal_2, inTensorsGM_2[1ULL * offset], copyParams, padParams);
    } else {
        DataCopy(inLocal_1, inTensorsGM_1[1ULL * offset], dataCount);
        DataCopy(inLocal_2, inTensorsGM_2[1ULL * offset], dataCount);
    }

    InQueue_1.EnQue(inLocal_1);
    InQueue_2.EnQue(inLocal_2);
}

template <typename T>
__aicore__ inline void ForeachPowList<T>::Compute(
    uint32_t offset, int64_t dataCount, LocalTensor<float>& float32Tensor, LocalTensor<uint8_t>& bufferTensor)
{
    LocalTensor<T> inLocal_1 = InQueue_1.DeQue<T>();
    LocalTensor<T> inLocal_2 = InQueue_2.DeQue<T>();
    LocalTensor<T> outLocal = OutQueue.AllocTensor<T>();

    InnerComputer<T> computer;
    computer.Compute(inLocal_1, inLocal_2, outLocal, float32Tensor, bufferTensor, maxCastDataCount, dataCount);

    OutQueue.EnQue(outLocal);
    InQueue_1.FreeTensor(inLocal_1);
    InQueue_2.FreeTensor(inLocal_2);
}

template <typename T>
__aicore__ inline void ForeachPowList<T>::CopyOut(uint32_t offset, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> outLocal = OutQueue.DeQue<T>();

    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorsGM[1ULL * offset], outLocal, copyParams);
    } else {
        DataCopy(outTensorsGM[1ULL * offset], outLocal, dataCount);
    }

    OutQueue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline __gm__ T* ForeachPowList<T>::GetTensorAddr(GM_ADDR TensorsPtr, uint16_t index, int64_t offset)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(TensorsPtr);
    // The offset of the data address from the first address.
    uint64_t tensorPtrOffset = *dataAddr;
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return (reinterpret_cast<__gm__ T*>(*(tensorPtr + index))) + offset;
}
template <typename T>
__aicore__ inline __gm__ uint64_t ForeachPowList<T>::GetTmpBufferUBSize()
{
    if (std::is_same<T, float>::value) {
        return POW_TENSOR_TENSOR_FLOAT_BUFFER_NUM;
    } else if (std::is_same<T, bfloat16_t>::value) {
        return POW_TENSOR_TENSOR_BF16_BUFFER_NUM;
    } else if (std::is_same<T, half>::value) {
        return POW_TENSOR_TENSOR_HALF_BUFFER_NUM;
    } else if (std::is_same<T, int>::value) {
        return POW_TENSOR_TENSOR_INT32_BUFFER_NUM;
    }
    return POW_TENSOR_TENSOR_FLOAT_BUFFER_NUM;
}

} // namespace ForeachPow

#endif // FOREACH_POW_LIST
