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
 * \file foreach_pow_scalar_list.h
 * \brief
 */

#ifndef FOREACH_POW_SCALAR_H
#define FOREACH_POW_SCALAR_H

#include <type_traits>
#include "kernel_operator.h"

namespace ForeachPowScalarList {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr uint8_t COPY_SPACE_MULTIPLE = 9;

template <typename T>
class InnerComputer
{
public:
    __aicore__ inline void Compute(
        LocalTensor<T>& dataLocal, LocalTensor<T>& outLocal, TQue<QuePosition::VECIN, BUFFER_NUM>& dataCastQueue,
        TQue<QuePosition::VECIN, BUFFER_NUM>& outCastQueue, T scalarValue, uint32_t maxCastDataCount, int64_t dataCount)
    {
        Power<T, false>(outLocal, dataLocal, scalarValue);
    }
};

template <>
class InnerComputer<bfloat16_t>
{
public:
    __aicore__ inline void Compute(
        LocalTensor<bfloat16_t>& dataLocal, LocalTensor<bfloat16_t>& outLocal,
        TQue<QuePosition::VECIN, BUFFER_NUM>& dataCastQueue, TQue<QuePosition::VECIN, BUFFER_NUM>& outCastQueue,
        float scalarValue, uint32_t maxCastDataCount, int64_t dataCount)
    {
        uint32_t castTimes;
        uint32_t castTimesRemainder;

        if (maxCastDataCount != 0) {
            castTimes = dataCount / maxCastDataCount;
            castTimesRemainder = dataCount % maxCastDataCount;
        }

        for (uint32_t i = 0; i < castTimes; i++) {
            CopyPerCast(dataLocal, dataCastQueue, maxCastDataCount, i, maxCastDataCount);
            ComputerPerCast(outLocal, dataCastQueue, outCastQueue, scalarValue, maxCastDataCount, i, maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            CopyPerCast(dataLocal, dataCastQueue, maxCastDataCount, castTimes, castTimesRemainder);
            ComputerPerCast(
                outLocal, dataCastQueue, outCastQueue, scalarValue, maxCastDataCount, castTimes, castTimesRemainder);
        }
    }

private:
    __aicore__ inline void CopyPerCast(
        LocalTensor<bfloat16_t>& dataLocal, TQue<QuePosition::VECIN, BUFFER_NUM>& dataCastQueue,
        uint32_t maxCastDataCount, uint16_t index, int64_t dataCount)
    {
        LocalTensor<float> dataCastLocal = dataCastQueue.AllocTensor<float>();
        PipeBarrier<PIPE_V>();
        Cast(dataCastLocal, dataLocal[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        dataCastQueue.EnQue(dataCastLocal);
    }

    __aicore__ inline void ComputerPerCast(
        LocalTensor<bfloat16_t>& outLocal, TQue<QuePosition::VECIN, BUFFER_NUM>& dataCastQueue,
        TQue<QuePosition::VECIN, BUFFER_NUM>& outCastQueue, float scalarValue, uint32_t maxCastDataCount,
        uint16_t index, int64_t dataCount)
    {
        LocalTensor<float> dataCastLocal = dataCastQueue.DeQue<float>();
        LocalTensor<float> outCastLocal = outCastQueue.AllocTensor<float>();

        PipeBarrier<PIPE_V>();
        Power<float, false>(outCastLocal, dataCastLocal, scalarValue);
        PipeBarrier<PIPE_V>();

        outCastQueue.EnQue(outCastLocal);
        dataCastQueue.FreeTensor(dataCastLocal);

        LocalTensor<float> tempLocal = outCastQueue.DeQue<float>();
        PipeBarrier<PIPE_V>();
        Cast(outLocal[index * maxCastDataCount], tempLocal, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
        outCastQueue.FreeTensor(tempLocal);
    }
};

template <typename T>
class ForeachPowScalarListND
{
public:
    __aicore__ inline ForeachPowScalarListND(){};
    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void ComputeAndCopyOut(uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline __gm__ T* GetInputTensorAddr(uint16_t index);
    __aicore__ inline __gm__ T* GetOutputTensorAddr(uint16_t index);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> inTensorGM;
    GlobalTensor<T> outTensorGM;
    GlobalTensor<DTYPE_SCALARS> inScalarGM;

    GM_ADDR inTensorPtr = nullptr;
    GM_ADDR outTensorPtr = nullptr;
    int64_t blockIdx = 0;
    using TT = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;
    TT scalarValue = 0;

    uint32_t maxDataCount = {0};
    // tiling params
    uint64_t inputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = {0};
    uint16_t tensorEnd = {0};
    int64_t tensorStartOffset = {0};
    int64_t tensorEndOffset = {0};

    TQue<QuePosition::VECIN, BUFFER_NUM> dataCastQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> outCastQueue;
    uint32_t maxCastDataCount = {0};
    uint64_t reserved4CastingUbSize = 0;
};

template <typename T>
__aicore__ inline void ForeachPowScalarListND<T>::Init(
    GM_ADDR inputs, GM_ADDR scalar, GM_ADDR outputs, GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    inTensorPtr = inputs;
    outTensorPtr = outputs;
    ParseTilingData(tilingData);
    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_SCALARS*)scalar, 1);

    // Init for bfloat16
    if (std::is_same<T, bfloat16_t>::value) {
        uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        pipe.InitBuffer(dataQueue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, totalTensorUbSize);
        maxDataCount = totalTensorUbSize / sizeof(T);
        reserved4CastingUbSize = inputsTensorUbSize;
        pipe.InitBuffer(dataCastQueue, BUFFER_NUM, reserved4CastingUbSize);
        pipe.InitBuffer(outCastQueue, BUFFER_NUM, reserved4CastingUbSize);
        maxCastDataCount = reserved4CastingUbSize / sizeof(float);
    } else {
        pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, inputsTensorUbSize);
        maxDataCount = inputsTensorUbSize / sizeof(T);
    }
}

template <typename T>
__aicore__ inline void ForeachPowScalarListND<T>::Process()
{
    for (uint16_t i = tensorStart; i <= tensorEnd; i++) {
        int64_t cursorStart = 0;
        int64_t cursorEnd = tensorDataCountList[i] - 1;
        int64_t dataCount = 0;
        if (i == tensorStart) {
            cursorStart = tensorStartOffset;
        }
        if (i == tensorEnd) {
            cursorEnd = tensorEndOffset;
        }

        if (std::is_same<T, bfloat16_t>::value) {
            scalarValue = float(inScalarGM.GetValue(i));
        } else {
            scalarValue = T(inScalarGM.GetValue(i));
        }

        dataCount = cursorEnd - cursorStart + 1;
        inTensorGM.SetGlobalBuffer(GetInputTensorAddr(i) + cursorStart);
        outTensorGM.SetGlobalBuffer(GetOutputTensorAddr(i) + cursorStart);
        SingleTensorProcess(dataCount);
    }
}

template <typename T>
__aicore__ inline void ForeachPowScalarListND<T>::SingleTensorProcess(int64_t dataCount)
{
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / maxDataCount;
    uint32_t copyTimesRemainder = dataCount % maxDataCount;

    for (uint32_t i = 0; i < copyTimes; i++) {
        CopyIn(i, maxDataCount, false);
        ComputeAndCopyOut(i, maxDataCount, false);
    }

    if (copyTimesRemainder > 0) {
        CopyIn(copyTimes, copyTimesRemainder, true);
        ComputeAndCopyOut(copyTimes, copyTimesRemainder, true);
    }
}

template <typename T>
__aicore__ inline void ForeachPowScalarListND<T>::ParseTilingData(const ForeachCommonTilingData* tilingData)
{
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachPowScalarListND<T>::CopyIn(uint16_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> dataLocal = dataQueue.AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorGM[1ULL * index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorGM[1ULL * index * maxDataCount], dataCount);
    }
    dataQueue.EnQue(dataLocal);
}

template <typename T>
__aicore__ inline void ForeachPowScalarListND<T>::ComputeAndCopyOut(uint16_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> dataLocal = dataQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    InnerComputer<T> computer;
    computer.Compute(dataLocal, outLocal, dataCastQueue, outCastQueue, scalarValue, maxCastDataCount, dataCount);

    dataQueue.FreeTensor(dataLocal);
    outQueue.EnQue<T>(outLocal);

    LocalTensor<T> retLocal = outQueue.DeQue<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorGM[1ULL * index * maxDataCount], retLocal, copyParams);
    } else {
        DataCopy(outTensorGM[1ULL * index * maxDataCount], retLocal, dataCount);
    }
    outQueue.FreeTensor(retLocal);
}

template <typename T>
__aicore__ inline __gm__ T* ForeachPowScalarListND<T>::GetInputTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(inTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

template <typename T>
__aicore__ inline __gm__ T* ForeachPowScalarListND<T>::GetOutputTensorAddr(uint16_t index)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(outTensorPtr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}

} // namespace ForeachPowScalarList

#endif // FOREACH_POW_SCALAR_H