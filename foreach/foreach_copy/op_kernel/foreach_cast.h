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
 * \file foreach_cast.h
 * \brief
 */
#ifndef FOREACH_CAST_N_D_H
#define FOREACH_CAST_N_D_H

#include <type_traits>
#include "kernel_operator.h"

namespace ForeachCast {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T_in, typename T_out>

class ForeachCastND {
public:
    __aicore__ inline ForeachCastND(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void ComputeAndCopyOut(
        uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline __gm__ T_in* GetInputTensorAddr(uint16_t index, GM_ADDR tensorPtr);
    __aicore__ inline __gm__ T_out* GetOutputTensorAddr(uint16_t index, GM_ADDR tensorPtr);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T_in> inTensorsGM;
    GlobalTensor<T_out> outTensorsGM;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    uint32_t maxDataCount = {0};
    uint64_t inputsTensorUbSize = 0;
    uint64_t outputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = {0};
    uint16_t tensorEnd = {0};
    int64_t tensorStartOffset = {0};
    int64_t tensorEndOffset = {0};
};

template <typename T_in, typename T_out>
__aicore__ inline void ForeachCastND<T_in, T_out>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    inTensorsPtr = x;
    outTensorsPtr = y;
    ParseTilingData(tilingData);
    pipe.InitBuffer(dataQueue, BUFFER_NUM, inputsTensorUbSize);
    pipe.InitBuffer(outQueue, BUFFER_NUM, outputsTensorUbSize);
    maxDataCount = inputsTensorUbSize / sizeof(float);
}

template <typename T_in, typename T_out>
__aicore__ inline void ForeachCastND<T_in, T_out>::Process()
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

        dataCount = cursorEnd - cursorStart + 1;
        inTensorsGM.SetGlobalBuffer(GetInputTensorAddr(i, inTensorsPtr) + cursorStart);
        outTensorsGM.SetGlobalBuffer(GetOutputTensorAddr(i, outTensorsPtr) + cursorStart);
        SingleTensorProcess(dataCount);
    }
}

template <typename T_in, typename T_out>
__aicore__ inline void ForeachCastND<T_in, T_out>::SingleTensorProcess(int64_t dataCount)
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
            tempDataCount = copyTimesRemainder;
            isRemainder = true;
        }
        CopyIn(i, tempDataCount, isRemainder);
        ComputeAndCopyOut(i, tempDataCount, isRemainder);
    }
}

template <typename T_in, typename T_out>
__aicore__ inline void ForeachCastND<T_in, T_out>::ParseTilingData(const ForeachCommonTilingData* tilingData)
{
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    outputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T_in, typename T_out>
__aicore__ inline void ForeachCastND<T_in, T_out>::CopyIn(uint16_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T_in> dataLocal = dataQueue.AllocTensor<T_in>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T_in)), 0, 0, 0};
        DataCopyPadExtParams<T_in> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[1ULL * index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[1ULL * index * maxDataCount], dataCount);
    }
    dataQueue.EnQue(dataLocal);
}

template <typename T_in, typename T_out>
__aicore__ inline void ForeachCastND<T_in, T_out>::ComputeAndCopyOut(
    uint16_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T_in> dataLocal = dataQueue.DeQue<T_in>();
    LocalTensor<T_out> outLocal = outQueue.AllocTensor<T_out>();

    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T_out, float>::value) {
        Cast(outLocal, dataLocal, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(outLocal, dataLocal, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }

    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T_out)), 0, 0, 0};
        DataCopyPad(outTensorsGM[1ULL * index * maxDataCount], outLocal, copyParams);
    } else {
        DataCopy(outTensorsGM[1ULL * index * maxDataCount], outLocal, dataCount);
    }

    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

    dataQueue.FreeTensor(dataLocal);
    outQueue.FreeTensor(outLocal);
}

template <typename T_in, typename T_out>
__aicore__ inline __gm__ T_in* ForeachCastND<T_in, T_out>::GetInputTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T_in*>(*(retPtr + index));
}

template <typename T_in, typename T_out>
__aicore__ inline __gm__ T_out* ForeachCastND<T_in, T_out>::GetOutputTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T_out*>(*(retPtr + index));
}

} // namespace ForeachCast

#endif // FOREACH_CAST_N_D_H