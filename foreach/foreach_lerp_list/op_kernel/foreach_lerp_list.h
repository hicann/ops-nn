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
 * \file foreach_lerp_list.h
 * \brief
 */

#ifndef FOREACH_LERP_LIST_H
#define FOREACH_LERP_LIST_H

#include <type_traits>
#include "kernel_operator.h"

namespace ForeachLerpList {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

constexpr int32_t LERP_BUFFER_NUM = 2;

constexpr uint8_t COPY_SPACE_MULTIPLE = 9;

constexpr uint8_t INPUT_PARAMETER_COUNT = 4;

constexpr uint8_t BYTE_PER_BLOCK = 32;

constexpr int16_t MAX_REPEATS = 255;

constexpr uint32_t ELEMENT_PER_REPEAT = 64;

constexpr uint32_t FACTOR_NUMBER_TWO = 2;
constexpr uint32_t FACTOR_NUMBER_THREE = 3;
constexpr float FLOAT_NUM_NEG = -0.5f;
constexpr float FLOAT_NUM_POS = 0.5;
constexpr float FLOAT_NUM_ONE = 1.0;

template <typename T>
class InnerComputer
{
public:
    __aicore__ inline void Compute(
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local, LocalTensor<T>& weightLocal, LocalTensor<T>& yLocal,
        LocalTensor<float>& float32Tensor, LocalTensor<float>& lerpLocal1, LocalTensor<float>& lerpLocal2,
        LocalTensor<uint8_t>& maskLocal, uint32_t maxCastDataCount, int64_t dataCount)
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
            ComputePerCast(
                x1Local, x2Local, weightLocal, yLocal, float32Tensor, lerpLocal1, lerpLocal2, maskLocal,
                maxCastDataCount, i, maxCastDataCount);
        }

        if (castTimesRemainder > 0) {
            ComputePerCast(
                x1Local, x2Local, weightLocal, yLocal, float32Tensor, lerpLocal1, lerpLocal2, maskLocal,
                maxCastDataCount, castTimes, castTimesRemainder);
        }
    }

private:
    __aicore__ inline void ComputePerCast(
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local, LocalTensor<T>& weightLocal, LocalTensor<T>& yLocal,
        LocalTensor<float>& float32Tensor, LocalTensor<float>& lerpLocal1, LocalTensor<float>& lerpLocal2,
        LocalTensor<uint8_t>& maskLocal, uint32_t maxCastDataCount, uint16_t index, int64_t dataCount)
    {
        ComputePerCastInner(
            x1Local, x2Local, weightLocal, yLocal, float32Tensor, lerpLocal1, lerpLocal2, maskLocal, maxCastDataCount,
            index, maxCastDataCount);

        uint32_t totalRepeatCnt = dataCount / ELEMENT_PER_REPEAT;
        uint32_t totalRepeatCntRemainder = dataCount % ELEMENT_PER_REPEAT; // should calc
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;            // limit by L0 API, should calc
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;   // should calc
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            PipeBarrier<PIPE_V>();
            Select(
                float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE + offset], maskLocal[offset],
                float32Tensor[maxCastDataCount + offset],
                float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE + offset], SELMODE::VSEL_TENSOR_TENSOR_MODE,
                ELEMENT_PER_REPEAT, MAX_REPEATS, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * ELEMENT_PER_REPEAT;
        }
        if (repeatBatchCntRemainder > 0) {
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            PipeBarrier<PIPE_V>();
            Select(
                float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE + offset], maskLocal[offset],
                float32Tensor[maxCastDataCount + offset],
                float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE + offset], SELMODE::VSEL_TENSOR_TENSOR_MODE,
                ELEMENT_PER_REPEAT, repeatBatchCntRemainder, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * ELEMENT_PER_REPEAT;
        }
        if (totalRepeatCntRemainder > 0) {
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            PipeBarrier<PIPE_V>();
            Select(
                float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE + offset], maskLocal[offset],
                float32Tensor[maxCastDataCount + offset],
                float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE + offset], SELMODE::VSEL_TENSOR_TENSOR_MODE,
                totalRepeatCntRemainder, 1, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        Cast(
            yLocal[index * maxCastDataCount], float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE],
            RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputePerCastInner(
        LocalTensor<T>& x1Local, LocalTensor<T>& x2Local, LocalTensor<T>& weightLocal, LocalTensor<T>& yLocal,
        LocalTensor<float>& float32Tensor, LocalTensor<float>& lerpLocal1, LocalTensor<float>& lerpLocal2,
        LocalTensor<uint8_t>& maskLocal, uint32_t maxCastDataCount, uint16_t index, int64_t dataCount)
    {
        // y = x1 +  weightVal * (x2 - x1)
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor[maxCastDataCount], x2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(
            float32Tensor[maxCastDataCount * FACTOR_NUMBER_TWO], weightLocal[index * maxCastDataCount],
            RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Sub(float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount], float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Mul(float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount],
            float32Tensor[maxCastDataCount * FACTOR_NUMBER_TWO], dataCount);
        PipeBarrier<PIPE_V>();
        Add(float32Tensor[maxCastDataCount * FACTOR_NUMBER_THREE], float32Tensor, float32Tensor[maxCastDataCount],
            dataCount);
        PipeBarrier<PIPE_V>();

        // y = x2 - (x2 - x1)*(1 - weight)
        Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor[maxCastDataCount], x2Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(
            float32Tensor[maxCastDataCount * FACTOR_NUMBER_TWO], weightLocal[index * maxCastDataCount],
            RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(lerpLocal1, FLOAT_NUM_POS, dataCount);
        PipeBarrier<PIPE_V>();
        Abs(lerpLocal2, float32Tensor[maxCastDataCount * FACTOR_NUMBER_TWO], dataCount);
        PipeBarrier<PIPE_V>();
        Compare(
            maskLocal, lerpLocal2, lerpLocal1, CMPMODE::GE,
            (dataCount + ELEMENT_PER_REPEAT - 1) / ELEMENT_PER_REPEAT * ELEMENT_PER_REPEAT);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(lerpLocal1, FLOAT_NUM_ONE, dataCount);
        PipeBarrier<PIPE_V>();
        Sub(lerpLocal1, lerpLocal1, float32Tensor[maxCastDataCount * FACTOR_NUMBER_TWO], dataCount);
        PipeBarrier<PIPE_V>();
        Sub(float32Tensor, float32Tensor[maxCastDataCount], float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Mul(float32Tensor, float32Tensor, lerpLocal1, dataCount);
        PipeBarrier<PIPE_V>();
        Sub(float32Tensor[maxCastDataCount], float32Tensor[maxCastDataCount], float32Tensor, dataCount);
    }
};

template <>
class InnerComputer<float>
{
public:
    __aicore__ inline void Compute(
        LocalTensor<float>& x1Local, LocalTensor<float>& x2Local, LocalTensor<float>& weightLocal,
        LocalTensor<float>& yLocal, LocalTensor<float>& float32Tensor, LocalTensor<float>& lerpLocal1,
        LocalTensor<float>& lerpLocal2, LocalTensor<uint8_t>& maskLocal, uint32_t maxCastDataCount, int64_t dataCount)
    {
        // y = x1 +  weightVal * (x2 - x1)
        PipeBarrier<PIPE_V>();
        Sub(yLocal, x2Local, x1Local, dataCount);
        PipeBarrier<PIPE_V>();
        Mul(yLocal, yLocal, weightLocal, dataCount);
        PipeBarrier<PIPE_V>();
        Add(yLocal, yLocal, x1Local, dataCount);
        PipeBarrier<PIPE_V>();

        ComputeInner(
            x1Local, x2Local, weightLocal, yLocal, float32Tensor, lerpLocal1, lerpLocal2, maskLocal, dataCount);

        uint32_t totalRepeatCnt = dataCount / ELEMENT_PER_REPEAT;
        uint32_t totalRepeatCntRemainder = dataCount % ELEMENT_PER_REPEAT;
        uint32_t repeatBatchCnt = totalRepeatCnt / MAX_REPEATS;
        uint32_t repeatBatchCntRemainder = totalRepeatCnt % MAX_REPEATS;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < repeatBatchCnt; i++) {
            // todo
            // 已提issue。select接口里面有s操作的，循环操作的时候，需要插入一个s等v的同步，否则可能会出现一些同步依赖的问题
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            PipeBarrier<PIPE_V>();
            Select(
                yLocal[offset], maskLocal[offset], x2Local[offset], yLocal[offset], SELMODE::VSEL_TENSOR_TENSOR_MODE,
                ELEMENT_PER_REPEAT, MAX_REPEATS, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            offset += MAX_REPEATS * ELEMENT_PER_REPEAT;
        }

        if (repeatBatchCntRemainder > 0) {
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            PipeBarrier<PIPE_V>();
            Select(
                yLocal[offset], maskLocal[offset], x2Local[offset], yLocal[offset], SELMODE::VSEL_TENSOR_TENSOR_MODE,
                ELEMENT_PER_REPEAT, repeatBatchCntRemainder, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
            offset += repeatBatchCntRemainder * ELEMENT_PER_REPEAT;
        }

        if (totalRepeatCntRemainder > 0) {
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            PipeBarrier<PIPE_V>();
            Select(
                yLocal[offset], maskLocal[offset], x2Local[offset], yLocal[offset], SELMODE::VSEL_TENSOR_TENSOR_MODE,
                totalRepeatCntRemainder, 1, {1, 1, 1, 8, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeInner(
        LocalTensor<float>& x1Local, LocalTensor<float>& x2Local, LocalTensor<float>& weightLocal,
        LocalTensor<float>& yLocal, LocalTensor<float>& float32Tensor, LocalTensor<float>& lerpLocal1,
        LocalTensor<float>& lerpLocal2, LocalTensor<uint8_t>& maskLocal, int64_t dataCount)
    {
        // y = x2 - (x2 - x1) * (1 - weight)
        PipeBarrier<PIPE_V>();
        Duplicate<float>(lerpLocal1, FLOAT_NUM_POS, dataCount);
        PipeBarrier<PIPE_V>();
        Abs(lerpLocal2, weightLocal, dataCount);
        PipeBarrier<PIPE_V>();
        Compare(
            maskLocal, lerpLocal2, lerpLocal1, CMPMODE::GE,
            (dataCount + ELEMENT_PER_REPEAT - 1) / ELEMENT_PER_REPEAT * ELEMENT_PER_REPEAT);
        PipeBarrier<PIPE_V>();
        Duplicate<float>(lerpLocal1, FLOAT_NUM_ONE, dataCount);
        PipeBarrier<PIPE_V>();
        Sub(lerpLocal1, lerpLocal1, weightLocal, dataCount);
        PipeBarrier<PIPE_V>();
        Sub(x1Local, x2Local, x1Local, dataCount);
        PipeBarrier<PIPE_V>();
        Mul(x1Local, x1Local, lerpLocal1, dataCount);
        PipeBarrier<PIPE_V>();
        Sub(x2Local, x2Local, x1Local, dataCount);
        PipeBarrier<PIPE_V>();
    }
};

template <typename T>
class ForeachLerpListND
{
public:
    __aicore__ inline ForeachLerpListND(){};
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace,
        const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount, LocalTensor<float>& float32Tensor);
    __aicore__ inline void CopyIn(uint16_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void ComputeAndCopyOut(
        uint16_t index, int64_t dataCount, LocalTensor<float>& float32Tensor, bool isRemainder);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR gmAddr);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> weightQueue;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> lerpBuf;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    GlobalTensor<T> x1TensorGM;
    GlobalTensor<T> x2TensorGM;
    GlobalTensor<T> weightTensorGM;
    GlobalTensor<T> yTensorGM;

    GM_ADDR x1TensorPtr = nullptr;
    GM_ADDR x2TensorPtr = nullptr;
    GM_ADDR weightTensorPtr = nullptr;
    GM_ADDR yTensorPtr = nullptr;
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

    uint32_t maxCastDataCount = {0};
    int64_t maskBufSize = 0;
    int64_t lerpOffset = 0;
};

template <typename T>
__aicore__ inline void ForeachLerpListND<T>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    x1TensorPtr = x1;
    x2TensorPtr = x2;
    weightTensorPtr = weight;
    yTensorPtr = y;
    ParseTilingData(tilingData);
    weightTensorGM.SetGlobalBuffer((__gm__ T*)weight, 1);

// Init for bfloat16
#if __CCE_AICORE__ >= 220
    if (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
#else
    if (std::is_same<T, half>::value) {
#endif
        uint64_t totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
        pipe.InitBuffer(x1Queue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(weightQueue, BUFFER_NUM, totalTensorUbSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, totalTensorUbSize);
        maxDataCount = totalTensorUbSize / sizeof(T);
        pipe.InitBuffer(float32Queue, 1, inputsTensorUbSize * INPUT_PARAMETER_COUNT);
        LocalTensor<float> float32Tensor = float32Queue.AllocTensor<float>();
        float32Queue.EnQue(float32Tensor);
        maxCastDataCount = inputsTensorUbSize / sizeof(float);

        maskBufSize = (maxCastDataCount + BYTE_PER_BLOCK - 1) / BYTE_PER_BLOCK * BYTE_PER_BLOCK;
        pipe.InitBuffer(maskBuf, maskBufSize);
        pipe.InitBuffer(lerpBuf, inputsTensorUbSize * LERP_BUFFER_NUM);
        lerpOffset = maxCastDataCount;
    } else {
        pipe.InitBuffer(x1Queue, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(x2Queue, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(weightQueue, BUFFER_NUM, inputsTensorUbSize);
        pipe.InitBuffer(yQueue, BUFFER_NUM, inputsTensorUbSize);
        maxDataCount = inputsTensorUbSize / sizeof(T);

        maskBufSize = (maxDataCount + BYTE_PER_BLOCK - 1) / BYTE_PER_BLOCK * BYTE_PER_BLOCK;
        pipe.InitBuffer(maskBuf, maskBufSize);
        pipe.InitBuffer(lerpBuf, inputsTensorUbSize * LERP_BUFFER_NUM);
        lerpOffset = maxDataCount;
    }
}

template <typename T>
__aicore__ inline void ForeachLerpListND<T>::Process()
{
    /* 将中间量预留出来 */
    LocalTensor<float> float32Tensor;
#if __CCE_AICORE__ >= 220
    if (std::is_same<T, bfloat16_t>::value) {
        float32Tensor = float32Queue.DeQue<float>();
    }
#endif
    if (std::is_same<T, half>::value) {
        float32Tensor = float32Queue.DeQue<float>();
    }

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
        x1TensorGM.SetGlobalBuffer(GetTensorAddr(i, x1TensorPtr) + cursorStart);
        x2TensorGM.SetGlobalBuffer(GetTensorAddr(i, x2TensorPtr) + cursorStart);
        weightTensorGM.SetGlobalBuffer(GetTensorAddr(i, weightTensorPtr) + cursorStart);
        yTensorGM.SetGlobalBuffer(GetTensorAddr(i, yTensorPtr) + cursorStart);
        SingleTensorProcess(dataCount, float32Tensor);
    }
#if __CCE_AICORE__ >= 220
    if (std::is_same<T, bfloat16_t>::value) {
        float32Queue.FreeTensor(float32Tensor);
    }
#endif
    if (std::is_same<T, half>::value) {
        float32Queue.FreeTensor(float32Tensor);
    }
}

template <typename T>
__aicore__ inline void ForeachLerpListND<T>::SingleTensorProcess(int64_t dataCount, LocalTensor<float>& float32Tensor)
{
    // Batch handling and calculation.
    uint32_t copyTimes = dataCount / maxDataCount;
    uint32_t copyTimesRemainder = dataCount % maxDataCount;

    for (uint32_t i = 0; i < copyTimes; i++) {
        CopyIn(i, maxDataCount, false);
        ComputeAndCopyOut(i, maxDataCount, float32Tensor, false);
    }

    if (copyTimesRemainder > 0) {
        CopyIn(copyTimes, copyTimesRemainder, true);
        ComputeAndCopyOut(copyTimes, copyTimesRemainder, float32Tensor, true);
    }
}

template <typename T>
__aicore__ inline void ForeachLerpListND<T>::ParseTilingData(const ForeachCommonTilingData* tilingData)
{
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T>
__aicore__ inline void ForeachLerpListND<T>::CopyIn(uint16_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
    LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
    LocalTensor<T> weightLocal = weightQueue.AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(x1Local, x1TensorGM[index * maxDataCount], copyParams, padParams);
        DataCopyPad(x2Local, x2TensorGM[index * maxDataCount], copyParams, padParams);
        DataCopyPad(weightLocal, weightTensorGM[index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(x1Local, x1TensorGM[index * maxDataCount], dataCount);
        DataCopy(x2Local, x2TensorGM[index * maxDataCount], dataCount);
        DataCopy(weightLocal, weightTensorGM[index * maxDataCount], dataCount);
    }
    x1Queue.EnQue(x1Local);
    x2Queue.EnQue(x2Local);
    weightQueue.EnQue(weightLocal);
}

template <typename T>
__aicore__ inline void ForeachLerpListND<T>::ComputeAndCopyOut(
    uint16_t index, int64_t dataCount, LocalTensor<float>& float32Tensor, bool isRemainder)
{
    LocalTensor<T> x1Local = x1Queue.DeQue<T>();
    LocalTensor<T> x2Local = x2Queue.DeQue<T>();
    LocalTensor<T> weightLocal = weightQueue.DeQue<T>();
    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();

    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> lerpLocal = lerpBuf.Get<float>();
    LocalTensor<float> lerpLocal1 = lerpLocal;
    LocalTensor<float> lerpLocal2 = lerpLocal[lerpOffset];

    InnerComputer<T> computer;
    computer.Compute(
        x1Local, x2Local, weightLocal, yLocal, float32Tensor, lerpLocal1, lerpLocal2, maskLocal, maxCastDataCount,
        dataCount);

    yQueue.EnQue<T>(yLocal);
    x1Queue.FreeTensor(x1Local);
    x2Queue.FreeTensor(x2Local);
    weightQueue.FreeTensor(weightLocal);

    LocalTensor<T> retLocal = yQueue.DeQue<T>();
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(yTensorGM[index * maxDataCount], retLocal, copyParams);
    } else {
        DataCopy(yTensorGM[index * maxDataCount], retLocal, dataCount);
    }
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

    yQueue.FreeTensor(retLocal);
}

template <typename T>
__aicore__ inline __gm__ T* ForeachLerpListND<T>::GetTensorAddr(uint16_t index, GM_ADDR gmAddr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(gmAddr);
    uint64_t tensorPtrOffset = *dataAddr; // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + index));
}
} // namespace ForeachLerpList
#endif // FOREACH_LERP_LIST_H
