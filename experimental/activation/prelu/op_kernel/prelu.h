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
 * \file prelu.h
 * \brief Prelu 算子 kernel 类定义
 */

#ifndef PRELU_H
#define PRELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "prelu_tiling_data.h"
#include "prelu_tiling_key.h"

#include <type_traits>

namespace NsPrelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
__aicore__ inline void CopyGmToLocalPad(
    const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t dataNum, uint32_t alignedDataNum)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataNum * sizeof(T)), 0, 0, 0};
    uint32_t rightPadding = alignedDataNum > dataNum ? alignedDataNum - dataNum : 0;
    DataCopyPadExtParams<T> padParams{rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<T>(0)};
    DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyLocalToGmPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t dataNum)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataNum * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}

template <typename T>
class Prelu {
public:
    __aicore__ inline Prelu() {}

    __aicore__ inline void InitScalar(
        GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitChannel(
        GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitChannelSplitLParallel(
        GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitChannelNcResidentWeightReuse(
        GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitChannelNcWeightReuse(
        GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitChannelNcSplitCWeightReuse(
        GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void ProcessScalar();
    __aicore__ inline void ProcessChannelFullL();
    __aicore__ inline void ProcessChannelSplitL();
    __aicore__ inline void ProcessChannelSplitLParallel();
    __aicore__ inline void ProcessChannelNcResidentWeightReuse();
    __aicore__ inline void ProcessChannelNcWeightReuse();
    __aicore__ inline void ProcessChannelNcSplitCWeightReuse();

private:
    __aicore__ inline void CopyIn(int64_t progress, uint32_t currentNum);
    __aicore__ inline void CopyInByOffset(int64_t gmOffset, uint32_t currentNum, uint32_t alignedNum);
    __aicore__ inline void CopyInNcByRows(int64_t nOffset, int64_t tileRows);
    __aicore__ inline void CopyOut(int64_t progress, uint32_t currentNum);
    __aicore__ inline void CopyOutByOffset(int64_t gmOffset, uint32_t currentNum);
    __aicore__ inline void CopyOutNcByRows(int64_t nOffset, int64_t tileRows);
    __aicore__ inline void Compute(uint32_t currentNum);
    __aicore__ inline void ComputeNc(uint32_t computeLen);
    __aicore__ inline void BuildNcWeightVec(int64_t tileRows);
    __aicore__ inline void BuildNcResidentWeightVec(int64_t tileRows);
    __aicore__ inline void CopyWeightTile(int64_t cOffset, uint32_t realC, uint32_t alignedC);
    __aicore__ inline void LoadChannelWeight(int64_t channelIdx);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitNcBuffers();
    __aicore__ inline void InitNcResidentWeightBuffers();
    __aicore__ inline void SyncMte2ToV();
    __aicore__ inline void SyncMte2ToS();
    __aicore__ inline void SyncVToS();

private:
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> tmpXFp32;
    TBuf<TPosition::VECCALC> tmpBufPos;
    TBuf<TPosition::VECCALC> tmpBufNeg;
    TBuf<TPosition::VECCALC> weightBuf;
    TBuf<TPosition::VECCALC> weightVecBuf;
    TBuf<TPosition::VECCALC> weightFp32Buf;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    GM_ADDR weightGM = nullptr;
    T weightVal;
    float weightValFp32 = 0.0f;
    int64_t totalLength = 0;
    int64_t blockLength = 0;
    int64_t ubLength = 0;
    int64_t channelSize = 1;
    int64_t innerSize = 1;
    int64_t innerSizeAligned = 1;
    int64_t rowOffset = 0;
    int64_t blockRowNum = 0;
    int64_t taskOffset = 0;
    int64_t taskNum = 0;
    int64_t tilesPerRow = 0;
    int64_t cTileLength = 1;
    int64_t alignedChannelSize = 1;
    int64_t alignedWeightSize = 1;
    int64_t activeChannelSize = 1;
    int64_t rowsPerTile = 1;
};

template <typename T>
__aicore__ inline void Prelu<T>::SyncMte2ToV()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
}

template <typename T>
__aicore__ inline void Prelu<T>::SyncMte2ToS()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventId);
    WaitFlag<HardEvent::MTE2_S>(eventId);
}

template <typename T>
__aicore__ inline void Prelu<T>::SyncVToS()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

__aicore__ inline float LoadBf16ScalarAsFloat(GM_ADDR weight)
{
    uint16_t weightBits = *((__gm__ uint16_t*)weight);
    uint32_t floatBits = static_cast<uint32_t>(weightBits) << 16;
    return *reinterpret_cast<float*>(&floatBits);
}

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1U) / align) * align;
}

template <typename T>
__aicore__ inline void FillWeightSpan(const LocalTensor<T>& weightVec, int64_t offset, int64_t length, T value)
{
    uint32_t alignElements = static_cast<uint32_t>(32U / sizeof(T));
    int64_t current = offset;
    int64_t end = offset + length;
    while (current < end && (current % alignElements) != 0) {
        weightVec.SetValue(current, value);
        ++current;
    }

    int64_t duplicateLength = ((end - current) / alignElements) * alignElements;
    if (duplicateLength > 0) {
        Duplicate(weightVec[current], value, static_cast<uint32_t>(duplicateLength));
        current += duplicateLength;
    }

    while (current < end) {
        weightVec.SetValue(current, value);
        ++current;
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::InitBuffers()
{
    pipe_->InitBuffer(inputQueueX, BUFFER_NUM, ubLength * sizeof(T));
    pipe_->InitBuffer(outputQueueY, BUFFER_NUM, ubLength * sizeof(T));
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        pipe_->InitBuffer(tmpXFp32, ubLength * sizeof(float));
        pipe_->InitBuffer(tmpBufPos, ubLength * sizeof(float));
        pipe_->InitBuffer(tmpBufNeg, ubLength * sizeof(float));
    } else {
        pipe_->InitBuffer(tmpBufPos, ubLength * sizeof(T));
        pipe_->InitBuffer(tmpBufNeg, ubLength * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::InitNcBuffers()
{
    pipe_->InitBuffer(inputQueueX, BUFFER_NUM, ubLength * sizeof(T));
    pipe_->InitBuffer(outputQueueY, BUFFER_NUM, ubLength * sizeof(T));
    pipe_->InitBuffer(weightBuf, alignedWeightSize * sizeof(T));
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        pipe_->InitBuffer(tmpXFp32, ubLength * sizeof(float));
        pipe_->InitBuffer(tmpBufPos, ubLength * sizeof(float));
        pipe_->InitBuffer(tmpBufNeg, ubLength * sizeof(float));
        pipe_->InitBuffer(weightVecBuf, ubLength * sizeof(float));
        pipe_->InitBuffer(weightFp32Buf, alignedWeightSize * sizeof(float));
    } else {
        pipe_->InitBuffer(tmpBufPos, ubLength * sizeof(T));
        pipe_->InitBuffer(tmpBufNeg, ubLength * sizeof(T));
        pipe_->InitBuffer(weightVecBuf, ubLength * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::InitNcResidentWeightBuffers()
{
    pipe_->InitBuffer(inputQueueX, BUFFER_NUM, ubLength * sizeof(T));
    pipe_->InitBuffer(outputQueueY, BUFFER_NUM, ubLength * sizeof(T));
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        pipe_->InitBuffer(weightBuf, alignedWeightSize * sizeof(T));
        pipe_->InitBuffer(tmpXFp32, ubLength * sizeof(float));
        pipe_->InitBuffer(tmpBufPos, ubLength * sizeof(float));
        pipe_->InitBuffer(tmpBufNeg, ubLength * sizeof(float));
        pipe_->InitBuffer(weightVecBuf, ubLength * sizeof(float));
        pipe_->InitBuffer(weightFp32Buf, alignedWeightSize * sizeof(float));
    } else {
        pipe_->InitBuffer(tmpBufPos, ubLength * sizeof(T));
        pipe_->InitBuffer(tmpBufNeg, ubLength * sizeof(T));
        pipe_->InitBuffer(weightVecBuf, ubLength * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::InitScalar(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    int64_t blockIdx = GetBlockIdx();
    ubLength = tilingData->tileLength;
    int64_t blockOffset = blockIdx * tilingData->formerLength;
    if (blockIdx < tilingData->formerNum) {
        blockLength = tilingData->formerLength;
    } else if (blockIdx < tilingData->usedCoreNum) {
        blockLength = tilingData->tailLength;
    } else {
        blockLength = 0;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, blockLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, blockLength);

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        weightValFp32 = LoadBf16ScalarAsFloat(weight);
    } else {
        T scalarWeight = *((__gm__ T*)weight);
        weightVal = scalarWeight;
    }

    InitBuffers();
}

template <typename T>
__aicore__ inline void Prelu<T>::InitChannel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    int64_t blockIdx = GetBlockIdx();
    ubLength = tilingData->tileLength;
    channelSize = tilingData->channelSize;
    innerSize = tilingData->innerSize;
    innerSizeAligned = tilingData->innerSizeAligned;
    totalLength = tilingData->totalLength;
    weightGM = weight;

    if (blockIdx < tilingData->extraRows) {
        blockRowNum = tilingData->baseRows + 1;
        rowOffset = blockIdx * (tilingData->baseRows + 1);
    } else if (blockIdx < tilingData->usedCoreNum) {
        blockRowNum = tilingData->baseRows;
        rowOffset = tilingData->extraRows * (tilingData->baseRows + 1) +
                    (blockIdx - tilingData->extraRows) * tilingData->baseRows;
    } else {
        blockRowNum = 0;
        rowOffset = 0;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x, tilingData->totalLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, tilingData->totalLength);

    InitBuffers();
}

template <typename T>
__aicore__ inline void Prelu<T>::InitChannelSplitLParallel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    int64_t blockIdx = GetBlockIdx();
    ubLength = tilingData->tileLength;
    channelSize = tilingData->channelSize;
    innerSize = tilingData->innerSize;
    tilesPerRow = tilingData->tilesPerRow;
    weightGM = weight;

    if (blockIdx < tilingData->extraTasks) {
        taskNum = tilingData->baseTasks + 1;
        taskOffset = blockIdx * (tilingData->baseTasks + 1);
    } else if (blockIdx < tilingData->usedCoreNum) {
        taskNum = tilingData->baseTasks;
        taskOffset = tilingData->extraTasks * (tilingData->baseTasks + 1) +
                     (blockIdx - tilingData->extraTasks) * tilingData->baseTasks;
    } else {
        taskNum = 0;
        taskOffset = 0;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x, tilingData->totalLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, tilingData->totalLength);

    InitBuffers();
}

template <typename T>
__aicore__ inline void Prelu<T>::InitChannelNcResidentWeightReuse(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    int64_t blockIdx = GetBlockIdx();
    ubLength = tilingData->tileLength;
    channelSize = tilingData->channelSize;
    innerSize = tilingData->innerSize;
    alignedChannelSize = tilingData->innerSizeAligned;
    alignedWeightSize = AlignUp(static_cast<uint32_t>(channelSize), static_cast<uint32_t>(32U / sizeof(T)));
    activeChannelSize = channelSize;
    rowsPerTile = ubLength / alignedChannelSize;
    weightGM = weight;

    if (blockIdx < tilingData->extraRows) {
        blockRowNum = tilingData->baseRows + 1;
        rowOffset = blockIdx * (tilingData->baseRows + 1);
    } else if (blockIdx < tilingData->usedCoreNum) {
        blockRowNum = tilingData->baseRows;
        rowOffset = tilingData->extraRows * (tilingData->baseRows + 1) +
                    (blockIdx - tilingData->extraRows) * tilingData->baseRows;
    } else {
        blockRowNum = 0;
        rowOffset = 0;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x, tilingData->totalLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, tilingData->totalLength);

    InitNcResidentWeightBuffers();

    GlobalTensor<T> weightTensor;
    weightTensor.SetGlobalBuffer((__gm__ T*)weightGM, channelSize);
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<T> weightLocal = weightBuf.Get<T>();
        CopyGmToLocalPad(weightLocal, weightTensor, static_cast<uint32_t>(channelSize),
            static_cast<uint32_t>(alignedWeightSize));
        SyncMte2ToV();
        LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedWeightSize));
    } else {
        LocalTensor<T> weightVec = weightVecBuf.Get<T>();
        CopyGmToLocalPad(weightVec, weightTensor, static_cast<uint32_t>(channelSize),
            static_cast<uint32_t>(alignedWeightSize));
    }
    BuildNcResidentWeightVec(rowsPerTile);
}

template <typename T>
__aicore__ inline void Prelu<T>::InitChannelNcWeightReuse(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    int64_t blockIdx = GetBlockIdx();
    ubLength = tilingData->tileLength;
    channelSize = tilingData->channelSize;
    innerSize = tilingData->innerSize;
    alignedChannelSize = tilingData->innerSizeAligned;
    alignedWeightSize = AlignUp(static_cast<uint32_t>(channelSize), static_cast<uint32_t>(32U / sizeof(T)));
    activeChannelSize = channelSize;
    rowsPerTile = ubLength / alignedChannelSize;
    weightGM = weight;

    if (blockIdx < tilingData->extraRows) {
        blockRowNum = tilingData->baseRows + 1;
        rowOffset = blockIdx * (tilingData->baseRows + 1);
    } else if (blockIdx < tilingData->usedCoreNum) {
        blockRowNum = tilingData->baseRows;
        rowOffset = tilingData->extraRows * (tilingData->baseRows + 1) +
                    (blockIdx - tilingData->extraRows) * tilingData->baseRows;
    } else {
        blockRowNum = 0;
        rowOffset = 0;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x, tilingData->totalLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, tilingData->totalLength);

    InitNcBuffers();

    GlobalTensor<T> weightTensor;
    weightTensor.SetGlobalBuffer((__gm__ T*)weightGM, channelSize);
    LocalTensor<T> weightLocal = weightBuf.Get<T>();
    CopyGmToLocalPad(weightLocal, weightTensor, static_cast<uint32_t>(channelSize),
        static_cast<uint32_t>(alignedWeightSize));
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        SyncMte2ToV();
        LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedWeightSize));
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::InitChannelNcSplitCWeightReuse(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, const PreluTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    int64_t blockIdx = GetBlockIdx();
    channelSize = tilingData->channelSize;
    innerSize = tilingData->innerSize;
    totalLength = tilingData->totalLength;
    cTileLength = tilingData->tileLength;
    ubLength = innerSize == 1 ? tilingData->tileLength : tilingData->innerSizeAligned;
    alignedChannelSize = tilingData->innerSizeAligned;
    alignedWeightSize = innerSize == 1 ? alignedChannelSize : cTileLength;
    activeChannelSize = alignedWeightSize;
    tilesPerRow = tilingData->tilesPerRow;
    weightGM = weight;

    if (blockIdx < tilingData->extraTasks) {
        taskNum = tilingData->baseTasks + 1;
        taskOffset = blockIdx * (tilingData->baseTasks + 1);
    } else if (blockIdx < tilingData->usedCoreNum) {
        taskNum = tilingData->baseTasks;
        taskOffset = tilingData->extraTasks * (tilingData->baseTasks + 1) +
                     (blockIdx - tilingData->extraTasks) * tilingData->baseTasks;
    } else {
        taskNum = 0;
        taskOffset = 0;
    }

    inputGMX.SetGlobalBuffer((__gm__ T*)x, tilingData->totalLength);
    outputGMY.SetGlobalBuffer((__gm__ T*)y, tilingData->totalLength);

    InitNcBuffers();
}

template <typename T>
__aicore__ inline void Prelu<T>::CopyIn(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    CopyGmToLocalPad(xLocal, inputGMX[progress * ubLength], currentNum, currentNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Prelu<T>::CopyInByOffset(int64_t gmOffset, uint32_t currentNum, uint32_t alignedNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    CopyGmToLocalPad(xLocal, inputGMX[gmOffset], currentNum, alignedNum);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Prelu<T>::CopyInNcByRows(int64_t nOffset, int64_t tileRows)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    int64_t rowElements = channelSize * innerSize;
    if (rowElements == alignedChannelSize) {
        uint32_t copyLen = static_cast<uint32_t>(tileRows * rowElements);
        CopyGmToLocalPad(xLocal, inputGMX[nOffset * rowElements], copyLen, copyLen);
        inputQueueX.EnQue(xLocal);
    } else {
        for (int64_t row = 0; row < tileRows; ++row) {
            int64_t gmOffset = (nOffset + row) * rowElements;
            int64_t localOffset = row * alignedChannelSize;
            CopyGmToLocalPad(xLocal[localOffset], inputGMX[gmOffset], static_cast<uint32_t>(rowElements),
                static_cast<uint32_t>(alignedChannelSize));
        }
        inputQueueX.EnQue(xLocal);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::CopyOut(int64_t progress, uint32_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    CopyLocalToGmPad(outputGMY[progress * ubLength], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Prelu<T>::CopyOutByOffset(int64_t gmOffset, uint32_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    CopyLocalToGmPad(outputGMY[gmOffset], yLocal, currentNum);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Prelu<T>::CopyOutNcByRows(int64_t nOffset, int64_t tileRows)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    int64_t rowElements = channelSize * innerSize;
    if (rowElements == alignedChannelSize) {
        CopyLocalToGmPad(outputGMY[nOffset * rowElements], yLocal, static_cast<uint32_t>(tileRows * rowElements));
        outputQueueY.FreeTensor(yLocal);
    } else {
        for (int64_t row = 0; row < tileRows; ++row) {
            int64_t gmOffset = (nOffset + row) * rowElements;
            int64_t localOffset = row * alignedChannelSize;
            CopyLocalToGmPad(outputGMY[gmOffset], yLocal[localOffset], static_cast<uint32_t>(rowElements));
        }
        outputQueueY.FreeTensor(yLocal);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::Compute(uint32_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> xFp32 = tmpXFp32.Get<float>();
        LocalTensor<float> pos = tmpBufPos.Get<float>();
        LocalTensor<float> neg = tmpBufNeg.Get<float>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, currentNum);
        Maxs(pos, xFp32, 0.0f, currentNum);
        Mins(neg, xFp32, 0.0f, currentNum);
        Muls(neg, neg, weightValFp32, currentNum);
        Add(pos, pos, neg, currentNum);
        Cast(yLocal, pos, RoundMode::CAST_RINT, currentNum);
    } else {
        LocalTensor<T> pos = tmpBufPos.Get<T>();
        LocalTensor<T> neg = tmpBufNeg.Get<T>();
        Maxs(pos, xLocal, static_cast<T>(0), currentNum);
        Mins(neg, xLocal, static_cast<T>(0), currentNum);
        Muls(neg, neg, weightVal, currentNum);
        Add(yLocal, pos, neg, currentNum);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Prelu<T>::BuildNcWeightVec(int64_t tileRows)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> weightLocal = weightFp32Buf.Get<float>();
        LocalTensor<float> weightVec = weightVecBuf.Get<float>();
        if (innerSize == 1) {
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < tileRows; ++row) {
                int64_t rowOffset = row * alignedChannelSize;
                DataCopy(weightVec[rowOffset], weightLocal, static_cast<uint32_t>(alignedWeightSize));
            }
            PipeBarrier<PIPE_V>();
        } else {
            SyncVToS();
            int64_t localOffset = 0;
            for (int64_t channelIdx = 0; channelIdx < activeChannelSize; ++channelIdx) {
                float weightValue = weightLocal.GetValue(channelIdx);
                FillWeightSpan(weightVec, localOffset, innerSize, weightValue);
                localOffset += innerSize;
            }
            PipeBarrier<PIPE_V>();
            if (tileRows > 1) {
                for (int64_t row = 1; row < tileRows; ++row) {
                    DataCopy(weightVec[row * alignedChannelSize], weightVec,
                        static_cast<uint32_t>(alignedChannelSize));
                }
                PipeBarrier<PIPE_V>();
            }
        }
    } else {
        LocalTensor<T> weightLocal = weightBuf.Get<T>();
        LocalTensor<T> weightVec = weightVecBuf.Get<T>();
        if (innerSize == 1) {
            SyncMte2ToV();
            for (int64_t row = 0; row < tileRows; ++row) {
                int64_t rowOffset = row * alignedChannelSize;
                DataCopy(weightVec[rowOffset], weightLocal, static_cast<uint32_t>(alignedWeightSize));
            }
            PipeBarrier<PIPE_V>();
        } else {
            SyncMte2ToS();
            int64_t localOffset = 0;
            for (int64_t channelIdx = 0; channelIdx < activeChannelSize; ++channelIdx) {
                T weightValue = weightLocal.GetValue(channelIdx);
                FillWeightSpan(weightVec, localOffset, innerSize, weightValue);
                localOffset += innerSize;
            }
            PipeBarrier<PIPE_V>();
            if (tileRows > 1) {
                for (int64_t row = 1; row < tileRows; ++row) {
                    DataCopy(weightVec[row * alignedChannelSize], weightVec,
                        static_cast<uint32_t>(alignedChannelSize));
                }
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::BuildNcResidentWeightVec(int64_t tileRows)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> weightLocal = weightFp32Buf.Get<float>();
        LocalTensor<float> weightVec = weightVecBuf.Get<float>();
        PipeBarrier<PIPE_V>();
        DataCopy(weightVec, weightLocal, static_cast<uint32_t>(alignedWeightSize));
        PipeBarrier<PIPE_V>();
        int64_t copiedRows = 1;
        while (copiedRows < tileRows) {
            int64_t rowsToCopy = tileRows - copiedRows;
            rowsToCopy = rowsToCopy > copiedRows ? copiedRows : rowsToCopy;
            DataCopy(weightVec[copiedRows * alignedChannelSize], weightVec,
                static_cast<uint32_t>(rowsToCopy * alignedChannelSize));
            PipeBarrier<PIPE_V>();
            copiedRows += rowsToCopy;
        }
    } else {
        LocalTensor<T> weightVec = weightVecBuf.Get<T>();
        SyncMte2ToV();
        int64_t copiedRows = 1;
        while (copiedRows < tileRows) {
            int64_t rowsToCopy = tileRows - copiedRows;
            rowsToCopy = rowsToCopy > copiedRows ? copiedRows : rowsToCopy;
            DataCopy(weightVec[copiedRows * alignedChannelSize], weightVec,
                static_cast<uint32_t>(rowsToCopy * alignedChannelSize));
            PipeBarrier<PIPE_V>();
            copiedRows += rowsToCopy;
        }
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ComputeNc(uint32_t computeLen)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> xFp32 = tmpXFp32.Get<float>();
        LocalTensor<float> pos = tmpBufPos.Get<float>();
        LocalTensor<float> neg = tmpBufNeg.Get<float>();
        LocalTensor<float> weightVec = weightVecBuf.Get<float>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, computeLen);
        Maxs(pos, xFp32, 0.0f, computeLen);
        Mins(neg, xFp32, 0.0f, computeLen);
        Mul(neg, neg, weightVec, computeLen);
        Add(pos, pos, neg, computeLen);
        Cast(yLocal, pos, RoundMode::CAST_RINT, computeLen);
    } else {
        LocalTensor<T> pos = tmpBufPos.Get<T>();
        LocalTensor<T> neg = tmpBufNeg.Get<T>();
        LocalTensor<T> weightVec = weightVecBuf.Get<T>();
        Maxs(pos, xLocal, static_cast<T>(0), computeLen);
        Mins(neg, xLocal, static_cast<T>(0), computeLen);
        Mul(neg, neg, weightVec, computeLen);
        Add(yLocal, pos, neg, computeLen);
    }

    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Prelu<T>::CopyWeightTile(int64_t cOffset, uint32_t realC, uint32_t alignedC)
{
    GlobalTensor<T> weightTensor;
    weightTensor.SetGlobalBuffer((__gm__ T*)weightGM + cOffset, realC);
    LocalTensor<T> weightLocal = weightBuf.Get<T>();
    CopyGmToLocalPad(weightLocal, weightTensor, realC, alignedC);
    activeChannelSize = realC;
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        SyncMte2ToV();
        LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignedC);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::LoadChannelWeight(int64_t channelIdx)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        weightValFp32 = LoadBf16ScalarAsFloat((GM_ADDR)((__gm__ T*)weightGM + channelIdx));
    } else {
        weightVal = *((__gm__ T*)weightGM + channelIdx);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ProcessScalar()
{
    uint32_t alignElements = static_cast<uint32_t>(32U / sizeof(T));
    int64_t tileNum = (blockLength + ubLength - 1) / ubLength;
    for (int64_t i = 0; i < tileNum; ++i) {
        uint32_t currentNum = static_cast<uint32_t>((i == tileNum - 1) ? (blockLength - i * ubLength) : ubLength);
        uint32_t computeLen = AlignUp(currentNum, alignElements);
        CopyInByOffset(i * ubLength, currentNum, computeLen);
        Compute(computeLen);
        CopyOut(i, currentNum);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ProcessChannelFullL()
{
    uint32_t realLen = static_cast<uint32_t>(innerSize);
    uint32_t computeLen = static_cast<uint32_t>(innerSizeAligned);
    for (int64_t rowProgress = 0; rowProgress < blockRowNum; ++rowProgress) {
        int64_t rowIdx = rowOffset + rowProgress;
        int64_t channelIdx = rowIdx % channelSize;
        int64_t gmOffset = rowIdx * innerSize;
        LoadChannelWeight(channelIdx);
        CopyInByOffset(gmOffset, realLen, computeLen);
        Compute(computeLen);
        CopyOutByOffset(gmOffset, realLen);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ProcessChannelSplitL()
{
    uint32_t alignElements = static_cast<uint32_t>(32U / sizeof(T));
    for (int64_t rowProgress = 0; rowProgress < blockRowNum; ++rowProgress) {
        int64_t rowIdx = rowOffset + rowProgress;
        int64_t channelIdx = rowIdx % channelSize;
        LoadChannelWeight(channelIdx);

        for (int64_t tileOffset = 0; tileOffset < innerSize; tileOffset += ubLength) {
            int64_t remainLen = innerSize - tileOffset;
            uint32_t realLen = static_cast<uint32_t>(remainLen > ubLength ? ubLength : remainLen);
            uint32_t computeLen = AlignUp(realLen, alignElements);
            int64_t gmOffset = rowIdx * innerSize + tileOffset;

            CopyInByOffset(gmOffset, realLen, computeLen);
            Compute(computeLen);
            CopyOutByOffset(gmOffset, realLen);
        }
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ProcessChannelSplitLParallel()
{
    uint32_t alignElements = static_cast<uint32_t>(32U / sizeof(T));
    int64_t lastChannelIdx = -1;
    for (int64_t taskProgress = 0; taskProgress < taskNum; ++taskProgress) {
        int64_t taskIdx = taskOffset + taskProgress;
        int64_t rowIdx = taskIdx / tilesPerRow;
        int64_t tileIdx = taskIdx % tilesPerRow;
        int64_t tileOffset = tileIdx * ubLength;
        int64_t remainLen = innerSize - tileOffset;
        uint32_t realLen = static_cast<uint32_t>(remainLen > ubLength ? ubLength : remainLen);
        uint32_t computeLen = AlignUp(realLen, alignElements);
        int64_t channelIdx = rowIdx % channelSize;
        int64_t gmOffset = rowIdx * innerSize + tileOffset;

        if (channelIdx != lastChannelIdx) {
            LoadChannelWeight(channelIdx);
            lastChannelIdx = channelIdx;
        }
        CopyInByOffset(gmOffset, realLen, computeLen);
        Compute(computeLen);
        CopyOutByOffset(gmOffset, realLen);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ProcessChannelNcResidentWeightReuse()
{
    if (blockRowNum <= 0) {
        return;
    }
    for (int64_t rowProgress = 0; rowProgress < blockRowNum; rowProgress += rowsPerTile) {
        int64_t tileRows = blockRowNum - rowProgress;
        tileRows = tileRows > rowsPerTile ? rowsPerTile : tileRows;
        uint32_t computeLen = static_cast<uint32_t>(tileRows * alignedChannelSize);
        int64_t nOffset = rowOffset + rowProgress;

        CopyInNcByRows(nOffset, tileRows);
        ComputeNc(computeLen);
        CopyOutNcByRows(nOffset, tileRows);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ProcessChannelNcWeightReuse()
{
    if (blockRowNum <= 0) {
        return;
    }
    BuildNcWeightVec(rowsPerTile);
    for (int64_t rowProgress = 0; rowProgress < blockRowNum; rowProgress += rowsPerTile) {
        int64_t tileRows = blockRowNum - rowProgress;
        tileRows = tileRows > rowsPerTile ? rowsPerTile : tileRows;
        uint32_t computeLen = static_cast<uint32_t>(tileRows * alignedChannelSize);
        int64_t nOffset = rowOffset + rowProgress;

        CopyInNcByRows(nOffset, tileRows);
        ComputeNc(computeLen);
        CopyOutNcByRows(nOffset, tileRows);
    }
}

template <typename T>
__aicore__ inline void Prelu<T>::ProcessChannelNcSplitCWeightReuse()
{
    uint32_t alignElements = static_cast<uint32_t>(32U / sizeof(T));
    int64_t batchSize = totalLength / (channelSize * innerSize);
    int64_t lastCTileIdx = -1;
    for (int64_t taskProgress = 0; taskProgress < taskNum; ++taskProgress) {
        int64_t taskIdx = taskOffset + taskProgress;
        int64_t cTileIdx = taskIdx / batchSize;
        int64_t nIdx = taskIdx % batchSize;
        int64_t cTileChannels = innerSize == 1 ? alignedChannelSize : cTileLength;
        int64_t cOffset = cTileIdx * cTileChannels;
        int64_t remainC = channelSize - cOffset;
        uint32_t realC = static_cast<uint32_t>(remainC > cTileChannels ? cTileChannels : remainC);
        uint32_t realLen = static_cast<uint32_t>(realC * innerSize);
        uint32_t computeLen = innerSize == 1 ? static_cast<uint32_t>(alignedChannelSize) :
            AlignUp(realLen, alignElements);
        uint32_t weightLen = innerSize == 1 ? computeLen : AlignUp(realC, alignElements);
        int64_t gmOffset = nIdx * channelSize * innerSize + cOffset * innerSize;

        if (cTileIdx != lastCTileIdx) {
            CopyWeightTile(cOffset, realC, weightLen);
            BuildNcWeightVec(1);
            lastCTileIdx = cTileIdx;
        }
        CopyInByOffset(gmOffset, realLen, computeLen);
        ComputeNc(computeLen);
        CopyOutByOffset(gmOffset, realLen);
    }
}

} // namespace NsPrelu
#endif // PRELU_H
