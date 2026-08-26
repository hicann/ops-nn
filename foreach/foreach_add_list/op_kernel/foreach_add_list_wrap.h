/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_add_list_wrap.h
 * \brief foreach_add_list int16/int8/uint8 kernel, 整数溢出按二进制补码回绕
 */

#ifndef FOREACH_ADD_LIST_WRAP_H
#define FOREACH_ADD_LIST_WRAP_H

#include "kernel_operator.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr int32_t ADD_LIST_WRAP_BUFFER_NUM = 2;

template <typename T, int32_t bufferNum = ADD_LIST_WRAP_BUFFER_NUM>
class ForeachAddListWrap {
public:
    __aicore__ inline ForeachAddListWrap(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y, GM_ADDR workspace,
                                const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void CopyIn2(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void CopyOut(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> dataQueue1;
    TQue<QuePosition::VECIN, bufferNum> dataQueue2;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;
    TBuf<QuePosition::VECCALC> halfBuf;
    TBuf<QuePosition::VECCALC> int16Buf;

    GlobalTensor<T> inTensorsGM1;
    GlobalTensor<T> inTensorsGM2;
    GlobalTensor<T> outTensorsGM;
    GlobalTensor<DTYPE_ALPHA> inScalarGM;

    GM_ADDR inTensorsPtr1 = nullptr;
    GM_ADDR inTensorsPtr2 = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int16_t alphaVal = 0;

    int64_t blockIdx = 0;
    uint32_t maxDataCount = 0;
    uint64_t inputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;
};

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y,
                                                              GM_ADDR workspace,
                                                              const ForeachCommonTilingData* tilingData)
{
    (void)workspace;
    blockIdx = GetBlockIdx();
    inTensorsPtr1 = x1;
    inTensorsPtr2 = x2;
    outTensorsPtr = y;
    ParseTilingData(tilingData);

    inScalarGM.SetGlobalBuffer((__gm__ DTYPE_ALPHA*)alpha, 1);
    alphaVal = static_cast<int16_t>(inScalarGM.GetValue(0));

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        // int8/uint8: 经 half 提升 int16 计算, 再按位掩码回绕到目标 dtype 低比特
        maxDataCount = static_cast<uint32_t>(inputsTensorUbSize);
        pipe.InitBuffer(dataQueue1, bufferNum, maxDataCount * sizeof(T));
        pipe.InitBuffer(dataQueue2, bufferNum, maxDataCount * sizeof(T));
        pipe.InitBuffer(outQueue, bufferNum, maxDataCount * sizeof(T));
        pipe.InitBuffer(halfBuf, ADD_LIST_WRAP_BUFFER_NUM * maxDataCount * sizeof(half));
        // [a16 | b16 | mask | lo | hi]
        pipe.InitBuffer(int16Buf, 5 * maxDataCount * sizeof(int16_t));
    } else {
        // int16: 直接 int16 域计算, 硬件按二进制补码回绕, 无需中间转换
        maxDataCount = static_cast<uint32_t>(inputsTensorUbSize / sizeof(T));
        pipe.InitBuffer(dataQueue1, bufferNum, maxDataCount * sizeof(T));
        pipe.InitBuffer(dataQueue2, bufferNum, maxDataCount * sizeof(T));
        pipe.InitBuffer(outQueue, bufferNum, maxDataCount * sizeof(T));
    }
}

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::ParseTilingData(const ForeachCommonTilingData* tilingData)
{
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::Process()
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
        inTensorsGM1.SetGlobalBuffer(GetTensorAddr(i, inTensorsPtr1) + cursorStart);
        inTensorsGM2.SetGlobalBuffer(GetTensorAddr(i, inTensorsPtr2) + cursorStart);
        outTensorsGM.SetGlobalBuffer(GetTensorAddr(i, outTensorsPtr) + cursorStart);
        SingleTensorProcess(dataCount);
    }
}

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::SingleTensorProcess(int64_t dataCount)
{
    uint32_t copyTimes = static_cast<uint32_t>(dataCount / maxDataCount);
    uint32_t copyTimesRemainder = static_cast<uint32_t>(dataCount % maxDataCount);
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
        CopyIn2(i, tempDataCount, isRemainder);
        Compute(i, tempDataCount, isRemainder);
        CopyOut(i, tempDataCount, isRemainder);
    }
}

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::CopyIn(uint32_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> dataLocal = dataQueue1.template AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM1[1ULL * index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM1[1ULL * index * maxDataCount], dataCount);
    }
    dataQueue1.EnQue(dataLocal);
}

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::CopyIn2(uint32_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> dataLocal = dataQueue2.template AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM2[1ULL * index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM2[1ULL * index * maxDataCount], dataCount);
    }
    dataQueue2.EnQue(dataLocal);
}

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::Compute(uint32_t index, int64_t dataCount, bool isRemainder)
{
    (void)index;
    (void)isRemainder;
    LocalTensor<T> inLocal1 = dataQueue1.template DeQue<T>();
    LocalTensor<T> inLocal2 = dataQueue2.template DeQue<T>();
    LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same_v<T, int16_t>) {
        // int16 域: a + alpha * b, 溢出按二进制补码回绕
        Muls(inLocal2, inLocal2, alphaVal, dataCount);
        PipeBarrier<PIPE_V>();
        Add(outLocal, inLocal1, inLocal2, dataCount);
        PipeBarrier<PIPE_V>();
    } else {
        // int8/uint8: 提升 int16 精确计算后取低 8 比特
        LocalTensor<half> h1 = halfBuf.GetWithOffset<half>(maxDataCount, 0);
        LocalTensor<half> h2 = halfBuf.GetWithOffset<half>(maxDataCount, maxDataCount * sizeof(half));
        LocalTensor<int16_t> a16 = int16Buf.GetWithOffset<int16_t>(maxDataCount, 0);
        LocalTensor<int16_t> b16 = int16Buf.GetWithOffset<int16_t>(maxDataCount, maxDataCount * sizeof(int16_t));
        LocalTensor<int16_t> mask16 = int16Buf.GetWithOffset<int16_t>(maxDataCount, 2 * maxDataCount * sizeof(int16_t));
        LocalTensor<int16_t> lo16 = int16Buf.GetWithOffset<int16_t>(maxDataCount, 3 * maxDataCount * sizeof(int16_t));
        LocalTensor<int16_t> hi16 = int16Buf.GetWithOffset<int16_t>(maxDataCount, 4 * maxDataCount * sizeof(int16_t));

        Cast(h1, inLocal1, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(h2, inLocal2, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(a16, h1, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(b16, h2, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
        Muls(b16, b16, alphaVal, dataCount);
        PipeBarrier<PIPE_V>();
        Add(a16, a16, b16, dataCount);
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same_v<T, uint8_t>) {
            // 无符号回绕: 保留低 8 位
            Duplicate(mask16, static_cast<int16_t>(0x00FF), dataCount);
            PipeBarrier<PIPE_V>();
            And(a16, a16, mask16, dataCount);
            PipeBarrier<PIPE_V>();
        } else {
            // 有符号回绕: (v & 0x7F) - (v & 0x80), 映射到 [-128, 127]
            Duplicate(mask16, static_cast<int16_t>(0x007F), dataCount);
            PipeBarrier<PIPE_V>();
            And(lo16, a16, mask16, dataCount);
            PipeBarrier<PIPE_V>();
            Duplicate(mask16, static_cast<int16_t>(0x0080), dataCount);
            PipeBarrier<PIPE_V>();
            And(hi16, a16, mask16, dataCount);
            PipeBarrier<PIPE_V>();
            Sub(a16, lo16, hi16, dataCount);
            PipeBarrier<PIPE_V>();
        }
        Cast(h1, a16, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, h1, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }

    outQueue.EnQue(outLocal);
    dataQueue1.FreeTensor(inLocal1);
    dataQueue2.FreeTensor(inLocal2);
}

template <typename T, int32_t bufferNum>
__aicore__ inline void ForeachAddListWrap<T, bufferNum>::CopyOut(uint32_t index, int64_t dataCount, bool isRemainder)
{
    LocalTensor<T> outLocal = outQueue.template DeQue<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPad(outTensorsGM[1ULL * index * maxDataCount], outLocal, copyParams);
    } else {
        DataCopy(outTensorsGM[1ULL * index * maxDataCount], outLocal, dataCount);
    }
    outQueue.FreeTensor(outLocal);
}

template <typename T, int32_t bufferNum>
__aicore__ inline __gm__ T* ForeachAddListWrap<T, bufferNum>::GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

} // namespace OpKernel
} // namespace Common

#endif // FOREACH_ADD_LIST_WRAP_H
