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
 * \file foreach_int_to_float_unary.h
 * \brief 整进浮出一元算子 kernel: 整数(int16/int8/uint8)输入张量列表提升 float32 计算,
 *        输出 float32 张量列表, 对齐 PyTorch(竞品)语义; 溢出遵循 IEEE-754(饱和至 inf)
 */

#ifndef FOREACH_INT_TO_FLOAT_UNARY_H
#define FOREACH_INT_TO_FLOAT_UNARY_H

#include "kernel_operator.h"

namespace Common {
namespace OpKernel {
using namespace AscendC;

constexpr int32_t INT_TO_FLOAT_BUFFER_NUM = 2;

using IntToFloatOp = void(const LocalTensor<float>&, const LocalTensor<float>&, const int32_t&);

template <typename T, IntToFloatOp* op, int32_t bufferNum = INT_TO_FLOAT_BUFFER_NUM>
class ForeachIntToFloatUnary {
public:
    __aicore__ inline ForeachIntToFloatUnary(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const ForeachCommonTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const ForeachCommonTilingData* tilingData);
    __aicore__ inline void SingleTensorProcess(int64_t dataCount);
    __aicore__ inline void CopyIn(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void Compute(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline void CopyOut(uint32_t index, int64_t dataCount, bool isRemainder);
    __aicore__ inline __gm__ T* GetInputTensorAddr(uint16_t index, GM_ADDR tensorPtr);
    __aicore__ inline __gm__ float* GetOutputTensorAddr(uint16_t index, GM_ADDR tensorPtr);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> dataQueue;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;
    TBuf<QuePosition::VECCALC> halfBuf;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<float> outTensorsGM;

    GM_ADDR inTensorsPtr = nullptr;
    GM_ADDR outTensorsPtr = nullptr;

    int64_t blockIdx = 0;
    uint32_t maxDataCount = 0;
    uint64_t inputsTensorUbSize = 0;
    const int64_t* tensorDataCountList = nullptr;
    uint16_t tensorStart = 0;
    uint16_t tensorEnd = 0;
    int64_t tensorStartOffset = 0;
    int64_t tensorEndOffset = 0;
};

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline void ForeachIntToFloatUnary<T, op, bufferNum>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                      const ForeachCommonTilingData* tilingData)
{
    (void)workspace;
    blockIdx = GetBlockIdx();
    inTensorsPtr = x;
    outTensorsPtr = y;
    ParseTilingData(tilingData);

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        // int8/uint8: 经 half 跳板提升 float32 计算(int8 精度无损, uint8 无符号需经 half 保真)
        maxDataCount = static_cast<uint32_t>(inputsTensorUbSize);
        pipe.InitBuffer(dataQueue, bufferNum, maxDataCount * sizeof(T));
        pipe.InitBuffer(outQueue, bufferNum, maxDataCount * sizeof(float));
        pipe.InitBuffer(halfBuf, INT_TO_FLOAT_BUFFER_NUM * maxDataCount * sizeof(half));
    } else {
        // int16: 直接 Cast 提升 float32
        maxDataCount = static_cast<uint32_t>(inputsTensorUbSize / sizeof(T));
        pipe.InitBuffer(dataQueue, bufferNum, maxDataCount * sizeof(T));
        pipe.InitBuffer(outQueue, bufferNum, maxDataCount * sizeof(float));
    }
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline void ForeachIntToFloatUnary<T, op, bufferNum>::ParseTilingData(
    const ForeachCommonTilingData* tilingData)
{
    inputsTensorUbSize = tilingData->inputsTensorUbSize;
    tensorDataCountList = tilingData->tensorDataCountList;
    tensorStart = tilingData->tensorStartList[blockIdx];
    tensorEnd = tilingData->tensorEndList[blockIdx];
    tensorStartOffset = tilingData->tensorStartOffsetList[blockIdx];
    tensorEndOffset = tilingData->tensorEndOffsetList[blockIdx];
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline void ForeachIntToFloatUnary<T, op, bufferNum>::Process()
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
        inTensorsGM.SetGlobalBuffer(GetInputTensorAddr(i, inTensorsPtr) + cursorStart);
        outTensorsGM.SetGlobalBuffer(GetOutputTensorAddr(i, outTensorsPtr) + cursorStart);
        SingleTensorProcess(dataCount);
    }
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline void ForeachIntToFloatUnary<T, op, bufferNum>::SingleTensorProcess(int64_t dataCount)
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
        Compute(i, tempDataCount, isRemainder);
        CopyOut(i, tempDataCount, isRemainder);
    }
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline void ForeachIntToFloatUnary<T, op, bufferNum>::CopyIn(uint32_t index, int64_t dataCount,
                                                                        bool isRemainder)
{
    LocalTensor<T> dataLocal = dataQueue.template AllocTensor<T>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[1ULL * index * maxDataCount], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[1ULL * index * maxDataCount], dataCount);
    }
    dataQueue.EnQue(dataLocal);
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline void ForeachIntToFloatUnary<T, op, bufferNum>::Compute(uint32_t index, int64_t dataCount,
                                                                         bool isRemainder)
{
    (void)index;
    (void)isRemainder;
    LocalTensor<T> dataLocal = dataQueue.template DeQue<T>();
    LocalTensor<float> outLocal = outQueue.template AllocTensor<float>();
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        // int8/uint8 经 half 跳板提升 float32, 在 float32 域复用注入的计算函数
        LocalTensor<half> halfTensor = halfBuf.GetWithOffset<half>(maxDataCount, 0);
        Cast(halfTensor, dataLocal, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(outLocal, halfTensor, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        op(outLocal, outLocal, static_cast<int32_t>(dataCount));
        PipeBarrier<PIPE_V>();
    } else {
        // int16 直接提升 float32 计算
        Cast(outLocal, dataLocal, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        op(outLocal, outLocal, static_cast<int32_t>(dataCount));
        PipeBarrier<PIPE_V>();
    }

    outQueue.EnQue(outLocal);
    dataQueue.FreeTensor(dataLocal);
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline void ForeachIntToFloatUnary<T, op, bufferNum>::CopyOut(uint32_t index, int64_t dataCount,
                                                                         bool isRemainder)
{
    LocalTensor<float> outLocal = outQueue.template DeQue<float>();
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(outTensorsGM[1ULL * index * maxDataCount], outLocal, copyParams);
    } else {
        DataCopy(outTensorsGM[1ULL * index * maxDataCount], outLocal, dataCount);
    }
    outQueue.FreeTensor(outLocal);
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline __gm__ T* ForeachIntToFloatUnary<T, op, bufferNum>::GetInputTensorAddr(uint16_t index,
                                                                                         GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

template <typename T, IntToFloatOp* op, int32_t bufferNum>
__aicore__ inline __gm__ float* ForeachIntToFloatUnary<T, op, bufferNum>::GetOutputTensorAddr(uint16_t index,
                                                                                              GM_ADDR tensorPtr)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ float*>(*(retPtr + index));
}

} // namespace OpKernel
} // namespace Common

#endif // FOREACH_INT_TO_FLOAT_UNARY_H
