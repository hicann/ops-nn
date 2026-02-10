/*
 * Copyright (c) 2025 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file silu_mul.h
 * \brief silu_mul head file
 */

#ifndef SILU_MUL_H
#define SILU_MUL_H

#include <type_traits>
#include "kernel_operator.h"

namespace SiluMul {

using namespace AscendC;

constexpr int32_t MAX_UB_SIZE = 184 * 1024;
constexpr int32_t ONE_BLOCK_SIZE = 32;

template <typename T>
class SiluMulND {
public:
    TPipe pipe;
    __aicore__ inline SiluMulND(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, const SiluMulTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BigTailProcess();
    __aicore__ inline void SmallTailProcess();
    __aicore__ inline void CopyIn(int64_t inputOffset, DataCopyExtParams dataCopyParams);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t outputOffset, int64_t dataCount, DataCopyExtParams dataCopyParams);
    __aicore__ inline void SmallTailCopyIn(
        int32_t gmOffset, uint32_t tmpCalNum, uint32_t dAlign, LocalTensor<T>& x1Pad, LocalTensor<T>& x2Pad);
    __aicore__ inline void SmallTailCompute(
        uint32_t calcLen, uint32_t maxN, uint32_t dAlign, LocalTensor<T>& x1Pad, LocalTensor<T>& x2Pad,
        LocalTensor<float>& floatBuf);
    __aicore__ inline void SmallTailCopyOut(
        int32_t gmOffset, uint32_t tmpCalNum, uint32_t dAlign, LocalTensor<T>& x1Pad);

private:
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<T> x1Tmp;
    LocalTensor<T> x2Tmp;

    LocalTensor<T> x1Tensor;
    LocalTensor<T> x2Tensor;

    LocalTensor<float> tempResTensor;

    LocalTensor<float> x1TensorFp32;
    LocalTensor<float> x2TensorFp32;

    GlobalTensor<T> inputGmX;
    GlobalTensor<T> inputGmY;
    GlobalTensor<T> outputGm;

    int64_t lastDimSize;
    int64_t batchSize;
    uint32_t d;

    int64_t PPMaxCalNum;

    uint32_t needCoreNumber;
    int32_t blockIdx;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void SiluMulND<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, const SiluMulTilingData* tilingData)
{
    inputGmX.SetGlobalBuffer((__gm__ T*)x);
    inputGmY.SetGlobalBuffer((__gm__ T*)y);
    outputGm.SetGlobalBuffer((__gm__ T*)z);

    batchSize = tilingData->batchSize;
    lastDimSize = tilingData->lastDimSize;
    needCoreNumber = tilingData->needCoreNum;
    PPMaxCalNum = tilingData->PPMaxCalNum;

    d = lastDimSize;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void SiluMulND<T>::Process()
{
    if (blockIdx >= needCoreNumber) {
        return;
    }
    if (d > PPMaxCalNum) {
        BigTailProcess();
    } else {
        SmallTailProcess();
    }
}

template <typename T>
__aicore__ inline void SiluMulND<T>::BigTailProcess()
{
    int32_t loopNum = batchSize / needCoreNumber;
    int32_t loopRemain = batchSize % needCoreNumber;
    if (loopRemain > 0 && blockIdx < loopRemain) {
        loopNum++;
    }
    for (int32_t i = 0; i < loopNum; i++) {
        int32_t totalOffset = i * needCoreNumber * lastDimSize + blockIdx * lastDimSize;
        int32_t outOffset = totalOffset;
        int32_t eachLineLoop = d / PPMaxCalNum;
        uint32_t remain = d % PPMaxCalNum;
        if (remain > 0) {
            eachLineLoop++;
        }
        pingPongFlag = 0;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int32_t j = 0; j < eachLineLoop; j++) {
            uint32_t dataCount = PPMaxCalNum;
            if (j == eachLineLoop - 1 && remain > 0) {
                dataCount = remain;
            }
            int32_t localOffset = j * PPMaxCalNum;
            eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
            DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
            CopyIn(totalOffset + localOffset, dataCopyParams);
            Compute(dataCount);
            CopyOut(outOffset + localOffset, dataCount, dataCopyParams);
            pingPongFlag = 1 - pingPongFlag;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

template <typename T>
__aicore__ inline void SiluMulND<T>::CopyIn(int64_t inputOffset, DataCopyExtParams dataCopyParams)
{
    x1Tensor = pingPongFlag ? tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() : tmpTensor[0].ReinterpretCast<T>();
    x2Tensor = pingPongFlag ? tmpTensor[PPMaxCalNum * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                              tmpTensor[PPMaxCalNum * sizeof(float)].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int32_t elementByte = PPMaxCalNum * sizeof(T);
        x1Tmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                               tmpTensor[elementByte].ReinterpretCast<T>();
        x2Tmp = pingPongFlag ?
                    tmpTensor[elementByte + PPMaxCalNum * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                    tmpTensor[elementByte + PPMaxCalNum * sizeof(float)].ReinterpretCast<T>();
        DataCopyPad(x1Tmp, inputGmX[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tmp, inputGmY[inputOffset], dataCopyParams, padParams);
    } else {
        DataCopyPad(x1Tensor, inputGmX[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tensor, inputGmY[inputOffset], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
}

template <typename T>
__aicore__ inline void SiluMulND<T>::Compute(int64_t dataCount)
{
    x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
    x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(x2TensorFp32, x2Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
    tempResTensor = pingPongFlag ?
                        tmpTensor[PPMaxCalNum * 2 * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<float>() :
                        tmpTensor[PPMaxCalNum * 2 * sizeof(float)].ReinterpretCast<float>();

    // Silu Calculation: x1 * Sigmoid(x1)
    Sigmoid(tempResTensor, x1TensorFp32, dataCount);

    PipeBarrier<PIPE_V>();

    Mul(x1TensorFp32, x1TensorFp32, tempResTensor, dataCount);

    PipeBarrier<PIPE_V>();

    // Silu * x2
    Mul(x1TensorFp32, x1TensorFp32, x2TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SiluMulND<T>::CopyOut(int64_t outputOffset, int64_t dataCount, DataCopyExtParams dataCopyParams)
{
    if (std::is_same_v<T, half>) {
        Cast(x1Tensor, x1TensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else if (std::is_same_v<T, bfloat16_t>) {
        Cast(x1Tensor, x1TensorFp32, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopyPad(outputGm[outputOffset], x1Tensor, dataCopyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}
template <typename T>
__aicore__ inline void SiluMulND<T>::SmallTailCopyIn(
    int32_t gmOffset, uint32_t tmpCalNum, uint32_t dAlign, LocalTensor<T>& x1Pad, LocalTensor<T>& x2Pad)
{
    for (int32_t j = 0; j < tmpCalNum; j++) {
        int32_t batchOffset = gmOffset + j * lastDimSize;
        for (int32_t k = 0; k < d; k++) {
            x1Pad.SetValue(j * dAlign + k, inputGmX.GetValue(batchOffset + k));
            x2Pad.SetValue(j * dAlign + k, inputGmY.GetValue(batchOffset + k));
        }
    }
}

template <typename T>
__aicore__ inline void SiluMulND<T>::SmallTailCompute(
    uint32_t calcLen, uint32_t maxN, uint32_t dAlign, LocalTensor<T>& x1Pad, LocalTensor<T>& x2Pad,
    LocalTensor<float>& floatBuf)
{
    LocalTensor<float> compRes = floatBuf;
    LocalTensor<float> compX1 = floatBuf[maxN * dAlign];
    LocalTensor<float> compX2 = floatBuf[2 * maxN * dAlign];

    if constexpr (std::is_same_v<T, float>) {
        Sigmoid(compRes, x1Pad, calcLen);
        PipeBarrier<PIPE_V>();
        Mul(x1Pad, x1Pad, compRes, calcLen);
        PipeBarrier<PIPE_V>();
        Mul(x1Pad, x1Pad, x2Pad, calcLen);
    } else {
        Cast(compX1, x1Pad, RoundMode::CAST_NONE, calcLen);
        Cast(compX2, x2Pad, RoundMode::CAST_NONE, calcLen);
        PipeBarrier<PIPE_V>();

        Sigmoid(compRes, compX1, calcLen);
        PipeBarrier<PIPE_V>();
        Mul(compX1, compX1, compRes, calcLen);
        PipeBarrier<PIPE_V>();
        Mul(compX1, compX1, compX2, calcLen);

        if constexpr (std::is_same_v<T, half>) {
            Cast(x1Pad, compX1, RoundMode::CAST_NONE, calcLen);
        } else {
            Cast(x1Pad, compX1, RoundMode::CAST_RINT, calcLen);
        }
    }
}

template <typename T>
__aicore__ inline void SiluMulND<T>::SmallTailCopyOut(
    int32_t gmOffset, uint32_t tmpCalNum, uint32_t dAlign, LocalTensor<T>& x1Pad)
{
    int32_t outOffset = gmOffset;
    for (int32_t j = 0; j < tmpCalNum; j++) {
        for (int32_t k = 0; k < d; k++) {
            T val = x1Pad.GetValue(j * dAlign + k);
            outputGm.SetValue(outOffset + j * d + k, val);
        }
    }
}

template <typename T>
__aicore__ inline void SiluMulND<T>::SmallTailProcess()
{
    uint32_t sizeOfT = sizeof(T);
    uint32_t oneBlockNum = 32 / sizeOfT;
    uint32_t dAlign = (d + oneBlockNum - 1) / oneBlockNum * oneBlockNum;

    uint32_t bytesPerRow = 2 * dAlign * sizeOfT + 3 * dAlign * sizeof(float);
    uint32_t n = (MAX_UB_SIZE) / bytesPerRow;
    if (n == 0)
        n = 1;
    if (n > PPMaxCalNum / dAlign)
        n = PPMaxCalNum / dAlign;

    int32_t eachCoreNum = batchSize / needCoreNumber;
    int32_t remain = batchSize % needCoreNumber;
    if (remain > 0 && blockIdx < remain)
        eachCoreNum++;

    int32_t loopNum = eachCoreNum / n;
    int32_t loopRemain = eachCoreNum % n;
    if (loopRemain > 0)
        loopNum++;

    int32_t totalOffset = eachCoreNum * blockIdx * lastDimSize;
    if (remain > 0) {
        totalOffset = (blockIdx < remain) ? (eachCoreNum * blockIdx * lastDimSize) :
                                            ((blockIdx * eachCoreNum + remain) * lastDimSize);
    }

    LocalTensor<T> x1Pad = tmpTensor[0].ReinterpretCast<T>();
    LocalTensor<T> x2Pad = tmpTensor[n * dAlign * sizeOfT].ReinterpretCast<T>();
    LocalTensor<float> floatBuf = tmpTensor[2 * n * dAlign * sizeOfT].ReinterpretCast<float>();

    for (int32_t i = 0; i < loopNum; i++) {
        uint32_t tmpCalNum = (loopRemain > 0 && i == loopNum - 1) ? loopRemain : n;
        int32_t gmOffset = totalOffset + i * n * lastDimSize;

        SmallTailCopyIn(gmOffset, tmpCalNum, dAlign, x1Pad, x2Pad);

        PipeBarrier<PIPE_ALL>();

        SmallTailCompute(tmpCalNum * dAlign, n, dAlign, x1Pad, x2Pad, floatBuf);

        PipeBarrier<PIPE_ALL>();

        SmallTailCopyOut(gmOffset, tmpCalNum, dAlign, x1Pad);

        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace SiluMul
#endif