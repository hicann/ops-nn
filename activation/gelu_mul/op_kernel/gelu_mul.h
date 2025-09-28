/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gelu_mul.h
 * \brief gelu_mul head file
 */

#ifndef GELU_MUL_H
#define GELU_MUL_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace GeluMul {

using namespace AscendC;

constexpr int32_t MAX_UB_SIZE = 192 * 1024;
constexpr int32_t ONE_REPEAT_ELE_NUM_FP32 = 64;
constexpr int32_t ONE_BLOCK_SIZE = 32;

// Gelu 计算参数
constexpr float SCALAR_ONE = 1.0f;
constexpr float BETA = 0.044715f;
constexpr float ALPHA = -1.5957691f;

constexpr float ERF_PARAM1 = -0.3512339572e-8f;
constexpr float ERF_PARAM2 = 0.2645266170e-6f;
constexpr float ERF_PARAM3 = -0.7929488134e-5f;
constexpr float ERF_PARAM4 = 0.1106123840e-3f;
constexpr float ERF_PARAM5 = 0.6518995814e-4f;
constexpr float ERF_PARAM6 = -0.7266616915e-1f;
constexpr float ERF_PARAM7 = -0.1595769883e1f;
constexpr float ERF_MIN = 5.751f;
constexpr float ERF_MAX = -13.15f;
constexpr float MAX_INT8 = 127.0f;

template <typename T>
class GeluMulND {
public:
    TPipe pipe;
    __aicore__ inline GeluMulND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                const GeluMulTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BigTailProcess();
    __aicore__ inline void SmallTailProcess();
    __aicore__ inline void CopyIn(int64_t inputOffset, DataCopyExtParams dataCopyParams);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void ComputeGeluErf(LocalTensor<float> &castFp32, LocalTensor<float> &tempRes,
                                          LocalTensor<float> &xSquared, int64_t calCount);
    __aicore__ inline void ComputeGeluTanh(LocalTensor<float> &castFp32, LocalTensor<float> &tempRes,
                                           int64_t calCount);
    __aicore__ inline void CopyOut(int64_t outputOffset, int64_t dataCount, DataCopyExtParams dataCopyParams);

private:

    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<T> x1Tmp;
    LocalTensor<T> x2Tmp;

    LocalTensor<T> x1Tensor;
    LocalTensor<T> x2Tensor;

    LocalTensor<float> tempResTensor;   // Gelu计算tanh模式或者erf模式均用到的临时空间
    LocalTensor<float> xSquaredTensor;   // Gelu计算erf模式用到的临时空间

    LocalTensor<float> x1TensorFp32;
    LocalTensor<float> x2TensorFp32;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<float> inScalarGM;

    int64_t lastDimSize;
    int64_t batchSize;
    uint32_t d;

    int64_t approximateMode;
    int64_t PPMaxCalNum;

    uint32_t needCoreNumber;
    int32_t blockIdx;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void GeluMulND<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                              const GeluMulTilingData* tilingData) {
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);

    batchSize = tilingData->batchSize;
    lastDimSize = tilingData->lastDimSize;
    needCoreNumber = tilingData->needCoreNum;
    approximateMode = tilingData->approximateMode;
    PPMaxCalNum = tilingData->PPMaxCalNum;

    d = lastDimSize / 2;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void GeluMulND<T>::Process() {
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
__aicore__ inline void GeluMulND<T>::BigTailProcess() {
    int32_t loopNum = batchSize / needCoreNumber;
    int32_t loopRemain = batchSize % needCoreNumber;
    if (loopRemain > 0 && blockIdx < loopRemain) {
        loopNum++;
    }
    for (int32_t i = 0; i < loopNum; i++) {
        int32_t totalOffset = i * needCoreNumber * lastDimSize + blockIdx * lastDimSize;
        int32_t outOffset = i * needCoreNumber * d + blockIdx * d;
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
            if (j == eachLineLoop -1 && remain > 0) {
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
__aicore__ inline void GeluMulND<T>::CopyIn(int64_t inputOffset, DataCopyExtParams dataCopyParams) {

    x1Tensor = pingPongFlag ?
            tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() :
            tmpTensor[0].ReinterpretCast<T>();
    x2Tensor = pingPongFlag ?
            tmpTensor[PPMaxCalNum * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
            tmpTensor[PPMaxCalNum * sizeof(float)].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int32_t elementByte = PPMaxCalNum * sizeof(T);
        x1Tmp = pingPongFlag ?
                tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                tmpTensor[elementByte].ReinterpretCast<T>();
        x2Tmp = pingPongFlag ?
                tmpTensor[elementByte + PPMaxCalNum * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                tmpTensor[elementByte + PPMaxCalNum * sizeof(float)].ReinterpretCast<T>();
        DataCopyPad(x1Tmp, inputGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tmp, inputGm[inputOffset + d], dataCopyParams, padParams);
    } else {
        DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(x2Tensor, inputGm[inputOffset + d], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
}

template <typename T>
__aicore__ inline void GeluMulND<T>::Compute(int64_t dataCount) {
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

    if (approximateMode == 0) {
        xSquaredTensor = pingPongFlag ?
            tmpTensor[PPMaxCalNum * 3 * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<float>() :
            tmpTensor[PPMaxCalNum * 3 * sizeof(float)].ReinterpretCast<float>();

        ComputeGeluErf(x1TensorFp32, tempResTensor, xSquaredTensor, dataCount);

    } else if (approximateMode == 1){
        ComputeGeluTanh(x1TensorFp32, tempResTensor, dataCount);
    }

    Mul(x1TensorFp32, x1TensorFp32, x2TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GeluMulND<T>::ComputeGeluErf(LocalTensor<float> &castFp32, LocalTensor<float> &tempRes,
                                        LocalTensor<float> &xSquared, int64_t calCount)
{
    Maxs(castFp32, castFp32, ERF_MAX, calCount);
    PipeBarrier<PIPE_V>();
    Mins(tempRes, castFp32, ERF_MIN, calCount);
    PipeBarrier<PIPE_V>();

    Mul(xSquared, tempRes, tempRes, calCount);
    PipeBarrier<PIPE_V>();

    Muls(tempRes, xSquared, ERF_PARAM1, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM2, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM3, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM4, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM5, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM6, calCount);
    PipeBarrier<PIPE_V>();

    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, ERF_PARAM7, calCount);
    PipeBarrier<PIPE_V>();

    Mins(xSquared, castFp32, ERF_MIN, calCount);
    PipeBarrier<PIPE_V>();
    Mul(tempRes, tempRes, xSquared, calCount);
    PipeBarrier<PIPE_V>();

    Exp(tempRes, tempRes, calCount);
    PipeBarrier<PIPE_V>();

    Adds(tempRes, tempRes, 1.0f, calCount);
    PipeBarrier<PIPE_V>();

    Div(castFp32, castFp32, tempRes, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GeluMulND<T>::ComputeGeluTanh(LocalTensor<float> &castFp32, LocalTensor<float> &tempRes,
                                           int64_t calCount)
{
    Mul(tempRes, castFp32, castFp32, calCount); // x^2
    PipeBarrier<PIPE_V>();
    Mul(tempRes, castFp32, tempRes, calCount); // x^3
    PipeBarrier<PIPE_V>();
    Muls(tempRes, tempRes, BETA, calCount); // 0.044715 * x^3
    PipeBarrier<PIPE_V>();
    Add(tempRes, castFp32, tempRes, calCount); // x + 0.044715 * x^3
    PipeBarrier<PIPE_V>();
    Muls(tempRes, tempRes, ALPHA, calCount); // -sqrt(8/pi)(x + 0.044715 * x^3)
    PipeBarrier<PIPE_V>();
    Exp(tempRes, tempRes, calCount); // exp(-sqrt(8/pi)(x + 0.044715 * x^3))
    PipeBarrier<PIPE_V>();
    Adds(tempRes, tempRes, SCALAR_ONE, calCount); // 1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3))
    PipeBarrier<PIPE_V>();
    Div(castFp32, castFp32, tempRes, calCount); // x / (1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3)))
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GeluMulND<T>::CopyOut(int64_t outputOffset, int64_t dataCount,
                                                 DataCopyExtParams dataCopyParams) {
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
__aicore__ inline void GeluMulND<T>::SmallTailProcess() {

    auto oneBlockNum = ONE_BLOCK_SIZE / sizeof(T);
    auto dAlign = (d + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
    int32_t n = PPMaxCalNum / dAlign;

    int32_t eachCoreNum = batchSize / needCoreNumber;

    int32_t remain = batchSize % needCoreNumber;

    if (remain > 0 && blockIdx < remain) {
        eachCoreNum++;
    }
    int32_t loopNum = eachCoreNum / n;
    int32_t loopRemain = eachCoreNum % n;


    if (loopRemain > 0) {
        loopNum++;
    }

    int32_t totalOffset = eachCoreNum * blockIdx * lastDimSize;
    if (remain > 0) {
        if (blockIdx < remain) {
            totalOffset = (eachCoreNum * blockIdx) * lastDimSize;
        } else {
            totalOffset = (blockIdx * eachCoreNum + remain) * lastDimSize;
        }
    }
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t i = 0; i < loopNum; i++) {
        uint16_t tmpCalNum = n;
        if (loopRemain > 0 && i == loopNum - 1) {
            tmpCalNum = loopRemain;
        }

        int32_t localOffset = i * n * lastDimSize;
        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        DataCopyExtParams dataCopyParamsIn{tmpCalNum, static_cast<uint32_t>(d * sizeof(T)),
                                           static_cast<uint32_t>(d * sizeof(T)), 0, 0};

        CopyIn(totalOffset + localOffset, dataCopyParamsIn);
        Compute(tmpCalNum * dAlign);
        DataCopyExtParams dataCopyParamsOut{tmpCalNum, static_cast<uint32_t>(d * sizeof(T)),
                                            0, 0, 0};
        CopyOut((totalOffset + localOffset) / 2, tmpCalNum * dAlign, dataCopyParamsOut);
        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

}
#endif