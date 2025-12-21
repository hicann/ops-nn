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
 * \file logit.h
 * \brief
 */

#ifndef LOGIT_H
#define LOGIT_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace Logit {

using namespace AscendC;

constexpr int64_t MAX_UB_SIZE = 192 * 1024;
constexpr int64_t PP_ELEMENT_NUM = 8 * 1024;
constexpr int64_t ONE_REPEAT_ELE_NUM_FP32 = 64;

template <typename T>
class LogitND
{
public:
    TPipe pipe;
    __aicore__ inline LogitND(){};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const LogitTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndCast(int64_t inputOffset, int64_t dataCount);
    __aicore__ inline void ComputeStepOne(int64_t dataCount);
    __aicore__ inline void ComputeStepTwo(int64_t dataCount);
    __aicore__ inline void CastAndCopyOut(int64_t outputOffset, int64_t dataCount);

private:
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<uint8_t> selMaskOne;
    LocalTensor<uint8_t> selMaskTwo;

    LocalTensor<T> x1Tmp;
    LocalTensor<T> x2Tmp;

    LocalTensor<T> x1Tensor;
    LocalTensor<T> x2Tensor;

    LocalTensor<float> x1TensorFp32;
    LocalTensor<float> x2TensorFp32;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    int64_t elementNum;
    uint64_t needCoreNumber;
    int64_t blockIdx;
    float eps;

    event_t eventId = EVENT_ID0;
    int64_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void LogitND<T>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, const LogitTilingData* tilingData)
{
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    outputGm.SetGlobalBuffer((__gm__ T*)output);

    eps = tilingData->eps;
    elementNum = tilingData->elementNum;
    needCoreNumber = tilingData->needCoreNum;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void LogitND<T>::Process()
{
    if (blockIdx >= needCoreNumber) {
        return;
    }

    int64_t totalTimes = elementNum / PP_ELEMENT_NUM;
    int64_t remain = elementNum % PP_ELEMENT_NUM;
    if (remain > 0) {
        totalTimes++;
    }
    int64_t loopNum = totalTimes / needCoreNumber;
    int64_t loopRemain = totalTimes % needCoreNumber;

    if (loopRemain > 0 && blockIdx < loopRemain) {
        loopNum++;
    }
    int64_t eachCoreStartOffset = loopNum * blockIdx * PP_ELEMENT_NUM;
    if (loopRemain > 0) {
        if (blockIdx >= loopRemain) {
            eachCoreStartOffset += elementNum % (PP_ELEMENT_NUM * needCoreNumber);
        }
    }

    int64_t calNum = PP_ELEMENT_NUM;
    int64_t lastCoreNum = loopRemain == 0 ? needCoreNumber - 1 : loopRemain - 1;
    pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t localOffset = i * PP_ELEMENT_NUM;

        // 最后一轮的最后一个核处理余数
        if (remain > 0 && i == loopNum - 1 && blockIdx == lastCoreNum) {
            calNum = remain;
        }
        eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        CopyInAndCast(eachCoreStartOffset + localOffset, calNum);

        if (eps >= 0) {
            ComputeStepOne(calNum);
        }
        ComputeStepTwo(calNum);
        CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);
        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void LogitND<T>::CopyInAndCast(int64_t inputOffset, int64_t dataCount)
{
    x1Tensor = pingPongFlag ? tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() : tmpTensor[0].ReinterpretCast<T>();
    x2Tensor = pingPongFlag ? tmpTensor[PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                              tmpTensor[PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int64_t elementByte = PP_ELEMENT_NUM * sizeof(T);
        x1Tmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                               tmpTensor[elementByte].ReinterpretCast<T>();
        x2Tmp = pingPongFlag ?
                    tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                    tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
        DataCopyPad(x1Tmp, inputGm[inputOffset], dataCopyParams, padParams);

    } else {
        DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
    x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void LogitND<T>::ComputeStepOne(int64_t dataCount)
{
    // 对x用eps进行处理
    int64_t elementByte = PP_ELEMENT_NUM * sizeof(float);
    selMaskOne = pingPongFlag ? tmpTensor[elementByte * 2 + MAX_UB_SIZE / 2] : tmpTensor[elementByte * 2];
    selMaskTwo = pingPongFlag ? tmpTensor[elementByte * 2 + elementByte / 2 + MAX_UB_SIZE / 2] :
                                tmpTensor[elementByte * 2 + elementByte / 2];

    float hi = static_cast<float>(1.0) - eps;
    auto tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;
    CompareScalar(selMaskOne, x1TensorFp32, (float)eps, CMPMODE::GE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    CompareScalar(selMaskTwo, x1TensorFp32, (float)hi, CMPMODE::LE, tmpDataCount);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMaskTwo, x1TensorFp32, (float)hi, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMaskOne, x1TensorFp32, (float)eps, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LogitND<T>::ComputeStepTwo(int64_t dataCount)
{
    Muls(x2TensorFp32, x1TensorFp32, float(-1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Adds(x2TensorFp32, x2TensorFp32, float(1.0), dataCount);
    PipeBarrier<PIPE_V>();
    Div(x1TensorFp32, x1TensorFp32, x2TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
    Ln(x1TensorFp32, x1TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LogitND<T>::CastAndCopyOut(int64_t outputOffset, int64_t dataCount)
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
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[outputOffset], x1Tensor, dataCopyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

} // namespace Logit
#endif