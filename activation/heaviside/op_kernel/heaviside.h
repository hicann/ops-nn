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
 * \file heaviside.h
 * \brief heaviside head file
 */
#ifndef HEAVISIDE_H
#define HEAVISIDE_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace Heaviside {

using namespace AscendC;

constexpr int32_t MAX_UB_SIZE = 192 * 1024;
constexpr int32_t PP_ELEMENT_NUM = 8 * 1024;
constexpr int32_t ONE_REPEAT_BYTES = 256;

template <typename T>
class HeavisideND
{
public:
    TPipe pipe;
    __aicore__ inline HeavisideND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR values, GM_ADDR output, GM_ADDR workspace, const HeavisideTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInAndCast(int64_t inputOffset, int64_t dataCount);
    __aicore__ inline void Compute(int64_t dataCount);
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

    LocalTensor<float> x1TensorCast;
    LocalTensor<float> x2TensorCast;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> valuesGm;
    GlobalTensor<T> outputGm;

    int64_t elementNum;
    uint32_t needCoreNumber;
    int32_t blockIdx;
    bool valuesType;
    T values;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void HeavisideND<T>::Init(
    GM_ADDR input, GM_ADDR values, GM_ADDR output, GM_ADDR workspace, const HeavisideTilingData* tilingData)
{
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    valuesGm.SetGlobalBuffer((__gm__ T*)values);
    outputGm.SetGlobalBuffer((__gm__ T*)output);

    valuesType = tilingData->valuesType;
    elementNum = tilingData->elementNum;
    needCoreNumber = tilingData->needCoreNum;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void HeavisideND<T>::Process()
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

    int32_t calNum = PP_ELEMENT_NUM;
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

        Compute(calNum);

        CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);
        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <typename T>
__aicore__ inline void HeavisideND<T>::CopyInAndCast(int64_t inputOffset, int64_t dataCount)
{
    int32_t elementByte = PP_ELEMENT_NUM * sizeof(float);
    x1Tensor = pingPongFlag ? tmpTensor[elementByte * 2].ReinterpretCast<T>() : tmpTensor[0].ReinterpretCast<T>();
    x2Tensor =
        pingPongFlag ? tmpTensor[elementByte * 3].ReinterpretCast<T>() : tmpTensor[elementByte].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t>) {
        int32_t bf16Byte = PP_ELEMENT_NUM * sizeof(T);
        x1Tmp = pingPongFlag ? tmpTensor[bf16Byte + elementByte * 2].ReinterpretCast<T>() :
                               tmpTensor[bf16Byte].ReinterpretCast<T>();
        x2Tmp = pingPongFlag ? tmpTensor[bf16Byte + elementByte * 3].ReinterpretCast<T>() :
                               tmpTensor[bf16Byte + elementByte].ReinterpretCast<T>();
        DataCopyPad(x1Tmp, inputGm[inputOffset], dataCopyParams, padParams);
        if (valuesType == false) {
            DataCopyPad(x2Tmp, valuesGm[inputOffset], dataCopyParams, padParams);
        } else {
            values = T(valuesGm.GetValue(0));
        }

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        x1TensorCast = x1Tensor.template ReinterpretCast<float>();
        x2TensorCast = x2Tensor.template ReinterpretCast<float>();
        Cast(x1TensorCast, x1Tmp, RoundMode::CAST_NONE, dataCount);
        if (valuesType == false) {
            Cast(x2TensorCast, x2Tmp, RoundMode::CAST_NONE, dataCount);
        }
        PipeBarrier<PIPE_V>();
    } else {
        DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
        if (valuesType == false) {
            DataCopyPad(x2Tensor, valuesGm[inputOffset], dataCopyParams, padParams);
        } else {
            values = T(valuesGm.GetValue(0));
        }

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }
}

template <typename T>
__aicore__ inline void HeavisideND<T>::Compute(int64_t dataCount)
{
    int32_t maskStartByte = PP_ELEMENT_NUM * sizeof(float) * 4;
    selMaskOne =
        pingPongFlag ? tmpTensor[maskStartByte + PP_ELEMENT_NUM * sizeof(uint8_t) * 2] : tmpTensor[maskStartByte];
    selMaskTwo = pingPongFlag ? tmpTensor[maskStartByte + PP_ELEMENT_NUM * sizeof(uint8_t) * 3] :
                                tmpTensor[maskStartByte + PP_ELEMENT_NUM * sizeof(uint8_t)];

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        auto oneRepeatEleNum = ONE_REPEAT_BYTES / sizeof(float);
        auto tmpDataCount = (dataCount + oneRepeatEleNum - 1) / oneRepeatEleNum * oneRepeatEleNum;

        // NAN eq NAN is false, 去除NAN
        Compare(selMaskOne, x1TensorCast, x1TensorCast, CMPMODE::EQ, tmpDataCount);
        PipeBarrier<PIPE_V>();
        Select(x1TensorCast, selMaskOne, x1TensorCast, (float)-1, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        PipeBarrier<PIPE_V>();

        // 复用selMaskOne，标记0元素用于取对应位置values值
        CompareScalar(selMaskOne, x1TensorCast, (float)0, CMPMODE::NE, tmpDataCount);
        CompareScalar(selMaskTwo, x1TensorCast, (float)0, CMPMODE::LE, tmpDataCount);
        PipeBarrier<PIPE_V>();

        Maxs(x1TensorCast, x1TensorCast, (float)0, tmpDataCount);
        PipeBarrier<PIPE_V>();

        Select(x1TensorCast, selMaskTwo, x1TensorCast, (float)1, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        PipeBarrier<PIPE_V>();

        if (valuesType == false) {
            Select(x1TensorCast, selMaskOne, x1TensorCast, x2TensorCast, SELMODE::VSEL_TENSOR_TENSOR_MODE, dataCount);
        } else {
            Select(
                x1TensorCast, selMaskOne, x1TensorCast, ToFloat(values), SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        }
        PipeBarrier<PIPE_V>();
    } else {
        auto oneRepeatEleNum = ONE_REPEAT_BYTES / sizeof(T);
        auto tmpDataCount = (dataCount + oneRepeatEleNum - 1) / oneRepeatEleNum * oneRepeatEleNum;

        // NAN eq NAN is false, 去除NAN
        Compare(selMaskOne, x1Tensor, x1Tensor, CMPMODE::EQ, tmpDataCount);
        PipeBarrier<PIPE_V>();
        Select(x1Tensor, selMaskOne, x1Tensor, (T)-1, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        PipeBarrier<PIPE_V>();

        // 复用selMaskOne，标记0元素位置
        CompareScalar(selMaskOne, x1Tensor, (T)0, CMPMODE::NE, tmpDataCount);
        CompareScalar(selMaskTwo, x1Tensor, (T)0, CMPMODE::LE, tmpDataCount);
        PipeBarrier<PIPE_V>();

        Maxs(x1Tensor, x1Tensor, (T)0, tmpDataCount);
        PipeBarrier<PIPE_V>();

        Select(x1Tensor, selMaskTwo, x1Tensor, (T)1, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        PipeBarrier<PIPE_V>();

        if (valuesType == false) {
            Select(x1Tensor, selMaskOne, x1Tensor, x2Tensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, dataCount);
        } else {
            Select(x1Tensor, selMaskOne, x1Tensor, values, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void HeavisideND<T>::CastAndCopyOut(int64_t outputOffset, int64_t dataCount)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        Cast(x1Tensor, x1TensorCast, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(outputGm[outputOffset], x1Tensor, dataCopyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}
} // namespace Heaviside
#endif