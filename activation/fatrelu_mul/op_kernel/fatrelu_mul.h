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
 * \file fatrelu_mul.h
 * \brief fatrelu_mul head file
 */

#ifndef FATRELU_MUL_H
#define FATRELU_MUL_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace FatreluMul {

using namespace AscendC;

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
constexpr int32_t MAX_UB_SIZE = 120 * 1024;
constexpr int32_t PP_ELEMENT_NUM = 6 * 1024;
#else
constexpr int32_t MAX_UB_SIZE = 192 * 1024;
constexpr int32_t PP_ELEMENT_NUM = 10 * 1024;
#endif

constexpr int32_t ONE_REPEAT_ELE_NUM_FP32 = 64;
constexpr int32_t ONE_REPEAT_ELE_NUM_FP16 = 128;
constexpr int32_t ONE_BLOCK_SIZE = 32;

template <typename T>
class FatreluMulND
{
public:
    TPipe pipe;
    __aicore__ inline FatreluMulND(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR scalar, GM_ADDR output, GM_ADDR workspace, const FatreluMulTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BigTailProcess();
    __aicore__ inline void SmallTailProcess();
    __aicore__ inline void CopyIn(int64_t inputOffset, DataCopyExtParams dataCopyParams);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t outputOffset, int64_t dataCount, DataCopyExtParams dataCopyParams);

private:
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<uint8_t> selMask;

    LocalTensor<T> x1Tmp;
    LocalTensor<T> x2Tmp;

    LocalTensor<T> x1Tensor;
    LocalTensor<T> x2Tensor;

    LocalTensor<float> x1TensorFp32;
    LocalTensor<float> x2TensorFp32;

    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> inScalarGM;

    T threshold;
    int64_t lastDimSize;
    int64_t batchSize;
    uint32_t d;

    uint32_t needCoreNumber;
    int32_t blockIdx;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;
};

template <typename T>
__aicore__ inline void FatreluMulND<T>::Init(
    GM_ADDR input, GM_ADDR scalar, GM_ADDR output, GM_ADDR workspace, const FatreluMulTilingData* tilingData)
{
    inputGm.SetGlobalBuffer((__gm__ T*)input);
    inScalarGM.SetGlobalBuffer((__gm__ T*)scalar, 1);
    outputGm.SetGlobalBuffer((__gm__ T*)output);
    threshold = T(inScalarGM.GetValue(0));
    batchSize = tilingData->batchSize;
    lastDimSize = tilingData->lastDimSize;
    needCoreNumber = tilingData->needCoreNum;

    d = lastDimSize / 2;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, MAX_UB_SIZE);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void FatreluMulND<T>::Process()
{
    if (blockIdx >= needCoreNumber) {
        return;
    }
    if (d > PP_ELEMENT_NUM) {
        BigTailProcess();
    } else {
        SmallTailProcess();
    }
}

template <typename T>
__aicore__ inline void FatreluMulND<T>::BigTailProcess()
{
    int32_t loopNum = batchSize / needCoreNumber;
    int32_t loopRemain = batchSize % needCoreNumber;
    if (loopRemain > 0 && blockIdx < loopRemain) {
        loopNum++;
    }
    for (int32_t i = 0; i < loopNum; i++) {
        int32_t totalOffset = i * needCoreNumber * lastDimSize + blockIdx * lastDimSize;
        int32_t outOffset = i * needCoreNumber * d + blockIdx * d;
        int32_t eachLineLoop = d / PP_ELEMENT_NUM;
        uint32_t remain = d % PP_ELEMENT_NUM;
        if (remain > 0) {
            eachLineLoop++;
        }
        pingPongFlag = 0;
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int32_t j = 0; j < eachLineLoop; j++) {
            uint32_t dataCount = PP_ELEMENT_NUM;
            if (j == eachLineLoop - 1 && remain > 0) {
                dataCount = remain;
            }
            int32_t localOffset = j * PP_ELEMENT_NUM;
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
__aicore__ inline void FatreluMulND<T>::CopyIn(int64_t inputOffset, DataCopyExtParams dataCopyParams)
{
    x1Tensor = pingPongFlag ? tmpTensor[MAX_UB_SIZE / 2].ReinterpretCast<T>() : tmpTensor[0].ReinterpretCast<T>();
    x2Tensor = pingPongFlag ? tmpTensor[PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                              tmpTensor[PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        int32_t elementByte = PP_ELEMENT_NUM * sizeof(T);
        x1Tmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                               tmpTensor[elementByte].ReinterpretCast<T>();
        x2Tmp = pingPongFlag ?
                    tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                    tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
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
__aicore__ inline void FatreluMulND<T>::Compute(int64_t dataCount)
{
    x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
    x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();
    if (std::is_same<T, half>::value) {
        Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(x2TensorFp32, x2Tmp, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
    selMask = pingPongFlag ? tmpTensor[PP_ELEMENT_NUM * sizeof(float) * 2 + MAX_UB_SIZE / 2] :
                             tmpTensor[PP_ELEMENT_NUM * sizeof(float) * 2];

    auto tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;

    CompareScalar(selMask, x1TensorFp32, (float)threshold, CMPMODE::GT, tmpDataCount);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMask, x1TensorFp32, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();

    Mul(x1TensorFp32, x1TensorFp32, x2TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void FatreluMulND<bfloat16_t>::Compute(int64_t dataCount)
{
    x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
    x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();

    Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
    Cast(x2TensorFp32, x2Tmp, RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();

    selMask = pingPongFlag ? tmpTensor[PP_ELEMENT_NUM * sizeof(float) * 2 + MAX_UB_SIZE / 2] :
                             tmpTensor[PP_ELEMENT_NUM * sizeof(float) * 2];

    auto tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;

    float tmpThreshold = ToFloat(threshold);
    CompareScalar(selMask, x1TensorFp32, tmpThreshold, CMPMODE::GT, tmpDataCount);
    PipeBarrier<PIPE_V>();

    Select(x1TensorFp32, selMask, x1TensorFp32, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();

    Mul(x1TensorFp32, x1TensorFp32, x2TensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void FatreluMulND<T>::CopyOut(
    int64_t outputOffset, int64_t dataCount, DataCopyExtParams dataCopyParams)
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
__aicore__ inline void FatreluMulND<T>::SmallTailProcess()
{
    auto oneBlockNum = ONE_BLOCK_SIZE / sizeof(T);
    auto dAlign = (d + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
    int32_t n = PP_ELEMENT_NUM / dAlign;

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
        DataCopyExtParams dataCopyParamsIn{
            tmpCalNum, static_cast<uint32_t>(d * sizeof(T)), static_cast<uint32_t>(d * sizeof(T)), 0, 0};

        CopyIn(totalOffset + localOffset, dataCopyParamsIn);
        Compute(tmpCalNum * dAlign);
        DataCopyExtParams dataCopyParamsOut{tmpCalNum, static_cast<uint32_t>(d * sizeof(T)), 0, 0, 0};
        CopyOut((totalOffset + localOffset) / 2, tmpCalNum * dAlign, dataCopyParamsOut);
        pingPongFlag = 1 - pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}

} // namespace FatreluMul
#endif