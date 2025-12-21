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
 * \file hard_swish_grad_v2_220.h
 * \brief hard_swish_grad_v2_220 head file
 */
 
#ifndef HARD_SWISH_GRAD_V2_220_H
#define HARD_SWISH_GRAD_V2_220_H

#include "hard_swish_grad_v2_base.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hard_swish_grad_v2_base.h"

namespace HardSwishGradV2 {

using namespace AscendC;

template <typename T>
class HardSwishGradV2220 : public HardSwishGradV2Base<T>{
public:
    __aicore__ inline HardSwishGradV2220(){};

protected:
__aicore__ inline void CopyInAndCast (int64_t inputOffset, int64_t dataCount) {
    this->gradTensor = this->pingPongFlag ? this->tmpTensor[this->ubSize / NUM_TWO].template ReinterpretCast<T>() : this->tmpTensor[0].template ReinterpretCast<T>();
    this->selfTensor = this->pingPongFlag ? this->tmpTensor[this->elementNumEachCore * sizeof(float) + this->ubSize / NUM_TWO].template ReinterpretCast<T>() :
                              this->tmpTensor[this->elementNumEachCore * sizeof(float)].template ReinterpretCast<T>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        DataCopyPad(this->gradTensor[this->elementNumEachCore], this->gradGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(this->selfTensor[this->elementNumEachCore], this->selfGm[inputOffset], dataCopyParams, padParams);
    } else {
        DataCopyPad(this->gradTensor, this->gradGm[inputOffset], dataCopyParams, padParams);
        DataCopyPad(this->selfTensor, this->selfGm[inputOffset], dataCopyParams, padParams);
    }

    SetFlag<HardEvent::MTE2_V>(this->eventId);
    WaitFlag<HardEvent::MTE2_V>(this->eventId);

    this->gradTensorFp32 = this->gradTensor.template ReinterpretCast<float>();
    this->selfTensorFp32 = this->selfTensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        Cast(this->gradTensorFp32, this->gradTensor[this->elementNumEachCore], RoundMode::CAST_NONE, dataCount);
        Cast(this->selfTensorFp32, this->selfTensor[this->elementNumEachCore], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void Compute(int64_t dataCount) {
    int64_t tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;

    int64_t elementByte = this->elementNumEachCore * sizeof(float);
    this->maskGreater = this->pingPongFlag ? this->tmpTensor[elementByte * NUM_TWO + this->ubSize / NUM_TWO] : this->tmpTensor[elementByte * NUM_TWO];
    this->maskLessThan = this->pingPongFlag ? this->tmpTensor[elementByte * NUM_TWO + elementByte / NUM_TWO + this->ubSize / NUM_TWO] :
                                this->tmpTensor[elementByte * NUM_TWO + elementByte / NUM_TWO];

    // step1
    CompareScalar(this->maskGreater, this->selfTensorFp32, this->negative, CMPMODE::GT, tmpDataCount);
    PipeBarrier<PIPE_V>();

    CompareScalar(this->maskLessThan, this->selfTensorFp32, this->positive , CMPMODE::LT, tmpDataCount);
    PipeBarrier<PIPE_V>();

    // step2
    Muls(this->selfTensorFp32, this->selfTensorFp32, this->oneThird , dataCount);
    PipeBarrier<PIPE_V>();

    Adds(this->selfTensorFp32, this->selfTensorFp32, this->oneHalf, dataCount);
    PipeBarrier<PIPE_V>();

    // step3
    Select(this->selfTensorFp32, this->maskGreater, this->selfTensorFp32, this->zero, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();

    Select(this->selfTensorFp32, this->maskLessThan, this->selfTensorFp32, this->one, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataCount);
    PipeBarrier<PIPE_V>();

    // step4
    Mul(this->selfTensorFp32, this->gradTensorFp32, this->selfTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CastAndCopyOut (int64_t outputOffset, int64_t dataCount) {
    if (std::is_same_v<T, half>) {
        Cast(this->selfTensor, this->selfTensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    } else if (std::is_same_v<T, bfloat16_t>) {
        Cast(this->selfTensor, this->selfTensorFp32, RoundMode::CAST_RINT, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(this->eventId);
    WaitFlag<HardEvent::V_MTE3>(this->eventId);
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->outGm[outputOffset], this->selfTensor, dataCopyParams);
}

__aicore__ inline void ProcessAlign(int64_t inputOffset, int64_t dataCount) {
    CopyInAndCast(inputOffset, dataCount);
    Compute(dataCount);
    CastAndCopyOut(inputOffset, dataCount);
}

public:
__aicore__ inline void Process() {
    if (this->blockIdx >= this->needCoreNumber) {
        return;
    }
    
    int64_t totalTime = this->elementNum / this->elementNumEachCore;
    int64_t remain = this->elementNum % this->elementNumEachCore;
    if (remain > 0) {
        totalTime++;
    }
    int64_t loopNumber = totalTime / this->needCoreNumber;
    int64_t loopRemain = totalTime % this->needCoreNumber;
    
    if (loopRemain > 0 && this->blockIdx < loopRemain) {
        loopNumber++;
    }
    int64_t eachCoreStartOffset = loopNumber * this->blockIdx * this->elementNumEachCore;
    if (loopRemain > 0) {
        if (this->blockIdx >= loopRemain) {
            eachCoreStartOffset += this->elementNum % (this->elementNumEachCore * this->needCoreNumber);
        }
    }
    
    int64_t calNumber = this->elementNumEachCore;
    int64_t lastCoreNum = loopRemain == 0 ? this->needCoreNumber - 1 : loopRemain - 1;
    this->pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < loopNumber; i++) {
        this->eventId = this->pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        WaitFlag<HardEvent::MTE3_MTE2>(this->eventId);

        int64_t localOffset = i * this->elementNumEachCore;
        if (remain > 0 && i == loopNumber -1 && this->blockIdx == lastCoreNum) {
            calNumber = remain;
        }
        
        ProcessAlign(eachCoreStartOffset + localOffset, calNumber);

        SetFlag<HardEvent::MTE3_MTE2>(this->eventId);
        this->pingPongFlag = 1 - this->pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}
};
}// namespace HardSwishGradV2
#endif