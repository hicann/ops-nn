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
 * \file hard_swish_grad_v2_100.h
 * \brief hard_swish_grad_v2_100 head file
 */
 
#ifndef HARD_SWISH_GRAD_V2_100_H
#define HARD_SWISH_GRAD_V2_100_H

#include "hard_swish_grad_v2_base.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "hard_swish_grad_v2_base.h"

namespace HardSwishGradV2 {

using namespace AscendC;

template <typename T>
class HardSwishGradV2100 : public HardSwishGradV2Base<T>{
public:
    __aicore__ inline HardSwishGradV2100(){};

protected:
    LocalTensor<float> negativeTensorFp32;
    LocalTensor<float> positiveTensorFp32;
    LocalTensor<float> zeroTensorFp32;
    LocalTensor<float> oneTensorFp32;

protected:
__aicore__ inline void CastIn(int64_t dataCount) {
    SetFlag<HardEvent::MTE2_V>(this->eventId);
    WaitFlag<HardEvent::MTE2_V>(this->eventId);

    this->gradTensorFp32 = this->gradTensor.template ReinterpretCast<float>();
    this->selfTensorFp32 = this->selfTensor.template ReinterpretCast<float>();
    if (std::is_same_v<T, half>) {
        Cast(this->gradTensorFp32, this->gradTensor[this->elementNumEachCore], RoundMode::CAST_NONE, dataCount);
        Cast(this->selfTensorFp32, this->selfTensor[this->elementNumEachCore], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void CopyInAndCast (int64_t inputOffset, int64_t dataCount) {
    this->gradTensor = this->pingPongFlag ? this->tmpTensor[this->ubSize / NUM_TWO].template ReinterpretCast<T>() : this->tmpTensor[0].template ReinterpretCast<T>();
    this->selfTensor = this->pingPongFlag ? this->tmpTensor[this->elementNumEachCore * sizeof(float) + this->ubSize / NUM_TWO].template ReinterpretCast<T>() :
                              this->tmpTensor[this->elementNumEachCore * sizeof(float)].template ReinterpretCast<T>();
    if (std::is_same_v<T, half>) {
        DataCopy(this->gradTensor[this->elementNumEachCore], this->gradGm[inputOffset], dataCount);
        DataCopy(this->selfTensor[this->elementNumEachCore], this->selfGm[inputOffset], dataCount);
    } else {
        DataCopy(this->gradTensor, this->gradGm[inputOffset], dataCount);
        DataCopy(this->selfTensor, this->selfGm[inputOffset], dataCount);
    }

    CastIn(dataCount);
}

__aicore__ inline void CopyInAndCastTail(int64_t inputOffset, int64_t dataCount) {
    this->gradTensor = this->pingPongFlag ? this->tmpTensor[this->ubSize / NUM_TWO].template ReinterpretCast<T>() : this->tmpTensor[0].template ReinterpretCast<T>();
    this->selfTensor = this->pingPongFlag ? this->tmpTensor[this->elementNumEachCore * sizeof(float) + this->ubSize / NUM_TWO].template ReinterpretCast<T>() :
                              this->tmpTensor[this->elementNumEachCore * sizeof(float)].template ReinterpretCast<T>();
    for (int i = 0; i < dataCount; i++) {
        if (std::is_same_v<T, half>) {
            this->gradTensor[this->elementNumEachCore].SetValue(i, this->gradGm[inputOffset].GetValue(i));
            this->selfTensor[this->elementNumEachCore].SetValue(i, this->selfGm[inputOffset].GetValue(i));
        } else {
            this->gradTensor.SetValue(i, this->gradGm[inputOffset].GetValue(i));
            this->selfTensor.SetValue(i, this->selfGm[inputOffset].GetValue(i));
        }
    }
    PipeBarrier<PIPE_ALL>();

    CastIn(dataCount);
}

__aicore__ inline void Compute (int64_t dataCount) {
    int64_t tmpDataCount = (dataCount + ONE_REPEAT_ELE_NUM_FP32 - 1) / ONE_REPEAT_ELE_NUM_FP32 * ONE_REPEAT_ELE_NUM_FP32;

    int64_t elementByte = this->elementNumEachCore * sizeof(float);
    this->maskGreater = this->pingPongFlag ? this->tmpTensor[elementByte * NUM_TWO + this->ubSize / NUM_TWO] : this->tmpTensor[elementByte * NUM_TWO];
    this->maskLessThan = this->pingPongFlag ? this->tmpTensor[elementByte * NUM_THREE + this->ubSize / NUM_TWO] :
                                this->tmpTensor[elementByte * NUM_THREE];

    // step1
    negativeTensorFp32 = this->pingPongFlag ?
        this->tmpTensor[elementByte * NUM_FOUR + this->ubSize / NUM_TWO].template ReinterpretCast<float>() :
        this->tmpTensor[elementByte * NUM_FOUR].template ReinterpretCast<float>();
    Duplicate<float>(negativeTensorFp32, this->negative, dataCount);
    PipeBarrier<PIPE_V>();

    for (int i = 0; i * ONE_REPEAT_ELE_NUM_FP32 < dataCount; i++) {
        Compare(this->maskGreater[i * ONE_REPEAT_ELE_NUM_FP32], this->selfTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], negativeTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], CMPMODE::GT, ONE_REPEAT_ELE_NUM_FP32);
    }
    PipeBarrier<PIPE_V>();

    positiveTensorFp32 = negativeTensorFp32;
    Duplicate<float>(positiveTensorFp32, this->positive, dataCount);
    PipeBarrier<PIPE_V>();

    for (int i = 0; i * ONE_REPEAT_ELE_NUM_FP32 < dataCount; i++) {
        Compare(this->maskLessThan[i * ONE_REPEAT_ELE_NUM_FP32], this->selfTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], positiveTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], CMPMODE::LT, ONE_REPEAT_ELE_NUM_FP32);
    }
    PipeBarrier<PIPE_V>();

    // step2
    Muls(this->selfTensorFp32, this->selfTensorFp32, this->oneThird , dataCount);
    PipeBarrier<PIPE_V>();

    Adds(this->selfTensorFp32, this->selfTensorFp32, this->oneHalf, dataCount);
    PipeBarrier<PIPE_V>();

    // step3
    zeroTensorFp32 = negativeTensorFp32;
    Duplicate<float>(zeroTensorFp32, this->zero, ONE_REPEAT_ELE_NUM_FP32);
    PipeBarrier<PIPE_V>();

    for (int i = 0; i * ONE_REPEAT_ELE_NUM_FP32 < dataCount; i++) {
        Select(this->selfTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], this->maskGreater[i * ONE_REPEAT_ELE_NUM_FP32], this->selfTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], zeroTensorFp32, SELMODE::VSEL_CMPMASK_SPR , ONE_REPEAT_ELE_NUM_FP32);
    }
    PipeBarrier<PIPE_V>();

    oneTensorFp32 = negativeTensorFp32;
    Duplicate<float>(oneTensorFp32, this->one, ONE_REPEAT_ELE_NUM_FP32);
    PipeBarrier<PIPE_V>();

    for (int i = 0; i * ONE_REPEAT_ELE_NUM_FP32 < dataCount; i++) {
        Select(this->selfTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], this->maskLessThan[i * ONE_REPEAT_ELE_NUM_FP32], this->selfTensorFp32[i * ONE_REPEAT_ELE_NUM_FP32], oneTensorFp32, SELMODE::VSEL_CMPMASK_SPR , ONE_REPEAT_ELE_NUM_FP32);
    }
    PipeBarrier<PIPE_V>();

    // step4
    Mul(this->selfTensorFp32, this->gradTensorFp32, this->selfTensorFp32, dataCount);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void CastAndCopyOut (int64_t outputOffset, int64_t dataCount) {
    if (std::is_same_v<T, half>) {
        Cast(this->selfTensor, this->selfTensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(this->eventId);
    WaitFlag<HardEvent::V_MTE3>(this->eventId);
    DataCopy(this->outGm[outputOffset], this->selfTensor, dataCount);
}

__aicore__ inline void CastAndCopyOutTail(int64_t outputOffset, int64_t dataCount) {
    if (std::is_same_v<T, half>) {
        Cast(this->selfTensor, this->selfTensorFp32, RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
    }
    SetFlag<HardEvent::V_MTE3>(this->eventId);
    WaitFlag<HardEvent::V_MTE3>(this->eventId);

    for (int i = 0; i < dataCount; i++) {
        this->outGm[outputOffset].SetValue(i, this->selfTensor.GetValue(i));
    }
}

__aicore__ inline void ProcessAlign(int64_t inputOffset, int64_t dataCount) {
    CopyInAndCast(inputOffset, dataCount);
    Compute(dataCount);
    CastAndCopyOut(inputOffset, dataCount);
}

__aicore__ inline void ProcessTail (int64_t inputOffset, int64_t dataCount) {
    int64_t alignMinCount = UB_ALIGN_BYTE / sizeof(T);
    int64_t alignDataCount = (dataCount / alignMinCount) * alignMinCount;
    int64_t misAlignDataCount = dataCount - alignDataCount;
    int64_t padAlignCount = ((misAlignDataCount + alignMinCount - 1) / alignMinCount) * alignMinCount;
    if (alignDataCount > 0) {
        this->ProcessAlign(inputOffset, alignDataCount);
    }
    if (this->elementNum >= padAlignCount) {
        this->ProcessAlign(inputOffset + alignDataCount - (padAlignCount - misAlignDataCount), padAlignCount);
    } else {
        CopyInAndCastTail(inputOffset, dataCount);
        Compute(dataCount);
        CastAndCopyOutTail(inputOffset, dataCount);
    }
}

public:
__aicore__ inline void Process() {
    if (this->blockIdx >= this->needCoreNumber) {
        return;
    }
    
    int64_t totalTimes = this->elementNum / this->elementNumEachCore;
    int64_t remain = this->elementNum % this->elementNumEachCore;
    if (remain > 0) {
        totalTimes++;
    }
    int64_t loopNum = totalTimes / this->needCoreNumber;
    int64_t loopRemain = totalTimes % this->needCoreNumber;
    
    if (loopRemain > 0 && this->blockIdx < loopRemain) {
        loopNum++;
    }
    int64_t eachCoreStartOffset = loopNum * this->blockIdx * this->elementNumEachCore;
    if (loopRemain > 0) {
        if (this->blockIdx >= loopRemain) {
            eachCoreStartOffset += this->elementNum % (this->elementNumEachCore * this->needCoreNumber);
        }
    }
    
    int64_t calNum = this->elementNumEachCore;
    int64_t lastCoreNum = loopRemain == 0 ? this->needCoreNumber - 1 : loopRemain - 1;
    this->pingPongFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t i = 0; i < loopNum; i++) {
        this->eventId = this->pingPongFlag ? EVENT_ID1 : EVENT_ID0;
        WaitFlag<HardEvent::MTE3_MTE2>(this->eventId);

        int64_t localOffset = i * this->elementNumEachCore;
        if (remain > 0 && i == loopNum -1 && this->blockIdx == lastCoreNum) {
            calNum = remain;
        }
        
        if (remain > 0 && i == loopNum -1 && this->blockIdx == lastCoreNum && calNum * sizeof(T) % UB_ALIGN_BYTE != 0) {
            ProcessTail(eachCoreStartOffset + localOffset, calNum);
        } else {
            ProcessAlign(eachCoreStartOffset + localOffset, calNum);
        }

        SetFlag<HardEvent::MTE3_MTE2>(this->eventId);
        this->pingPongFlag = 1 - this->pingPongFlag;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
}
};
}// namespace HardSwishGradV2
#endif