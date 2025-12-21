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
 * \file hard_swish_grad_v2_base.h
 * \brief hard_swish_grad_v2_base head file
 */
#ifndef HARD_SWISH_GRAD_V2_BASE_H
#define HARD_SWISH_GRAD_V2_BASE_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace HardSwishGradV2 {

using namespace AscendC;

constexpr int64_t ONE_REPEAT_ELE_NUM_FP32 = 64;

constexpr int64_t UB_ALIGN_BYTE = 32;

constexpr int64_t BITS_ONE_BYTE = 8;

constexpr int64_t NUM_TWO = 2;

constexpr int64_t NUM_THREE = 3;

constexpr int64_t NUM_FOUR = 4;

template <typename T>
class HardSwishGradV2Base {
public:
    TPipe pipe;
    __aicore__ inline HardSwishGradV2Base(){};
    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                const HardSwishGradV2TilingData* tilingData);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CopyInAndCast(int64_t inputOffset, int64_t dataCount);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CastAndCopyOut(int64_t outputOffset, int64_t dataCount);
    __aicore__ inline void ProcessTail(int64_t inputOffset, int64_t dataCount);
    __aicore__ inline void ProcessAlign(int64_t inputOffset, int64_t dataCount);

protected:
    TBuf<QuePosition::VECCALC> ubTBuf;
    LocalTensor<uint8_t> tmpTensor;

    LocalTensor<uint8_t> maskGreater;
    LocalTensor<uint8_t> maskLessThan;

    LocalTensor<T> gradTensor;
    LocalTensor<float> gradTensorFp32;

    LocalTensor<T> selfTensor;
    LocalTensor<float> selfTensorFp32;

    GlobalTensor<T> gradGm;
    GlobalTensor<T> selfGm;
    GlobalTensor<T> outGm;

    int64_t elementNum = 0;
    uint32_t needCoreNumber = 0;
    uint64_t ubSize = 0;
    int64_t elementNumEachCore = 0;
    int32_t blockIdx = 0;

    event_t eventId = EVENT_ID0;
    int32_t pingPongFlag = 0;

    float oneThird = static_cast<float>(0.33333334);
    float oneHalf= static_cast<float>(0.5);

    float negative= static_cast<float>(-3.0);
    float positive = static_cast<float>(3.0);
    float zero= static_cast<float>(0.0);
    float one= static_cast<float>(1.0);
};

template <typename T>
__aicore__ inline void HardSwishGradV2Base<T>::Init(GM_ADDR gradOutput, GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                              const HardSwishGradV2TilingData* tilingData) {
    gradGm.SetGlobalBuffer((__gm__ T*)gradOutput);
    selfGm.SetGlobalBuffer((__gm__ T*)self);
    outGm.SetGlobalBuffer((__gm__ T*)out);

    elementNum = tilingData->elementNum;
    needCoreNumber = tilingData->needCoreNum;
    ubSize = tilingData->ubSize;
    elementNumEachCore = tilingData->elementNumEachCore;

    blockIdx = GetBlockIdx();
    pipe.InitBuffer(ubTBuf, ubSize);
    tmpTensor = ubTBuf.Get<uint8_t>();
}

template <typename T>
__aicore__ inline void HardSwishGradV2Base<T>::Process() {
}

template <typename T>
__aicore__ inline void HardSwishGradV2Base<T>::ProcessAlign(int64_t inputOffset, int64_t dataCount) {
}

template <typename T>
__aicore__ inline void HardSwishGradV2Base<T>::ProcessTail(int64_t inputOffset, int64_t dataCount) {
}

template <typename T>
__aicore__ inline void HardSwishGradV2Base<T>::CopyInAndCast(int64_t inputOffset, int64_t dataCount) {
}

template <typename T>
__aicore__ inline void HardSwishGradV2Base<T>::Compute(int64_t dataCount) {
}


template <typename T>
__aicore__ inline void HardSwishGradV2Base<T>::CastAndCopyOut(int64_t outputOffset, int64_t dataCount) {
}
}// namespace HardSwishGradV2
#endif