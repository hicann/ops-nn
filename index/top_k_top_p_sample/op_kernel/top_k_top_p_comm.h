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
 * \file top_k_top_p_comm.h
 * \brief
 */

#ifndef TOP_K_TOP_P_COMM_H
#define TOP_K_TOP_P_COMM_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using AscendC::TEventID;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::WaitFlag;

namespace TopKTopPSample{
    constexpr uint32_t SIZEOF_FP32 = sizeof(float);
    constexpr uint32_t KERNEL_BUFFER_SIZE = 32768;
    constexpr uint32_t NUM_ZERO = 0;
    constexpr uint32_t NUM_ONE = 1;
    constexpr uint32_t NUM_TWO = 2;
    constexpr uint32_t NUM_THREE = 3;
    constexpr uint32_t NUM_FOUR = 4;
    constexpr uint32_t NUM_SIX = 6;
    constexpr uint32_t MRG_PER_ELE = 2;
    constexpr uint32_t BUFFER_NUM = 1;
    constexpr uint32_t FIFTEEN = 15;
    constexpr uint32_t THIRTY_ONE = 31;
    constexpr uint32_t THIRTY_TWO = 32;
    constexpr uint32_t SIXTY_FOUR = THIRTY_TWO * NUM_TWO;
    constexpr float FLOAT_MIN = -3.40282e+38f;
    constexpr uint32_t SOFTMAX_PER_LEN = (KERNEL_BUFFER_SIZE / SIZEOF_FP32);  

    constexpr uint32_t SIZEOF_UINT32 = sizeof(uint32_t);
    constexpr uint32_t SIZEOF_UINT64 = sizeof(uint64_t);

    __aicore__ inline int64_t SafeCeil(int64_t a, int64_t b){
        if(b == 0){
            return 0;
        }
        return (a + b - 1) / b;
    }

    __aicore__ inline int64_t Align(int64_t a, int64_t b){
        if(b == 0){
            return 0;
        }
        return SafeCeil(a, b) * b;
    }

    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt){
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    template <typename T, template <typename U> typename R, template <typename U> typename S>
    __aicore__ inline void DataCopyEx(
        const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count = 1, const bool ubAligned = false)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = count;
        copyParams.blockLen = len * sizeof(T);
        if constexpr (std::is_same<R<T>, AscendC::LocalTensor<T>>::value) {
            copyParams.srcStride = 0;
            copyParams.dstStride = (ubAligned) ? 1 : 0;
            DataCopyPad(dst, src, copyParams, {});
        } else {
            copyParams.srcStride = (ubAligned) ? 1 : 0;
            copyParams.dstStride = 0;
            DataCopyPad(dst, src, copyParams);
        }
    }

    using TOPKPParams = struct TOPKPParams {
        int32_t rowId;
        uint32_t rowNum;
        uint32_t rowLen;
        uint32_t partLen;
        uint32_t partLoopTime;
        uint32_t partLoopTail;
        uint32_t partLoopTailPad;
        uint32_t softmaxLoopTime;
        uint32_t softmaxLoopEleTail;
        uint32_t softmaxLoopEleTailPad;
        uint32_t eightKPartNum;
        uint32_t eightKPartTail;
        uint32_t eightKPartTailPad;
        float topp;
        uint32_t toppNum;
        LocalTensor<float> tensor0;
    };
}
#endif