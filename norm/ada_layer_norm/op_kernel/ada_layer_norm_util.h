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
 * \file ada_layer_norm_util.h
 * \brief
 */
#ifndef ADA_LAYER_NORM_UTIL_H
#define ADA_LAYER_NORM_UTIL_H

#include "kernel_operator.h"

namespace AdaLayerNormNS {
using namespace AscendC;

constexpr uint8_t BASE_OP_CODE = 1;
constexpr uint8_t BASE_V2_OP_CODE = 12;
constexpr uint8_t QUANT_OP_CODE = 2;
constexpr int32_t MAX_X_SIZE = 16384;
constexpr int32_t TENSOR_NUM = 7;
constexpr int32_t DATA_COUNT = 6144;
constexpr int32_t BATCH_COUNT = 1024;
constexpr int32_t INT8_BLOCK_NUM = 32;
constexpr int32_t HALF_BLOCK_NUM = 16;
constexpr int32_t FLOAT_BLOCK_NUM = 8;
constexpr float MAX_INT8 = 127.0f;
constexpr float ONE_FLOAT = 1.0f;
constexpr float FACTOR_INT8 = 1.0f / 127.0f;
constexpr int64_t NUM_PER_REP_FP32 = 64;
constexpr int64_t MOD_64_MASK = 0x3f;
constexpr int64_t LOG2_64 = 6;

struct RowRange {
    int64_t rowStart;
    int64_t rowEnd;
    int64_t actualRowNum;
    int64_t batchStart;
    int64_t batchEnd;
    int64_t dataCount;
};

struct GmAddr {
    const GM_ADDR x = nullptr;
    const GM_ADDR scale = nullptr;
    const GM_ADDR shift = nullptr;
    const GM_ADDR weight = nullptr;
    const GM_ADDR bias = nullptr;
    const GM_ADDR smooth_scales = nullptr;
    const GM_ADDR out = nullptr;
    const GM_ADDR mean = nullptr;
    const GM_ADDR rstd = nullptr;
    const GM_ADDR quant_scale = nullptr;
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilA2B(T1 a, T2 b)
{
    return (b != 0) ? (a + b - 1) / b : a;
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a < b) ? a : b;
}

__aicore__ inline void PIPE_MTE2_V()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

__aicore__ inline void PIPE_V_MTE2()
{
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
}

__aicore__ inline void PIPE_V_MTE3()
{
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
}

// count <= 64 * 256(16K)
__aicore__ inline void ReduceMaxCustom(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, int64_t count)
{
    int64_t repeatTimes = count >> LOG2_64;
    int64_t tailCount = count & MOD_64_MASK;

    BinaryRepeatParams repeatParams = {1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0};
    if (likely(repeatTimes > 1)) {
        Max(srcLocal, srcLocal[NUM_PER_REP_FP32], srcLocal, NUM_PER_REP_FP32, repeatTimes - 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount > 0)) {
        Max(srcLocal, srcLocal[repeatTimes << LOG2_64], srcLocal, tailCount, 1, repeatParams);
        PipeBarrier<PIPE_V>();
    }
    WholeReduceMax(dstLocal, srcLocal, repeatTimes > 0 ? NUM_PER_REP_FP32 : count, 1, 0, 1, 0, ReduceOrder::ORDER_ONLY_VALUE);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint16_t blockCount, int64_t len)
{
    PIPE_V_MTE3();
    DataCopyExtParams copyOutParams{blockCount, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPad(dst, src, copyOutParams);
}

template <typename T>
__aicore__ inline void CopyInAndCast(const LocalTensor<float>& dst, const GlobalTensor<T>& src, int64_t len, int64_t midOffset)
{
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        DataCopyPad(dst, src, copyInParams, padParams);
    } else {
        DataCopyPad(dst.ReinterpretCast<T>()[midOffset], src, copyInParams, padParams);
    }
    PIPE_MTE2_V();
    if constexpr (!std::is_same_v<T, float>) {
        Cast(dst, dst.ReinterpretCast<T>()[midOffset], RoundMode::CAST_NONE, len);
        PipeBarrier<PIPE_V>();
    }
}

} // namespace AdaLayerNormNS

#endif  // ADA_LAYER_NORM_UTIL_H
