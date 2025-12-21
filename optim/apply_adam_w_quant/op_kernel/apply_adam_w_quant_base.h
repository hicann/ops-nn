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
 * \file apply_adam_w_quant_base.h
 * \brief
 */
#ifndef _APPLY_ADAM_W_QUANT_BASE_H_
#define _APPLY_ADAM_W_QUANT_BASE_H_

#include "kernel_operator.h"

namespace ApplyAdamWQuantNS {
using namespace AscendC;

constexpr int32_t Q_MAP_SIZE = 256;
constexpr uint32_t CALC_BUF_NUM = 6;
constexpr uint32_t PER_UINT8_8BITS = 8;
constexpr uint32_t REPEAT_NUM = 64;
constexpr uint32_t REPEAT_7_TIMES = 7;
constexpr uint32_t REPEAT_NUM_128 = 128;
constexpr uint32_t STRIDE_8 = 8;
constexpr uint32_t PER_4NUM_ONEMAX = 4;
constexpr uint32_t BROADCAST_DIM2 = 2;
constexpr uint32_t BROADCAST_AXIS1 = 1;

template <typename T>
__aicore__ inline void DataCopyIn(
    const AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void DataCopyOut(
    const AscendC::GlobalTensor<T>& dst, const AscendC::LocalTensor<T>& src, uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    
    AscendC::DataCopyPad(dst, src, copyParams);
}

template <typename T, typename T1>
__aicore__ inline void CastF16ToFp32(
    const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T1>& src, uint32_t count)
{
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}

template <typename T, typename T1>
__aicore__ inline void CastFp32ToF16(
    const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T1>& src, uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    } else { // bf16
        Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
    }
}

template <AscendC::HardEvent hardEvent>
__aicore__ inline void PipeSync()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
    AscendC::SetFlag<hardEvent>(eventID);
    AscendC::WaitFlag<hardEvent>(eventID);
}

__aicore__ inline float PowS(
    const AscendC::LocalTensor<float>& dst, float srcScalar, const AscendC::LocalTensor<float>& src)
{
    AscendC::Power<float>(dst, srcScalar, src);
    PipeSync<AscendC::HardEvent::V_S>();
    float ret = dst.GetValue(0);
    AscendC::PipeBarrier<PIPE_ALL>();
    return ret;
}

}

#endif // _APPLY_ADAM_W_QUANT_BASE_H_