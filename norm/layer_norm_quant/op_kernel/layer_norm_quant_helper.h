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
 * \file utils.h
 * \brief
 */
#ifndef UTILS_H
#define UTILS_H
#include "kernel_operator.h"
using namespace AscendC;
using AscendC::HardEvent;

namespace {
  static constexpr uint32_t BLOCK_SIZE = 32;
  static constexpr uint32_t BUFFER_NUM = 1;
  static constexpr uint32_t TAIL_BUFFER_SIZE = 32;
  static constexpr uint32_t MAX_UB_SIZE_NUM = 98304;
  static constexpr uint32_t INT8_DATA_BYTE = 1;
  static constexpr uint32_t BLOCK_NUMEL = 32;
  static constexpr float QUANT_MIN = -128;
}

__aicore__ inline uint32_t MIN(uint32_t x, uint32_t y) 
{ 
    return x < y ? x : y; 
}

__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 32)
{
  if (y == 0) {
    return x;
  }
  return (x + y - 1) / y * y;
}

__aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}
__aicore__ inline uint32_t MAX(uint32_t x, uint32_t y) 
{ 
    return x > y ? x : y; 
}

template <typename T>
__aicore__ inline void CastFrom32To16(const AscendC::LocalTensor<T> &out, const AscendC::LocalTensor<float> &in,
    uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(out, in, AscendC::RoundMode::CAST_NONE, count); // 310p cast fp32->half 只能用CAST_NONE，这里拉齐310p和910b
    } else { // bf16
        Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CastFromF16ToI8(const AscendC::LocalTensor<int8_t> &res, const AscendC::LocalTensor<half> &in,
    half quantMin, uint32_t count)
{
    Maxs(in, in, quantMin, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mins(in, in, (half)127, count); // 127: limit
    AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    Cast(res, in, AscendC::RoundMode::CAST_RINT, count);
#else
    Cast(res, in, AscendC::RoundMode::CAST_NONE, count);
#endif
    AscendC::PipeBarrier<PIPE_V>();
}

#endif
