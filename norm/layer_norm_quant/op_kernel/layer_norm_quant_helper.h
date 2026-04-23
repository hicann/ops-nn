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
 * \file layer_norm_quant_helper.h
 * \brief
 */
#ifndef LAYER_NORM_QUANT_HELPER_H
#define LAYER_NORM_QUANT_HELPER_H
#include "kernel_operator.h"
using namespace AscendC;
using AscendC::HardEvent;

namespace {
  static constexpr uint32_t BLOCK_NUM = 16;
  static constexpr uint32_t DATA_BYTE = 2;
  static constexpr uint32_t BLOCK_SIZE = 32;
  static constexpr uint32_t BUFFER_NUM = 1;
  static constexpr uint32_t TAIL_BUFFER_SIZE = 32;
  static constexpr uint32_t MAX_UB_SIZE_NUM = 98304;
  static constexpr uint32_t INT8_DATA_BYTE = 1;
  static constexpr uint32_t BLOCK_NUMEL = 32;
  static constexpr float QUANT_MIN = -128;
}

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {
};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {
};

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

template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(
    const R<T>& dst, const S<T>& src, const AscendC::LocalTensor<T> &tmp, const uint32_t num_last_dim,
    const uint32_t size = 1)
{
    auto eleCount = num_last_dim * size;
    int32_t numPerBlock = BLOCK_SIZE / sizeof(T);
    if (eleCount % numPerBlock == 0) {
        DataCopy(dst, src, eleCount);
    } else {
        if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
            auto num = AlignUp(eleCount, numPerBlock);
            DataCopy(dst, src, num);
        } else {
            int32_t num = eleCount / numPerBlock * numPerBlock;
            DataCopy(dst, src, num);
            if (eleCount != num) {
                SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
                for (int32_t i = 0; i < numPerBlock; i++) {
                    auto tensorValue = src.GetValue(eleCount - numPerBlock + i);
                    tmp.SetValue(i, tensorValue);
                }
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
                DataCopy(dst[eleCount - numPerBlock], tmp, numPerBlock);
            }
        }
    }
}

#endif // LAYER_NORM_QUANT_HELPER_H
