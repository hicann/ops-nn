/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_MAX_H
#define UNSORTED_SEGMENT_MAX_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace UnsortedSegment {
using namespace AscendC;
using namespace platform;

template <typename T>
__aicore__ inline constexpr T GetDtypeMin()
{
    T dtypeMin = 0;
    if constexpr (IsSameType<T, int32_t>::value) {
        dtypeMin = INT32_MIN;
    } else if constexpr (IsSameType<T, int64_t>::value) {
        dtypeMin = INT64_MIN;
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        dtypeMin = static_cast<bfloat16_t>(-BFLOAT16_MAX);
    } else if constexpr (IsSameType<T, half>::value) {
        dtypeMin = static_cast<half>(-FLOAT16_MAX);
    } else if constexpr (IsSameType<T, float>::value) {
        dtypeMin = -FLOAT32_MAX;
    }
    return dtypeMin;
}

// SIMT gather: return max of two values
template <typename T>
struct SimtGatherMaxValue {
    __simt_callee__ __aicore__ inline SimtGatherMaxValue() {}
    __simt_callee__ __aicore__ inline T operator()(T midResP, T xUbLocalRes)
    {
        return midResP > xUbLocalRes ? midResP : xUbLocalRes;
    }
};

// SIMT atomic: write max to GM
template <typename T>
struct SimtComputeSegmentMax {
    __simt_callee__ __aicore__ inline SimtComputeSegmentMax() {}
    __simt_callee__ __aicore__ inline void operator()(__gm__ T* outputGm, const uint64_t outputIndex, T value)
    {
        Simt::AtomicMax(outputGm + outputIndex, value);
    }
};

// Identity value for max reduction (min of type), used as F0/F3
template <typename T>
struct GetInitMinValue {
    static __aicore__ inline constexpr T Get() { return GetDtypeMin<T>(); }
};

// GM initialization: fill with min value (F4)
template <typename T>
struct InitGmMinValue {
    __aicore__ inline InitGmMinValue() {}
    __aicore__ inline void operator()(GlobalTensor<T> yGmInit, uint64_t initCoreReal)
    {
        InitGlobalMemory(yGmInit, initCoreReal, GetDtypeMin<T>());
    }
};

// Vector Max operation (F5)
template <typename T, typename D>
struct ComputeMax {
    __aicore__ inline ComputeMax() {}
    __aicore__ inline void operator()(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal, D dstOffset, D srcOffset, D cols)
    {
        AscendC::Max(yLocal[dstOffset], yLocal[dstOffset], xLocal[srcOffset], cols);
    }
};

// Set atomic max mode (F6)
template <typename T>
struct SetAtomicMaxOp {
    __aicore__ inline SetAtomicMaxOp() {}
    __aicore__ inline void operator()() { AscendC::SetAtomicMax<T>(); }
};

} // namespace UnsortedSegment

#endif // UNSORTED_SEGMENT_MAX_H
