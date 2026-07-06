/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNSORTED_SEGMENT_MIN_H
#define UNSORTED_SEGMENT_MIN_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace UnsortedSegment {
using namespace AscendC;
using namespace platform;

// SIMT gather: return min of two values
template <typename T>
struct SimtGatherMinValue {
    __simt_callee__ __aicore__ inline SimtGatherMinValue() {}
    __simt_callee__ __aicore__ inline T operator()(T midResP, T xUbLocalRes)
    {
        return midResP < xUbLocalRes ? midResP : xUbLocalRes;
    }
};

// SIMT atomic: write min to GM
template <typename T>
struct SimtComputeSegmentMin {
    __simt_callee__ __aicore__ inline SimtComputeSegmentMin() {}
    __simt_callee__ __aicore__ inline void operator()(__gm__ T* outputGm, const uint64_t outputIndex, T value)
    {
        Simt::AtomicMin(outputGm + outputIndex, value);
    }
};

// Identity value for min reduction (max of type), used as F0/F3
template <typename T>
struct GetInitMaxValue {
    static __aicore__ inline constexpr T Get() { return GetDtypeMax<T>(); }
};

// GM initialization: fill with max value (F4)
template <typename T>
struct InitGmMaxValue {
    __aicore__ inline InitGmMaxValue() {}
    __aicore__ inline void operator()(GlobalTensor<T> yGmInit, uint64_t initCoreReal)
    {
        InitGlobalMemory(yGmInit, initCoreReal, GetDtypeMax<T>());
    }
};

// Vector Min operation (F5)
template <typename T, typename D>
struct ComputeMin {
    __aicore__ inline ComputeMin() {}
    __aicore__ inline void operator()(LocalTensor<T>& yLocal, LocalTensor<T>& xLocal, D dstOffset, D srcOffset, D cols)
    {
        AscendC::Min(yLocal[dstOffset], yLocal[dstOffset], xLocal[srcOffset], cols);
    }
};

// Set atomic min mode (F6)
template <typename T>
struct SetAtomicMinOp {
    __aicore__ inline SetAtomicMinOp() {}
    __aicore__ inline void operator()() { AscendC::SetAtomicMin<T>(); }
};

} // namespace UnsortedSegment

#endif // UNSORTED_SEGMENT_MIN_H
