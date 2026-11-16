/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_APPLY_CAME_PART1_C_REDUCE_COPY_OUT_H
#define OPS_NN_APPLY_CAME_PART1_C_REDUCE_COPY_OUT_H

#include "kernel_operator.h"

namespace ApplyCamePart1 {
using namespace AscendC;

constexpr uint64_t kApplyCamePart1Fp32BlockCount = 32 / sizeof(float);
constexpr uint64_t kApplyCamePart1MaxCopyCount = 16376;

__aicore__ inline void AtomicAddScalarCReduceResult(GlobalTensor<float>& output, LocalTensor<float> input,
                                                    uint64_t offset, uint64_t begin, uint64_t count)
{
    for (uint64_t i = begin; i < count; ++i) {
        AtomicAdd(output.GetPhyAddr(offset + i), input.GetValue(i));
    }
}

__aicore__ inline uint64_t AtomicAddAlignedCReduceResult(GlobalTensor<float>& output, LocalTensor<float> input,
                                                         uint64_t offset, uint64_t count)
{
    const uint64_t alignedCount = count / kApplyCamePart1Fp32BlockCount * kApplyCamePart1Fp32BlockCount;
    if (alignedCount == 0) {
        return 0;
    }

    SetAtomicAdd<float>();
    for (uint64_t copied = 0; copied < alignedCount; copied += kApplyCamePart1MaxCopyCount) {
        const uint64_t remaining = alignedCount - copied;
        const uint64_t copyCount = remaining < kApplyCamePart1MaxCopyCount ? remaining : kApplyCamePart1MaxCopyCount;
        DataCopy(output[offset + copied], input[copied], copyCount);
    }
    SetAtomicNone();
    event_t eventMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventMte3ToS);
    return alignedCount;
}

__aicore__ inline void AtomicAddCReduceResult(GlobalTensor<float>& output, LocalTensor<float> input, uint64_t offset,
                                              uint64_t count)
{
    // The reduction result is produced by the vector pipe. The scalar
    // prefix/tail reads it through the scalar pipe, while the aligned body is
    // written by MTE3, so establish both dependencies before mixing engines.
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    // Every write remains atomic because adjacent batch groups can share one
    // 32B GM block. DataCopy requires both addresses to start at a 32B
    // boundary; an unaligned GM offset cannot be repaired with a scalar prefix
    // because it would make the UB source unaligned, so keep that whole tile
    // on scalar AtomicAdd.
    uint64_t scalarBegin = 0;
    if ((offset % kApplyCamePart1Fp32BlockCount) == 0) {
        scalarBegin = AtomicAddAlignedCReduceResult(output, input, offset, count);
    }
    const bool hasScalarWrite = scalarBegin < count;
    AtomicAddScalarCReduceResult(output, input, offset, scalarBegin, count);

    if (hasScalarWrite) {
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    }
    event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
}
} // namespace ApplyCamePart1

#endif // OPS_NN_APPLY_CAME_PART1_C_REDUCE_COPY_OUT_H
