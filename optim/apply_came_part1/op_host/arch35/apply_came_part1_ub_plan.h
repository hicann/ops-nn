/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef APPLY_CAME_PART1_UB_PLAN_H
#define APPLY_CAME_PART1_UB_PLAN_H
#include <cstdint>
namespace apply_came_ub {
struct Plan {
    uint64_t tile = 0;
    uint64_t persistentBytes = 0;
    uint64_t arenaBytes = 0;
    uint64_t producerPeakBytes = 0;
    uint64_t postElements = 0;
    uint64_t directElements = 0;
    uint64_t gradQueueBytes = 0;
    uint64_t epsQueueBytes = 0;
    uint64_t sumGradRQueueBytes = 0;
    uint64_t sumGradCQueueBytes = 0;
    uint64_t sumGradRCQueueBytes = 0;
    uint64_t syncWorkQueueBytes = 0;
    uint64_t directGradQueueBytes = 0;
    uint64_t directEpsQueueBytes = 0;
    uint64_t directSumGradRQueueBytes = 0;
    uint64_t directSumGradCQueueBytes = 0;
    uint64_t directCastOutputBytes = 0;
    uint64_t directValueBytes = 0;
    uint64_t directReduceBytes = 0;
};
constexpr uint64_t kReserveBytes = 8192;
constexpr uint64_t AlignDown(uint64_t n, uint64_t a) { return n / a * a; }
constexpr uint64_t Width2ProducerPeakBytes(uint64_t tile)
{
    // ProcessWidth2 keeps two tile-square regions alive for values and row
    // high/low results, followed by three 8-float slots per row tile for the
    // packed C and RC high/low workspace payloads.
    return (2 * tile * tile + 12 * tile) * sizeof(float);
}
// Lifetimes in the reusable arena:
// elementwise: values[t*t], rowAccumulator[2*t], cast[t*t]
// R/C/RC:      values[t*t], rowAccumulator[2*t], scratch[12*t]
// FP32 M==2:   values[t*t], row high/low[t*t], packed C/RC[12*t]
// post:       six FP32 pair-reduction nodes plus one 64B scalar-output slot.
// Input/output queues remain physically allocated throughout all phases.
inline Plan MakePlan(uint64_t ub, uint64_t dtypeBytes, uint64_t cores)
{
    Plan p;
    if ((dtypeBytes != 2 && dtypeBytes != 4) || cores == 0 || cores > 4096 || ub <= kReserveBytes)
        return p;
    // Power-of-two tiles preserve alignment and the vector reduction contract.
    for (uint64_t t = 64; t <= 128; t *= 2) {
        // R and C both publish compensated high/low pairs.  Keep both
        // components alive until the post reduction adds eps and rounds once.
        const uint64_t gradQueueBytes = t * t * dtypeBytes;
        // FP16/BF16 add eps once in the compensated post reduction.  Only the
        // FP32 width-2 direct-R path consumes eps in the producer kernel.
        const uint64_t epsQueueBytes = dtypeBytes == sizeof(float) ? 8 * sizeof(float) : 0;
        const uint64_t sumGradRQueueBytes = 2 * t * 4;
        const uint64_t sumGradCQueueBytes = 2 * t * 4;
        constexpr uint64_t sumGradRCQueueBytes = 2 * 8 * 4;
        const uint64_t syncWorkQueueBytes = cores * 8 * sizeof(int32_t);
        const uint64_t fixed = gradQueueBytes + epsQueueBytes + sumGradRQueueBytes + sumGradCQueueBytes +
                               sumGradRCQueueBytes + syncWorkQueueBytes;
        const uint64_t work = t * t > 12 * t ? t * t : 12 * t;
        const uint64_t generalProducerPeak = (t * t + 2 * t + work) * sizeof(float);
        const uint64_t width2ProducerPeak = dtypeBytes == sizeof(float) ? Width2ProducerPeakBytes(t) : 0;
        const uint64_t producerPeak = generalProducerPeak > width2ProducerPeak ? generalProducerPeak :
                                                                                 width2ProducerPeak;
        if (ub <= kReserveBytes + fixed)
            continue;
        const uint64_t arena = AlignDown(ub - kReserveBytes - fixed, 256);
        if (arena < producerPeak || arena <= 64)
            continue;
        const uint64_t post = AlignDown((arena - 64) / 4, t);
        if (post < 6 * t)
            continue;
        p.tile = t;
        p.persistentBytes = fixed;
        p.arenaBytes = arena;
        p.producerPeakBytes = producerPeak;
        p.postElements = post;
        p.gradQueueBytes = gradQueueBytes;
        p.epsQueueBytes = epsQueueBytes;
        p.sumGradRQueueBytes = sumGradRQueueBytes;
        p.sumGradCQueueBytes = sumGradCQueueBytes;
        p.sumGradRCQueueBytes = sumGradRCQueueBytes;
        p.syncWorkQueueBytes = syncWorkQueueBytes;
    }
    // Direct: cast and output have disjoint lifetimes and share one slot.
    // grad(T), values(FP32), cast/output(FP32), plus fixed queue/scratch slots.
    // The direct R queue carries one 64-element high row and one 64-element
    // low row concurrently. Count both rows in the peak live set.
    constexpr uint64_t fixedDirect = 32 + 5 * 64 * 4 + 12 * 64 * 4 + (2 * 32 + 3 * 8) * 4;
    if (ub > kReserveBytes + fixedDirect) {
        const uint64_t count = AlignDown((ub - kReserveBytes - fixedDirect) / (dtypeBytes + 8), 64);
        p.directElements = count < 8192 ? count : 8192;
        p.directGradQueueBytes = p.directElements * dtypeBytes;
        p.directEpsQueueBytes = 8 * sizeof(float);
        p.directSumGradRQueueBytes = 2 * 64 * sizeof(float);
        p.directSumGradCQueueBytes = 2 * 64 * sizeof(float);
        p.directCastOutputBytes = (p.directElements + 2 * 32 + 3 * 8) * sizeof(float);
        p.directValueBytes = p.directElements * sizeof(float);
        p.directReduceBytes = 12 * 64 * sizeof(float);
    }
    return p;
}
} // namespace apply_came_ub
#endif
