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
};
constexpr uint64_t kReserveBytes = 8192;
constexpr uint64_t AlignDown(uint64_t n, uint64_t a) { return n / a * a; }
// Lifetimes in the reusable arena:
// elementwise: values[t*t], rowAccumulator[2*t], cast[t*t]
// R/C/RC:      values[t*t], rowAccumulator[2*t], scratch[12*t]
// post:       six FP32 pair-reduction nodes plus one 64B scalar-output slot.
// Input/output queues remain physically allocated throughout all phases.
inline Plan MakePlan(uint64_t ub, uint64_t dtypeBytes, uint64_t cores)
{
    Plan p;
    if ((dtypeBytes != 2 && dtypeBytes != 4) || cores == 0 || cores > 4096 || ub <= kReserveBytes)
        return p;
    // Power-of-two tiles preserve alignment and the vector reduction contract.
    for (uint64_t t = 64; t <= 128; t *= 2) {
        const uint64_t fixed = t * t * dtypeBytes + 32 + 2 * t * 4 + 16 * 4 + cores * 32;
        const uint64_t work = t * t > 12 * t ? t * t : 12 * t;
        const uint64_t producerPeak = (t * t + 2 * t + work) * 4;
        if (ub <= kReserveBytes + fixed)
            continue;
        const uint64_t arena = AlignDown(ub - kReserveBytes - fixed, 256);
        if (arena < producerPeak || arena <= 64)
            continue;
        const uint64_t post = AlignDown((arena - 64) / 4, t);
        if (post < 2 * t || post < 6 * 8)
            continue;
        p.tile = t;
        p.persistentBytes = fixed;
        p.arenaBytes = arena;
        p.producerPeakBytes = producerPeak;
        p.postElements = post;
    }
    // Direct: cast and output have disjoint lifetimes and share one slot.
    // grad(T), values(FP32), cast/output(FP32), plus fixed queue/scratch slots.
    constexpr uint64_t fixedDirect = 32 + 2 * 64 * 4 + 12 * 64 * 4 + (2 * 32 + 3 * 8) * 4;
    if (ub > kReserveBytes + fixedDirect) {
        const uint64_t count = AlignDown((ub - kReserveBytes - fixedDirect) / (dtypeBytes + 8), 64);
        p.directElements = count < 8192 ? count : 8192;
    }
    return p;
}
} // namespace apply_came_ub
#endif
