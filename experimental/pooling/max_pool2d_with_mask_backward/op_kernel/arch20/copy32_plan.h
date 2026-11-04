// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#ifndef MAXPOOL310P_COPY32_PLAN_H
#define MAXPOOL310P_COPY32_PLAN_H
#include <cstdint>
#ifndef MAXPOOL310P_PLAN_FUNCTION
#define MAXPOOL310P_PLAN_FUNCTION
#define MAXPOOL310P_UNDEF_PLAN_FUNCTION
#endif

namespace MaxPool310P {
struct CopyPlan {
    bool valid = false;
    uint32_t mainCount = 0;
    uint32_t tailCount = 0;
    uint64_t tailGm = 0;
    uint32_t tailUb = 0;
    uint32_t skip = 0;
    uint32_t span = 0;
};

// The same checked address plans are used by device code and native exhaustive
// tests. "physical" is a caller-guaranteed allocation extent, not allocator slack.
MAXPOOL310P_PLAN_FUNCTION constexpr bool ValidRange(uint64_t total, uint64_t physical, uint64_t offset, uint32_t count,
                                                    uint32_t capacity, uint32_t block)
{
    return (block == 8 || block == 16) && physical >= total && offset <= total && count <= total - offset &&
           (static_cast<uint64_t>(count) + block - 1) / block * block <= capacity;
}

MAXPOOL310P_PLAN_FUNCTION constexpr CopyPlan PlanIn(uint64_t total, uint64_t physical, uint64_t offset, uint32_t count,
                                                    uint32_t capacity, uint32_t block)
{
    CopyPlan p;
    if (!ValidRange(total, physical, offset, count, capacity, block))
        return p;
    if (count == 0) {
        p.valid = true;
        return p;
    }
    if (physical < block)
        return p;
    p.valid = true;
    p.mainCount = count / block * block;
    p.tailCount = count % block;
    if (p.tailCount) {
        const uint64_t requested = offset + p.mainCount;
        const uint64_t lastStart = total >= block ? total - block : 0;
        p.tailGm = requested < lastStart ? requested : lastStart;
        p.skip = static_cast<uint32_t>(requested - p.tailGm);
        p.tailUb = p.mainCount;
        p.span = block;
    }
    return p;
}

MAXPOOL310P_PLAN_FUNCTION constexpr CopyPlan PlanOut(uint64_t total, uint64_t physical, uint64_t offset, uint32_t count,
                                                     uint32_t capacity, uint32_t block)
{
    CopyPlan p;
    if (!ValidRange(total, physical, offset, count, capacity, block))
        return p;
    if (count == 0) {
        p.valid = true;
        return p;
    }
    // A sub-block partial write cannot be exact with 32B DMA. The caller must
    // merge it with its own adjacent block. Only a whole short tensor may use
    // explicitly reserved allocation padding.
    if (count < block && (offset != 0 || count != total || physical < block))
        return p;
    p.valid = true;
    p.mainCount = count / block * block;
    p.tailCount = count % block;
    if (p.tailCount) {
        if (count < block) {
            p.span = block;
        } else {
            p.tailGm = offset + count - block;
            p.tailUb = p.mainCount - block;
            p.skip = p.tailCount;
            p.tailCount = block;
            p.span = 2 * block;
        }
    }
    return p;
}
} // namespace MaxPool310P
#ifdef MAXPOOL310P_UNDEF_PLAN_FUNCTION
#undef MAXPOOL310P_PLAN_FUNCTION
#undef MAXPOOL310P_UNDEF_PLAN_FUNCTION
#endif
#endif
