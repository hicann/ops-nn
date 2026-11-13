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
 * \file advance_step_common.h
 * \brief
 */
#ifndef ADVANCE_STEP_COMMON_H
#define ADVANCE_STEP_COMMON_H

#include "kernel_operator.h"

namespace AdvanceStepNs {
using namespace AscendC;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;
constexpr int64_t INT64_ONE_BLOCK_NUM = 4;
constexpr int64_t MAX_ELEMENT_ONE_LOOP = 100;
// V2 向量化 gather 常量（向量 Gather 已回退标量查表，kernel 侧仅保留 Gather 对齐粒度；
// tiling 侧的 VEC_BT_CHUNK_CAP_BYTES / VEC_GATHER_MAX_LANES 预算常量见 advance_step_tiling.cpp）
constexpr int64_t GATHER_LANES_PER_REPEAT = 64; // Gather 每 repeat 偏移读取粒度（256B/4B，仅 kernel 侧使用）
constexpr int64_t VEC_SCRATCH_ALIGN_SLACK = 1024; // 向量化 scratch 尾部对齐槽余量（子张量 32B 对齐 + Compares/Select
                                                  // 256b 粒度计数取整）

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a < b ? b : a;
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace AdvanceStepNs
#endif // ADVANCE_STEP_COMMON_H
