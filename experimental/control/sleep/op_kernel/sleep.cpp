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
 * \file sleep.cpp
 * \brief Sleep operator kernel — busy-spin on AI Core SYS_CNT hardware counter.
 *
 * Semantics match CUDA's spin_kernel:  sleep(N) = spin until SYS_CNT
 * has advanced by N ticks.  Pre-computed target (do-while) saves two
 * instructions per iteration vs. the elapsed-subtraction loop.
 */

#include "kernel_operator.h"
#include "sleep_tiling_data.h"
#include "sleep_tiling_key.h"

using namespace AscendC;

template <uint32_t schMode>
__global__ __aicore__ void sleep(GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SleepTilingData);
    GET_TILING_DATA_WITH_STRUCT(SleepTilingData, tilingData, tiling);

    int64_t cycles = tilingData.cycles;
    if (cycles > 0) {
        uint64_t start = static_cast<uint64_t>(AscendC::GetSystemCycle());
        uint64_t target = start + static_cast<uint64_t>(cycles);
        do {
            // nothing — SYS_CNT advances on its own
        } while (static_cast<uint64_t>(AscendC::GetSystemCycle()) < target);
    }
}
