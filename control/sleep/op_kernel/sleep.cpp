/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sleep.cpp
 * \brief Sleep operator kernel — A5 (Ascend 950) SIMT busy-spin
 *
 * SIMT clock() via __cce_simt_get_CLOCK64(), directly analogous to CUDA
 * clock64().  User-supplied cycles are passed straight through — no
 * frequency conversion is needed in Tiling.
 * Launch: single thread (dim3(1)) via asc_vf_call, matching CUDA <<<1,1>>>.
 */

#include "kernel_operator.h"
#include "sleep_tiling_data.h"
#include "sleep_tiling_key.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(__CCE_KT_TEST__)
#include "simt_api/asc_simt.h"

using namespace AscendC;

__simt_vf__ __aicore__ LAUNCH_BOUND(1) inline void SleepSpin(int64_t cycles)
{
    uint64_t startClock = clock();
    uint64_t clockOffset = 0;
    while (clockOffset < static_cast<uint64_t>(cycles)) {
        clockOffset = clock() - startClock;
    }
}
#endif

template <uint32_t schMode>
__global__ __aicore__ void sleep(GM_ADDR cycles, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SleepTilingData);
    GET_TILING_DATA_WITH_STRUCT(SleepTilingData, tilingData, tiling);

    int64_t sleepCycles = tilingData.cycles;
    if (sleepCycles <= 0) {
        return;
    }

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(__CCE_KT_TEST__)
    asc_vf_call<SleepSpin>(dim3(1), sleepCycles);
#endif
}
