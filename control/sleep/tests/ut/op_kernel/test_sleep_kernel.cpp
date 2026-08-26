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
 * \file test_sleep_kernel.cpp
 * \brief Sleep kernel UT — CPU stub 模式
 *
 * 注意：sleep 的核心功能（SIMT clock() 忙等待）在 __CCE_KT_TEST__ 下不编译，
 * CPU stub 模式只能覆盖：
 *   1. 编译冒烟测试 — 验证 kernel 可为 ascend950 编译通过
 *   2. tiling 数据解析 — 验证 SleepTilingData 正确读取
 *   3. early return — 验证 cycles <= 0 时 kernel 安全返回
 * SIMT 忙等待行为需通过 op_api 上板测试验证。
 */

#include <vector>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"

#include "../../../op_kernel/sleep.cpp"
#include "../../../op_kernel/sleep_tiling_data.h"

using namespace std;

class sleep_kernel_test : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 正常 cycles：kernel 应正常执行（CPU stub 下 SIMT 路径不编译，仅验证不崩溃）
TEST_F(sleep_kernel_test, normal_cycles)
{
    int32_t blockDim = 1;
    size_t tilingSize = sizeof(SleepTilingData);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    SleepTilingData* tilingData = reinterpret_cast<SleepTilingData*>(tiling);
    tilingData->cycles = 1000000;

    uint8_t* cyclesGM = (uint8_t*)AscendC::GmAlloc(sizeof(int64_t));

    auto KernelSleep = [](GM_ADDR cycles, GM_ADDR workspace, GM_ADDR tiling) { ::sleep<0>(cycles, workspace, tiling); };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSleep, blockDim, cyclesGM, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(cyclesGM);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// cycles = 1：最小正整数，验证边界
TEST_F(sleep_kernel_test, minimum_cycles)
{
    int32_t blockDim = 1;
    size_t tilingSize = sizeof(SleepTilingData);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* cyclesGM = (uint8_t*)AscendC::GmAlloc(sizeof(int64_t));

    SleepTilingData* tilingData = reinterpret_cast<SleepTilingData*>(tiling);
    tilingData->cycles = 1;

    auto KernelSleep = [](GM_ADDR cycles, GM_ADDR workspace, GM_ADDR tiling) { ::sleep<0>(cycles, workspace, tiling); };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSleep, blockDim, cyclesGM, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(cyclesGM);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// cycles = 0：kernel 应 early return，不崩溃
TEST_F(sleep_kernel_test, zero_cycles_early_return)
{
    int32_t blockDim = 1;
    size_t tilingSize = sizeof(SleepTilingData);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* cyclesGM = (uint8_t*)AscendC::GmAlloc(sizeof(int64_t));

    SleepTilingData* tilingData = reinterpret_cast<SleepTilingData*>(tiling);
    tilingData->cycles = 0;

    auto KernelSleep = [](GM_ADDR cycles, GM_ADDR workspace, GM_ADDR tiling) { ::sleep<0>(cycles, workspace, tiling); };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSleep, blockDim, cyclesGM, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(cyclesGM);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// cycles = 负数：kernel 应 early return，不崩溃
TEST_F(sleep_kernel_test, negative_cycles_early_return)
{
    int32_t blockDim = 1;
    size_t tilingSize = sizeof(SleepTilingData);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint8_t* cyclesGM = (uint8_t*)AscendC::GmAlloc(sizeof(int64_t));

    SleepTilingData* tilingData = reinterpret_cast<SleepTilingData*>(tiling);
    tilingData->cycles = -100;

    auto KernelSleep = [](GM_ADDR cycles, GM_ADDR workspace, GM_ADDR tiling) { ::sleep<0>(cycles, workspace, tiling); };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSleep, blockDim, cyclesGM, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(cyclesGM);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
