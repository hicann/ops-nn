/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/apply_proximal_adagrad_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void apply_proximal_adagrad(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR l1,
                                                             GM_ADDR l2, GM_ADDR grad, GM_ADDR var_out,
                                                             GM_ADDR workspace, GM_ADDR tiling);

class ApplyProximalAdagradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ApplyProximalAdagradKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "ApplyProximalAdagradKernelTest TearDown" << endl; }
};

TEST_F(ApplyProximalAdagradKernelTest, test_fp32_basic)
{
    constexpr size_t numElements = 256;
    size_t dataSize = numElements * sizeof(float);
    size_t scalarSize = sizeof(float);
    size_t tilingSize = sizeof(ApplyProximalAdagradTilingData);
    uint32_t blockDim = 1;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* accum = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* lr = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* l1 = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* l2 = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    // 让 accum 不为 0 以避开 rsqrt(0) 奇点。
    auto* accumF = reinterpret_cast<float*>(accum);
    for (size_t i = 0; i < numElements; ++i) {
        accumF[i] = 1.0f;
    }
    *reinterpret_cast<float*>(lr) = 0.1f;
    *reinterpret_cast<float*>(l1) = 0.0f;
    *reinterpret_cast<float*>(l2) = 0.0f;

    auto* tilingData = reinterpret_cast<ApplyProximalAdagradTilingData*>(tiling);
    tilingData->totalElements = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(apply_proximal_adagrad, blockDim, var, accum, lr, l1, l2, grad, var_out, workspace, tiling);

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(lr);
    AscendC::GmFree(l1);
    AscendC::GmFree(l2);
    AscendC::GmFree(grad);
    AscendC::GmFree(var_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ApplyProximalAdagradKernelTest, test_fp32_large_with_ub_split)
{
    constexpr size_t numElements = 2048;
    size_t dataSize = numElements * sizeof(float);
    size_t scalarSize = sizeof(float);
    size_t tilingSize = sizeof(ApplyProximalAdagradTilingData);
    uint32_t blockDim = 1;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* accum = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* lr = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* l1 = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* l2 = (uint8_t*)AscendC::GmAlloc(scalarSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(dataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    auto* accumF = reinterpret_cast<float*>(accum);
    for (size_t i = 0; i < numElements; ++i) {
        accumF[i] = 1.0f;
    }
    *reinterpret_cast<float*>(lr) = 0.1f;
    // 走 l1 != 0 的通用 soft-threshold 分支。
    *reinterpret_cast<float*>(l1) = 0.01f;
    *reinterpret_cast<float*>(l2) = 0.0f;

    auto* tilingData = reinterpret_cast<ApplyProximalAdagradTilingData*>(tiling);
    tilingData->totalElements = numElements;
    tilingData->blockFactor = numElements;
    // 强制内层 UB 分块。
    tilingData->ubFactor = 1024;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(apply_proximal_adagrad, blockDim, var, accum, lr, l1, l2, grad, var_out, workspace, tiling);

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(lr);
    AscendC::GmFree(l1);
    AscendC::GmFree(l2);
    AscendC::GmFree(grad);
    AscendC::GmFree(var_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
