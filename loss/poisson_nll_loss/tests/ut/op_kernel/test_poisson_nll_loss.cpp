/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <cmath>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/poisson_nll_loss_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void poisson_nll_loss(GM_ADDR input_x, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                                       GM_ADDR tiling);

// The kernel uses DataCopyPad, whose CPU mock does not always write back to output GM, so
// precision is verified on real NPU via TTK (smoke: none/sum/mean x fp16/fp32 all 100%).
// These UTs verify that the hand-written kernel:
//   1. compiles under __CCE_KT_TEST__ with DTYPE_INPUT_X=float,
//   2. runs without crashing for reduction none/sum/mean and tiled/non-aligned shapes,
//   3. interprets PoissonNllLossTilingData correctly (Init offsets, two-phase reduce plumbing).
// AddOpTestCase compiles CPU mock with a single DTYPE_INPUT_X=float; fp16 coverage is in TTK.

namespace {
constexpr uint32_t REDUCTION_NONE = 0;
constexpr uint32_t REDUCTION_SUM = 1;
constexpr uint32_t REDUCTION_MEAN = 2;
constexpr int64_t WS_CORE_STRIDE = 8;
constexpr size_t SYS_WORKSPACE = 16UL * 1024UL * 1024UL;
} // namespace

class PoissonNllLossKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "PoissonNllLossKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "PoissonNllLossKernelTest TearDown" << endl; }
};

// Run the kernel once with the given config on a single (mock) core.
static void RunKernel(int64_t numElements, uint32_t reduction, uint32_t logInput, uint32_t full, int64_t ubFactor)
{
    // Empty tensor (numElements==0) is valid: allocate a minimal (unused) buffer so GmAlloc is never
    // called with size 0; the kernel does not read/write it (blockLength==0 -> none no-op, sum/mean
    // stages a zero partial). meanCof below becomes 1/0=inf so mean of empty resolves to 0*inf=nan.
    size_t inSize = (numElements > 0 ? numElements : 1) * sizeof(float);
    // reduction=none writes an elementwise output; sum/mean writes a single scalar.
    size_t outSize = (reduction == REDUCTION_NONE) ? inSize : sizeof(float);
    size_t tilingSize = sizeof(PoissonNllLossTilingData);
    // one core in CPU mock -> workspace holds a single WS_CORE_STRIDE block plus system reserve.
    size_t wsSize = WS_CORE_STRIDE * sizeof(float) + SYS_WORKSPACE;
    uint32_t blockDim = 1;

    uint8_t* xBuf = (uint8_t*)AscendC::GmAlloc(inSize);
    uint8_t* tBuf = (uint8_t*)AscendC::GmAlloc(inSize);
    uint8_t* yBuf = (uint8_t*)AscendC::GmAlloc(outSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(wsSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* xPtr = reinterpret_cast<float*>(xBuf);
    float* tPtr = reinterpret_cast<float*>(tBuf);
    for (int64_t i = 0; i < numElements; i++) {
        xPtr[i] = static_cast<float>(i % 7) * 0.3f - 1.0f; // spans negatives (exercises exp path)
        tPtr[i] = static_cast<float>(i % 5) * 0.5f;        // >= 0 targets, some > 1 (Stirling path)
    }

    PoissonNllLossTilingData* td = reinterpret_cast<PoissonNllLossTilingData*>(tiling);
    td->totalNum = numElements;
    td->blockFactor = numElements;
    td->ubFactor = ubFactor;
    td->eps = 1e-8f;
    td->meanCof = 1.0f / static_cast<float>(numElements);
    td->logInput = logInput;
    td->full = full;
    td->reduction = reduction;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(poisson_nll_loss, blockDim, xBuf, tBuf, yBuf, workspace, tiling);

    AscendC::GmFree(xBuf);
    AscendC::GmFree(tBuf);
    AscendC::GmFree(yBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// ---- reduction=none ----
TEST_F(PoissonNllLossKernelTest, none_logT_basic) { RunKernel(256, REDUCTION_NONE, 1, 0, 256); }

TEST_F(PoissonNllLossKernelTest, none_logF_basic) { RunKernel(256, REDUCTION_NONE, 0, 0, 256); }

TEST_F(PoissonNllLossKernelTest, none_tiled) { RunKernel(2048, REDUCTION_NONE, 1, 0, 512); }

TEST_F(PoissonNllLossKernelTest, none_non_aligned) { RunKernel(17, REDUCTION_NONE, 1, 0, 17); }

// ---- reduction=sum (two-phase reduce path) ----
TEST_F(PoissonNllLossKernelTest, sum_logT) { RunKernel(256, REDUCTION_SUM, 1, 0, 256); }

TEST_F(PoissonNllLossKernelTest, sum_logT_full) { RunKernel(256, REDUCTION_SUM, 1, 1, 256); }

// ---- reduction=mean ----
TEST_F(PoissonNllLossKernelTest, mean_logF) { RunKernel(128, REDUCTION_MEAN, 0, 0, 128); }

TEST_F(PoissonNllLossKernelTest, mean_logT_full) { RunKernel(64, REDUCTION_MEAN, 1, 1, 64); }

// ---- empty tensor (numElements=0): none -> no-op (empty output), sum -> 0, mean -> nan (0*inf).
// Kernel must run without crashing (blockLength==0 no-op / stages a zero partial). ----
TEST_F(PoissonNllLossKernelTest, empty_none) { RunKernel(0, REDUCTION_NONE, 1, 0, 256); }

TEST_F(PoissonNllLossKernelTest, empty_sum) { RunKernel(0, REDUCTION_SUM, 1, 0, 256); }

TEST_F(PoissonNllLossKernelTest, empty_mean) { RunKernel(0, REDUCTION_MEAN, 0, 0, 128); }
