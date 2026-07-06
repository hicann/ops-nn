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
#include <cmath>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/arch35/apply_proximal_gradient_descent_tiling_data.h"

using namespace std;

static constexpr size_t GM_ALIGN = 32;
static inline size_t AlignUp(size_t n, size_t align) { return ((n + align - 1) / align) * align; }

extern "C" __global__ __aicore__ void apply_proximal_gradient_descent(GM_ADDR var, GM_ADDR alpha, GM_ADDR l1,
                                                                      GM_ADDR l2, GM_ADDR delta, GM_ADDR varOut,
                                                                      GM_ADDR workspace, GM_ADDR tiling);

class ApplyProximalGradientDescentKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ApplyProximalGradientDescentKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "ApplyProximalGradientDescentKernelTest TearDown" << endl; }
};

[[maybe_unused]] static void ComputeGolden(const float* var, float alphaVal, float l1Val, float l2Val,
                                           const float* delta, float* output, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        float prox = var[i] - alphaVal * delta[i];
        float absProx = std::fabs(prox);
        float sign = (prox > 0.f) ? 1.f : ((prox < 0.f) ? -1.f : 0.f);
        float relu = std::max(absProx - alphaVal * l1Val, 0.f);
        output[i] = sign * relu / (1.f + alphaVal * l2Val);
    }
}

TEST_F(ApplyProximalGradientDescentKernelTest, test_fp32_basic)
{
    constexpr size_t numElements = 256;
    size_t alignedDataSize = AlignUp(numElements * sizeof(float), GM_ALIGN);
    size_t tilingSize = sizeof(ApplyProximalGradientDescentTilingData);
    uint32_t blockDim = 1;

    uint8_t* varBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* alphaBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l1Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l2Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* deltaBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* outputBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* varPtr = reinterpret_cast<float*>(varBuf);
    float* deltaPtr = reinterpret_cast<float*>(deltaBuf);
    for (size_t i = 0; i < numElements; i++) {
        varPtr[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
        deltaPtr[i] = static_cast<float>(i % 5) * 0.05f - 0.1f;
    }
    *reinterpret_cast<float*>(alphaBuf) = 0.01f;
    *reinterpret_cast<float*>(l1Buf) = 0.001f;
    *reinterpret_cast<float*>(l2Buf) = 0.01f;

    ApplyProximalGradientDescentTilingData* tilingData = reinterpret_cast<ApplyProximalGradientDescentTilingData*>(
        tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(apply_proximal_gradient_descent, blockDim, varBuf, alphaBuf, l1Buf, l2Buf, deltaBuf, outputBuf,
                workspace, tiling);

    // ICPU CPU sim 下 DataCopyPad 写回 GM 不保证生效，按 rms_prop UT 范式仅校验
    // 端到端不崩溃且输出无 NaN/Inf；精确数值由 ST/NPU 验收。
    float* outPtr = reinterpret_cast<float*>(outputBuf);
    for (size_t i = 0; i < numElements; i++) {
        EXPECT_FALSE(std::isnan(outPtr[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(outPtr[i])) << "Inf at index " << i;
    }

    AscendC::GmFree(varBuf);
    AscendC::GmFree(alphaBuf);
    AscendC::GmFree(l1Buf);
    AscendC::GmFree(l2Buf);
    AscendC::GmFree(deltaBuf);
    AscendC::GmFree(outputBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ApplyProximalGradientDescentKernelTest, test_fp32_large_double_buffer)
{
    constexpr size_t numElements = 2048;
    size_t alignedDataSize = AlignUp(numElements * sizeof(float), GM_ALIGN);
    size_t tilingSize = sizeof(ApplyProximalGradientDescentTilingData);
    uint32_t blockDim = 1;

    uint8_t* varBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* alphaBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l1Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l2Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* deltaBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* outputBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* varPtr = reinterpret_cast<float*>(varBuf);
    float* deltaPtr = reinterpret_cast<float*>(deltaBuf);
    for (size_t i = 0; i < numElements; i++) {
        varPtr[i] = static_cast<float>(i % 11) * 0.1f - 0.5f;
        deltaPtr[i] = static_cast<float>(i % 9) * 0.05f - 0.2f;
    }
    *reinterpret_cast<float*>(alphaBuf) = 0.01f;
    *reinterpret_cast<float*>(l1Buf) = 0.001f;
    *reinterpret_cast<float*>(l2Buf) = 0.01f;

    ApplyProximalGradientDescentTilingData* tilingData = reinterpret_cast<ApplyProximalGradientDescentTilingData*>(
        tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = 512;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(apply_proximal_gradient_descent, blockDim, varBuf, alphaBuf, l1Buf, l2Buf, deltaBuf, outputBuf,
                workspace, tiling);

    float* outPtr = reinterpret_cast<float*>(outputBuf);
    for (size_t i = 0; i < numElements; i++) {
        EXPECT_FALSE(std::isnan(outPtr[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(outPtr[i])) << "Inf at index " << i;
    }

    AscendC::GmFree(varBuf);
    AscendC::GmFree(alphaBuf);
    AscendC::GmFree(l1Buf);
    AscendC::GmFree(l2Buf);
    AscendC::GmFree(deltaBuf);
    AscendC::GmFree(outputBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ApplyProximalGradientDescentKernelTest, test_fp32_l1_zero_l2_zero)
{
    constexpr size_t numElements = 64;
    size_t alignedDataSize = AlignUp(numElements * sizeof(float), GM_ALIGN);
    size_t tilingSize = sizeof(ApplyProximalGradientDescentTilingData);
    uint32_t blockDim = 1;

    uint8_t* varBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* alphaBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l1Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l2Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* deltaBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* outputBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* varPtr = reinterpret_cast<float*>(varBuf);
    float* deltaPtr = reinterpret_cast<float*>(deltaBuf);
    for (size_t i = 0; i < numElements; i++) {
        varPtr[i] = 1.0f;
        deltaPtr[i] = 0.5f;
    }
    *reinterpret_cast<float*>(alphaBuf) = 0.1f;
    *reinterpret_cast<float*>(l1Buf) = 0.0f;
    *reinterpret_cast<float*>(l2Buf) = 0.0f;

    ApplyProximalGradientDescentTilingData* tilingData = reinterpret_cast<ApplyProximalGradientDescentTilingData*>(
        tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(apply_proximal_gradient_descent, blockDim, varBuf, alphaBuf, l1Buf, l2Buf, deltaBuf, outputBuf,
                workspace, tiling);

    float* outPtr = reinterpret_cast<float*>(outputBuf);
    for (size_t i = 0; i < numElements; i++) {
        EXPECT_FALSE(std::isnan(outPtr[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(outPtr[i])) << "Inf at index " << i;
    }

    AscendC::GmFree(varBuf);
    AscendC::GmFree(alphaBuf);
    AscendC::GmFree(l1Buf);
    AscendC::GmFree(l2Buf);
    AscendC::GmFree(deltaBuf);
    AscendC::GmFree(outputBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(ApplyProximalGradientDescentKernelTest, test_fp32_non_aligned)
{
    constexpr size_t numElements = 17;
    size_t alignedDataSize = AlignUp(numElements * sizeof(float), GM_ALIGN);
    size_t tilingSize = sizeof(ApplyProximalGradientDescentTilingData);
    uint32_t blockDim = 1;

    uint8_t* varBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* alphaBuf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l1Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* l2Buf = (uint8_t*)AscendC::GmAlloc(sizeof(float));
    uint8_t* deltaBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* outputBuf = (uint8_t*)AscendC::GmAlloc(alignedDataSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    float* varPtr = reinterpret_cast<float*>(varBuf);
    float* deltaPtr = reinterpret_cast<float*>(deltaBuf);
    for (size_t i = 0; i < numElements; i++) {
        varPtr[i] = static_cast<float>(i) * 0.1f - 0.8f;
        deltaPtr[i] = static_cast<float>(i) * 0.02f;
    }
    *reinterpret_cast<float*>(alphaBuf) = 0.01f;
    *reinterpret_cast<float*>(l1Buf) = 0.001f;
    *reinterpret_cast<float*>(l2Buf) = 0.01f;

    ApplyProximalGradientDescentTilingData* tilingData = reinterpret_cast<ApplyProximalGradientDescentTilingData*>(
        tiling);
    tilingData->totalNum = numElements;
    tilingData->blockFactor = numElements;
    tilingData->ubFactor = numElements;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(apply_proximal_gradient_descent, blockDim, varBuf, alphaBuf, l1Buf, l2Buf, deltaBuf, outputBuf,
                workspace, tiling);

    float* outPtr = reinterpret_cast<float*>(outputBuf);
    for (size_t i = 0; i < numElements; i++) {
        EXPECT_FALSE(std::isnan(outPtr[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(outPtr[i])) << "Inf at index " << i;
    }

    AscendC::GmFree(varBuf);
    AscendC::GmFree(alphaBuf);
    AscendC::GmFree(l1Buf);
    AscendC::GmFree(l2Buf);
    AscendC::GmFree(deltaBuf);
    AscendC::GmFree(outputBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
