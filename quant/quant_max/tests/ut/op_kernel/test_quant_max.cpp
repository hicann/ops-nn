/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_max.cpp
 * \brief Unit test for QuantMax op_kernel
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "aclrt_launcher.h"
#include "data_utils.h"

// Tiling data structure (matching QuantMaxTilingData)
struct TestTilingData {
    int64_t numCore;
    int64_t dim0;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t baseLen;
    int64_t roundMode;
};

extern "C" __global__ __aicore__ void quant_max(GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR amax, GM_ADDR workspace, GM_ADDR tiling);

class quant_max_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "quant_max_test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "quant_max_test TearDown" << std::endl;
    }
};

// Helper function to generate test data
template<typename T>
void GenerateInputData(void* buffer, size_t size, float minValue = -1.0f, float maxValue = 1.0f)
{
    T* data = static_cast<T*>(buffer);
    size_t count = size / sizeof(T);
    for (size_t i = 0; i < count; i++) {
        float val = minValue + (maxValue - minValue) * (static_cast<float>(i % 1000) / 1000.0f);
        if constexpr (std::is_same_v<T, float>) {
            data[i] = val;
        } else if constexpr (std::is_same_v<T, half>) {
            data[i] = static_cast<half>(val);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            data[i] = static_cast<bfloat16_t>(val);
        }
    }
}

template<typename T>
float ComputeAmax(const T* data, size_t count)
{
    float amax = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float val;
        if constexpr (std::is_same_v<T, float>) {
            val = std::abs(data[i]);
        } else if constexpr (std::is_same_v<T, half>) {
            val = std::abs(static_cast<float>(data[i]));
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            val = std::abs(static_cast<float>(data[i]));
        }
        if (val > amax) {
            amax = val;
        }
    }
    return amax;
}

// Test FP32 to FP8_E5M2 - 1D input
TEST_F(quant_max_test, test_fp32_to_fp8_e5m2_1d)
{
    size_t elementCount = 4096;
    size_t xSize = elementCount * sizeof(float);
    size_t ySize = elementCount * sizeof(uint8_t);  // FP8
    size_t amaxSize = sizeof(float);
    size_t scaleSize = sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    // Allocate memory
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    // Generate input data
    GenerateInputData<float>(x, xSize, -1.0f, 1.0f);
    float scaleVal = 1.0f;
    memcpy(scale, &scaleVal, scaleSize);

    // Setup tiling data
    size_t tilingSize = sizeof(TestTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestTilingData tilingData;
    tilingData.numCore = 1;
    tilingData.dim0 = elementCount;
    tilingData.blockFactor = elementCount;
    tilingData.blockTailFactor = 0;
    tilingData.baseLen = 1;
    tilingData.roundMode = 0;  // Rint
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(quant_max, blockDim, x, scale, y, amax, workspace, tiling);

    // Verify output
    float* amaxOut = reinterpret_cast<float*>(amax);
    float expectedAmax = ComputeAmax(reinterpret_cast<float*>(x), elementCount);
    EXPECT_NEAR(*amaxOut, expectedAmax, 0.01f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test FP32 to FP8_E5M2 - 2D input
TEST_F(quant_max_test, test_fp32_to_fp8_e5m2_2d)
{
    size_t elementCount = 128 * 256;
    size_t xSize = elementCount * sizeof(float);
    size_t ySize = elementCount * sizeof(uint8_t);
    size_t amaxSize = sizeof(float);
    size_t scaleSize = sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateInputData<float>(x, xSize, -2.0f, 2.0f);
    float scaleVal = 0.5f;
    memcpy(scale, &scaleVal, scaleSize);

    size_t tilingSize = sizeof(TestTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestTilingData tilingData;
    tilingData.numCore = 4;
    tilingData.dim0 = elementCount;
    tilingData.blockFactor = elementCount / 4;
    tilingData.blockTailFactor = 0;
    tilingData.baseLen = 1;
    tilingData.roundMode = 0;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(quant_max, blockDim, x, scale, y, amax, workspace, tiling);

    float* amaxOut = reinterpret_cast<float*>(amax);
    float expectedAmax = ComputeAmax(reinterpret_cast<float*>(x), elementCount);
    EXPECT_NEAR(*amaxOut, expectedAmax, 0.01f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test FP32 to FP8_E5M2 - 3D input
TEST_F(quant_max_test, test_fp32_to_fp8_e5m2_3d)
{
    size_t elementCount = 16 * 64 * 128;
    size_t xSize = elementCount * sizeof(float);
    size_t ySize = elementCount * sizeof(uint8_t);
    size_t amaxSize = sizeof(float);
    size_t scaleSize = sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 8;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateInputData<float>(x, xSize, -1.0f, 1.0f);
    float scaleVal = 1.0f;
    memcpy(scale, &scaleVal, scaleSize);

    size_t tilingSize = sizeof(TestTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestTilingData tilingData;
    tilingData.numCore = 8;
    tilingData.dim0 = elementCount;
    tilingData.blockFactor = elementCount / 8;
    tilingData.blockTailFactor = 0;
    tilingData.baseLen = 1;
    tilingData.roundMode = 0;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(quant_max, blockDim, x, scale, y, amax, workspace, tiling);

    float* amaxOut = reinterpret_cast<float*>(amax);
    float expectedAmax = ComputeAmax(reinterpret_cast<float*>(x), elementCount);
    EXPECT_NEAR(*amaxOut, expectedAmax, 0.01f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test FP32 to FP8_E5M2 - 4D input
TEST_F(quant_max_test, test_fp32_to_fp8_e5m2_4d)
{
    size_t elementCount = 2 * 4 * 8 * 16;
    size_t xSize = elementCount * sizeof(float);
    size_t ySize = elementCount * sizeof(uint8_t);
    size_t amaxSize = sizeof(float);
    size_t scaleSize = sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateInputData<float>(x, xSize, -3.0f, 3.0f);
    float scaleVal = 0.33f;
    memcpy(scale, &scaleVal, scaleSize);

    size_t tilingSize = sizeof(TestTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestTilingData tilingData;
    tilingData.numCore = 4;
    tilingData.dim0 = elementCount;
    tilingData.blockFactor = elementCount / 4;
    tilingData.blockTailFactor = 0;
    tilingData.baseLen = 1;
    tilingData.roundMode = 0;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(quant_max, blockDim, x, scale, y, amax, workspace, tiling);

    float* amaxOut = reinterpret_cast<float*>(amax);
    float expectedAmax = ComputeAmax(reinterpret_cast<float*>(x), elementCount);
    EXPECT_NEAR(*amaxOut, expectedAmax, 0.01f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test with Round mode
TEST_F(quant_max_test, test_round_mode)
{
    size_t elementCount = 1024;
    size_t xSize = elementCount * sizeof(float);
    size_t ySize = elementCount * sizeof(uint8_t);
    size_t amaxSize = sizeof(float);
    size_t scaleSize = sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateInputData<float>(x, xSize, -1.0f, 1.0f);
    float scaleVal = 1.0f;
    memcpy(scale, &scaleVal, scaleSize);

    size_t tilingSize = sizeof(TestTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestTilingData tilingData;
    tilingData.numCore = 1;
    tilingData.dim0 = elementCount;
    tilingData.blockFactor = elementCount;
    tilingData.blockTailFactor = 0;
    tilingData.baseLen = 1;
    tilingData.roundMode = 1;  // Round mode
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(quant_max, blockDim, x, scale, y, amax, workspace, tiling);

    float* amaxOut = reinterpret_cast<float*>(amax);
    float expectedAmax = ComputeAmax(reinterpret_cast<float*>(x), elementCount);
    EXPECT_NEAR(*amaxOut, expectedAmax, 0.01f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test FP16 input
TEST_F(quant_max_test, test_fp16_to_fp8_e5m2)
{
    size_t elementCount = 512;
    size_t xSize = elementCount * sizeof(half);
    size_t ySize = elementCount * sizeof(uint8_t);
    size_t amaxSize = sizeof(half);
    size_t scaleSize = sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateInputData<half>(x, xSize, -1.0f, 1.0f);
    float scaleVal = 1.0f;
    memcpy(scale, &scaleVal, scaleSize);

    size_t tilingSize = sizeof(TestTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestTilingData tilingData;
    tilingData.numCore = 1;
    tilingData.dim0 = elementCount;
    tilingData.blockFactor = elementCount;
    tilingData.blockTailFactor = 0;
    tilingData.baseLen = 1;
    tilingData.roundMode = 0;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(quant_max, blockDim, x, scale, y, amax, workspace, tiling);

    half* amaxOut = reinterpret_cast<half*>(amax);
    float expectedAmax = ComputeAmax(reinterpret_cast<half*>(x), elementCount);
    EXPECT_NEAR(static_cast<float>(*amaxOut), expectedAmax, 0.05f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

// Test BF16 input
TEST_F(quant_max_test, test_bf16_to_fp8_e5m2)
{
    size_t elementCount = 256;
    size_t xSize = elementCount * sizeof(bfloat16_t);
    size_t ySize = elementCount * sizeof(uint8_t);
    size_t amaxSize = sizeof(bfloat16_t);
    size_t scaleSize = sizeof(float);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);

    GenerateInputData<bfloat16_t>(x, xSize, -1.0f, 1.0f);
    float scaleVal = 1.0f;
    memcpy(scale, &scaleVal, scaleSize);

    size_t tilingSize = sizeof(TestTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    TestTilingData tilingData;
    tilingData.numCore = 1;
    tilingData.dim0 = elementCount;
    tilingData.blockFactor = elementCount;
    tilingData.blockTailFactor = 0;
    tilingData.baseLen = 1;
    tilingData.roundMode = 0;
    memcpy(tiling, &tilingData, tilingSize);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(quant_max, blockDim, x, scale, y, amax, workspace, tiling);

    bfloat16_t* amaxOut = reinterpret_cast<bfloat16_t*>(amax);
    float expectedAmax = ComputeAmax(reinterpret_cast<bfloat16_t*>(x), elementCount);
    EXPECT_NEAR(static_cast<float>(*amaxOut), expectedAmax, 0.05f);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)amax);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}