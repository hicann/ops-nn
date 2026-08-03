/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#define DTYPE_X float
#include "arch35/deep_norm.cpp"
#undef DTYPE_X

namespace {

constexpr uint32_t VL_FP32 = 256 / sizeof(float);

template <typename T>
void RunDeepNormPartialLoadArch35Kernel(GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    auto tilingData = reinterpret_cast<const DeepNormTilingData*>(tiling);
    NsDeepNorm::DeepNormPartialLoad<T> op;
    op.Init(x, gx, beta, gamma, mean, rstd, y, tilingData);
    op.Process();
}

uint32_t AlignToVl(uint32_t value) { return (value + VL_FP32 - 1) / VL_FP32 * VL_FP32; }

uint32_t PowerSplit(uint32_t value)
{
    uint32_t power = VL_FP32;
    if (value > VL_FP32) {
        while (power < value) {
            power *= 2;
        }
        power /= 2;
    }
    return power;
}

struct PartialLoadCase {
    uint32_t rows;
    uint32_t cols;
    uint32_t tileLength;
    uint32_t blockDim;
    uint32_t rowPerCore;
    bool useProductionEntry;
    bool zeroVariance;
    float meanTolerance;
    float rstdTolerance;
    float yTolerance;
};

template <typename T>
void RunPartialLoadCase(const PartialLoadCase& testCase)
{
    constexpr float alpha = 0.3f;
    constexpr float eps = 1e-6f;
    uint32_t rows = testCase.rows;
    uint32_t cols = testCase.cols;
    const uint32_t total = rows * cols;

    auto x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto gx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto beta = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(T)));
    auto gamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(T)));
    auto mean = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto rstd = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(DeepNormTilingData)));

    auto xData = reinterpret_cast<T*>(x);
    auto gxData = reinterpret_cast<T*>(gx);
    auto betaData = reinterpret_cast<T*>(beta);
    auto gammaData = reinterpret_cast<T*>(gamma);
    for (uint32_t i = 0; i < total; ++i) {
        float xValue = testCase.zeroVariance ? 0.0f : static_cast<float>(static_cast<int32_t>(i % 17) - 8) * 0.03125f;
        float gxValue = testCase.zeroVariance ? 0.0f : static_cast<float>(static_cast<int32_t>(i % 13) - 6) * 0.015625f;
        xData[i] = static_cast<T>(xValue);
        gxData[i] = static_cast<T>(gxValue);
    }
    for (uint32_t i = 0; i < cols; ++i) {
        betaData[i] = static_cast<T>(static_cast<float>(static_cast<int32_t>(i % 7) - 3) * 0.02f);
        gammaData[i] = static_cast<T>(0.75f + static_cast<float>(i % 11) * 0.01f);
    }

    std::vector<float> expectMean(rows, 0.0f);
    std::vector<float> expectRstd(rows, 0.0f);
    std::vector<float> expectY(total, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            sum += alpha * static_cast<float>(xData[idx]) + static_cast<float>(gxData[idx]);
        }
        expectMean[r] = sum / static_cast<float>(cols);
        float variance = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float centered = alpha * static_cast<float>(xData[idx]) + static_cast<float>(gxData[idx]) - expectMean[r];
            variance += centered * centered;
        }
        expectRstd[r] = 1.0f / std::sqrt(variance / static_cast<float>(cols) + eps);
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float centered = alpha * static_cast<float>(xData[idx]) + static_cast<float>(gxData[idx]) - expectMean[r];
            expectY[idx] = centered * expectRstd[r] * static_cast<float>(gammaData[c]) +
                           static_cast<float>(betaData[c]);
        }
    }

    auto tilingData = reinterpret_cast<DeepNormTilingData*>(tiling);
    tilingData->numCore = testCase.blockDim;
    tilingData->numCol = cols;
    tilingData->numRow = rows;
    tilingData->rowPerCore = testCase.rowPerCore;
    tilingData->numColAlign = AlignToVl(cols);
    tilingData->powerSplit = PowerSplit(cols);
    tilingData->eps = eps;
    tilingData->alpha = alpha;
    tilingData->avgFactor = 1.0f / static_cast<float>(cols);
    tilingData->tileLength = testCase.tileLength;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if (testCase.useProductionEntry) {
        ASSERT_TRUE((std::is_same<T, float>::value));
        ICPU_RUN_KF(deep_norm, testCase.blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, tiling);
    } else {
        ICPU_RUN_KF(RunDeepNormPartialLoadArch35Kernel<T>, testCase.blockDim, x, gx, beta, gamma, mean, rstd, y,
                    workspace, tiling);
    }

    auto meanData = reinterpret_cast<float*>(mean);
    auto rstdData = reinterpret_cast<float*>(rstd);
    auto yData = reinterpret_cast<T*>(y);
    for (uint32_t r = 0; r < rows; ++r) {
        EXPECT_NEAR(meanData[r], expectMean[r], testCase.meanTolerance) << "row " << r;
        EXPECT_NEAR(rstdData[r], expectRstd[r], testCase.rstdTolerance) << "row " << r;
    }
    for (uint32_t i = 0; i < total; ++i) {
        EXPECT_NEAR(static_cast<float>(yData[i]), expectY[i], testCase.yTolerance) << "index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace

TEST(DeepNormKernelArch35, Fp32UnalignedReduceAxis)
{
    constexpr uint32_t rows = 2;
    constexpr uint32_t cols = 100;
    constexpr float alpha = 0.3f;
    constexpr float eps = 1e-6f;
    constexpr uint32_t blockDim = 2;
    const uint32_t total = rows * cols;

    auto x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto gx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto beta = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(float)));
    auto gamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(float)));
    auto mean = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto rstd = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(DeepNormTilingData)));

    auto xData = reinterpret_cast<float*>(x);
    auto gxData = reinterpret_cast<float*>(gx);
    auto betaData = reinterpret_cast<float*>(beta);
    auto gammaData = reinterpret_cast<float*>(gamma);
    for (uint32_t i = 0; i < total; ++i) {
        xData[i] = static_cast<float>(static_cast<int32_t>(i % 17) - 8) * 0.03125f;
        gxData[i] = static_cast<float>(static_cast<int32_t>(i % 13) - 6) * 0.015625f;
    }
    for (uint32_t i = 0; i < cols; ++i) {
        betaData[i] = static_cast<float>(static_cast<int32_t>(i % 7) - 3) * 0.02f;
        gammaData[i] = 0.75f + static_cast<float>(i % 11) * 0.01f;
    }

    std::vector<float> expectMean(rows, 0.0f);
    std::vector<float> expectRstd(rows, 0.0f);
    std::vector<float> expectY(total, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            sum += alpha * xData[idx] + gxData[idx];
        }
        expectMean[r] = sum / static_cast<float>(cols);
        float var = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float centered = alpha * xData[idx] + gxData[idx] - expectMean[r];
            var += centered * centered;
        }
        expectRstd[r] = 1.0f / std::sqrt(var / static_cast<float>(cols) + eps);
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float centered = alpha * xData[idx] + gxData[idx] - expectMean[r];
            expectY[idx] = centered * expectRstd[r] * gammaData[c] + betaData[c];
        }
    }

    auto tilingData = reinterpret_cast<DeepNormTilingData*>(tiling);
    tilingData->numCore = blockDim;
    tilingData->numCol = cols;
    tilingData->numRow = rows;
    tilingData->rowPerCore = 1;
    tilingData->numColAlign = AlignToVl(cols);
    tilingData->powerSplit = PowerSplit(cols);
    tilingData->eps = eps;
    tilingData->alpha = alpha;
    tilingData->avgFactor = 1.0f / static_cast<float>(cols);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(deep_norm, blockDim, x, gx, beta, gamma, mean, rstd, y, workspace, tiling);

    auto meanData = reinterpret_cast<float*>(mean);
    auto rstdData = reinterpret_cast<float*>(rstd);
    auto yData = reinterpret_cast<float*>(y);
    for (uint32_t r = 0; r < rows; ++r) {
        EXPECT_NEAR(meanData[r], expectMean[r], 1e-4f);
        EXPECT_NEAR(rstdData[r], expectRstd[r], 2e-3f);
    }
    for (uint32_t i = 0; i < total; ++i) {
        EXPECT_NEAR(yData[i], expectY[i], 3e-3f) << "index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(beta);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST(DeepNormKernelArch35, Fp32PartialLoadMultiTile)
{
    RunPartialLoadCase<float>({5, 385, 128, 3, 2, true, false, 2e-4f, 5e-3f, 6e-3f});
}

TEST(DeepNormKernelArch35, Fp32PartialLoadProductionTile)
{
    RunPartialLoadCase<float>({1, 8193, 4096, 1, 1, true, false, 2e-4f, 5e-3f, 6e-3f});
}

TEST(DeepNormKernelArch35, Fp16PartialLoadProductionTile)
{
    RunPartialLoadCase<half>({1, 15361, 4096, 1, 1, false, false, 2e-3f, 2e-2f, 3e-2f});
}

TEST(DeepNormKernelArch35, Bf16PartialLoadProductionTile)
{
    RunPartialLoadCase<bfloat16_t>({1, 15361, 4096, 1, 1, false, false, 3e-3f, 3e-2f, 4e-2f});
}

TEST(DeepNormKernelArch35, Bf16PartialLoadZeroVariance)
{
    RunPartialLoadCase<bfloat16_t>({1, 257, 128, 1, 1, false, true, 1e-5f, 2e-2f, 2e-2f});
}
