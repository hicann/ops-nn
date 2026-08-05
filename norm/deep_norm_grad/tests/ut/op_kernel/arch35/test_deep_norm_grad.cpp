/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "arch35/deep_norm_grad.h"

namespace {

template <typename T>
void RunDeepNormGradArch35Kernel(GM_ADDR dy, GM_ADDR x, GM_ADDR gx, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                 GM_ADDR dx, GM_ADDR dgx, GM_ADDR dbeta, GM_ADDR dgamma, GM_ADDR workspace,
                                 GM_ADDR tiling)
{
    auto tilingData = reinterpret_cast<const DeepNormGradTilingDataArch35*>(tiling);
    DeepNormGradArch35::DeepNormGrad<T> op;
    op.Init(dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, workspace, tilingData);
    op.Process();
}

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t VL_FP32 = 256 / sizeof(float);

struct KernelCase {
    uint32_t rows;
    uint32_t cols;
    uint32_t tileLength;
    uint32_t maxBackwardCores;
    uint32_t maxGammaBetaCores;
    float dxTolerance;
    float dgxTolerance;
    float dbetaTolerance;
    float dgammaTolerance;
    bool rowSplit = false;
};

uint32_t CeilDiv(uint32_t value, uint32_t divisor) { return (value + divisor - 1) / divisor; }

uint32_t AlignUp(uint32_t value, uint32_t alignment) { return CeilDiv(value, alignment) * alignment; }

template <typename T>
void RunKernelCase(const KernelCase& testCase)
{
    constexpr float alpha = 0.3f;
    constexpr float eps = 1e-6f;
    const uint32_t rows = testCase.rows;
    const uint32_t cols = testCase.cols;
    const uint32_t total = rows * cols;
    ASSERT_GT(rows, 0);
    ASSERT_GT(cols, 0);
    ASSERT_GE(testCase.tileLength, VL_FP32);
    ASSERT_EQ(testCase.tileLength % VL_FP32, 0);

    auto dy = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto gx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto gamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(T)));
    auto mean = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto rstd = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto dx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto dgx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(T)));
    auto dbeta = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(float)));
    auto dgamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(float)));
    uint32_t smallColsAlign = AlignUp(cols, VL_FP32);
    uint64_t workspaceBytes = testCase.rowSplit ? static_cast<uint64_t>(testCase.maxBackwardCores) * 2 *
                                                      smallColsAlign * sizeof(float) :
                                                  BLOCK_SIZE;
    auto workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceBytes));
    auto tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(DeepNormGradTilingDataArch35)));

    auto dyData = reinterpret_cast<T*>(dy);
    auto xData = reinterpret_cast<T*>(x);
    auto gxData = reinterpret_cast<T*>(gx);
    auto gammaData = reinterpret_cast<T*>(gamma);
    auto meanData = reinterpret_cast<float*>(mean);
    auto rstdData = reinterpret_cast<float*>(rstd);
    for (uint32_t i = 0; i < total; ++i) {
        dyData[i] = static_cast<T>(static_cast<float>(static_cast<int32_t>(i % 11) - 5) * 0.02f);
        xData[i] = static_cast<T>(static_cast<float>(static_cast<int32_t>(i % 17) - 8) * 0.03125f);
        gxData[i] = static_cast<T>(static_cast<float>(static_cast<int32_t>(i % 13) - 6) * 0.015625f);
    }
    for (uint32_t c = 0; c < cols; ++c) {
        gammaData[c] = static_cast<T>(0.75f + static_cast<float>(c % 11) * 0.01f);
    }

    std::vector<float> centered(total, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            sum += alpha * static_cast<float>(xData[idx]) + static_cast<float>(gxData[idx]);
        }
        meanData[r] = sum / static_cast<float>(cols);
        float var = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            centered[idx] = alpha * static_cast<float>(xData[idx]) + static_cast<float>(gxData[idx]) - meanData[r];
            var += centered[idx] * centered[idx];
        }
        rstdData[r] = 1.0f / std::sqrt(var / static_cast<float>(cols) + eps);
    }

    std::vector<float> expectDx(total, 0.0f);
    std::vector<float> expectDgx(total, 0.0f);
    std::vector<float> expectDbeta(cols, 0.0f);
    std::vector<float> expectDgamma(cols, 0.0f);
    const float invCols = 1.0f / static_cast<float>(cols);
    for (uint32_t r = 0; r < rows; ++r) {
        float sumTmp = 0.0f;
        float sumTmpNorm = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float dyGamma = static_cast<float>(dyData[idx]) * static_cast<float>(gammaData[c]);
            sumTmp += dyGamma * centered[idx] * rstdData[r] * rstdData[r] * rstdData[r];
            sumTmpNorm += dyGamma * rstdData[r];
        }
        float avgTmp = -invCols * sumTmp;
        float avgTmpNorm = -invCols * sumTmpNorm;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float dyValue = static_cast<float>(dyData[idx]);
            float dyGamma = dyValue * static_cast<float>(gammaData[c]);
            expectDgx[idx] = dyGamma * rstdData[r] + centered[idx] * avgTmp + avgTmpNorm;
            expectDx[idx] = expectDgx[idx] * alpha;
            expectDbeta[c] += dyValue;
            expectDgamma[c] += dyValue * centered[idx] * rstdData[r];
        }
    }

    uint32_t rowsPerCore = CeilDiv(rows, testCase.maxBackwardCores);
    uint32_t smallRowStride = 0;
    uint32_t smallRowsPerTile = 0;
    if (testCase.rowSplit) {
        uint32_t rowBytes = cols * sizeof(T);
        uint32_t rowsAlignment = BLOCK_SIZE / std::gcd(rowBytes, BLOCK_SIZE);
        rowsPerCore = AlignUp(rowsPerCore, rowsAlignment);
        smallRowStride = AlignUp(rowBytes, BLOCK_SIZE) / sizeof(T);
        smallRowsPerTile = std::max(1U, testCase.tileLength / smallRowStride);
    }
    uint32_t backwardBlockDim = CeilDiv(rows, rowsPerCore);
    uint32_t blockElements = BLOCK_SIZE / sizeof(T);
    uint32_t colsPerCore = AlignUp(CeilDiv(cols, testCase.maxGammaBetaCores), blockElements);
    uint32_t gammaBetaBlockDim = CeilDiv(cols, colsPerCore);
    if (testCase.rowSplit) {
        gammaBetaBlockDim = backwardBlockDim;
    }
    uint32_t blockDim = std::max(backwardBlockDim, gammaBetaBlockDim);
    auto tilingData = reinterpret_cast<DeepNormGradTilingDataArch35*>(tiling);
    tilingData->numRows = rows;
    tilingData->numCols = cols;
    tilingData->rowsPerCore = rowsPerCore;
    tilingData->colsPerCore = colsPerCore;
    tilingData->backwardBlockDim = backwardBlockDim;
    tilingData->gammaBetaBlockDim = gammaBetaBlockDim;
    tilingData->tileLength = testCase.tileLength;
    tilingData->tileLengthAlign = testCase.tileLength;
    tilingData->alpha = alpha;
    tilingData->invCols = invCols;
    tilingData->gammaBetaRowSplit = static_cast<uint32_t>(testCase.rowSplit);
    tilingData->smallRowStride = smallRowStride;
    tilingData->smallRowsPerTile = smallRowsPerTile;
    tilingData->smallColsAlign = testCase.rowSplit ? smallColsAlign : 0;

    auto dxData = reinterpret_cast<T*>(dx);
    auto dgxData = reinterpret_cast<T*>(dgx);
    auto dbetaData = reinterpret_cast<float*>(dbeta);
    auto dgammaData = reinterpret_cast<float*>(dgamma);
    for (uint32_t i = 0; i < total; ++i) {
        dxData[i] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
        dgxData[i] = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    }
    for (uint32_t c = 0; c < cols; ++c) {
        dbetaData[c] = std::numeric_limits<float>::quiet_NaN();
        dgammaData[c] = std::numeric_limits<float>::quiet_NaN();
    }

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(RunDeepNormGradArch35Kernel<T>, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma,
                workspace, tiling);

    for (uint32_t i = 0; i < total; ++i) {
        EXPECT_NEAR(static_cast<float>(dxData[i]), expectDx[i], testCase.dxTolerance) << "dx index " << i;
        EXPECT_NEAR(static_cast<float>(dgxData[i]), expectDgx[i], testCase.dgxTolerance) << "dgx index " << i;
    }
    for (uint32_t c = 0; c < cols; ++c) {
        EXPECT_NEAR(dbetaData[c], expectDbeta[c], testCase.dbetaTolerance) << "dbeta index " << c;
        EXPECT_NEAR(dgammaData[c], expectDgamma[c], testCase.dgammaTolerance) << "dgamma index " << c;
    }

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace

TEST(DeepNormGradKernelArch35, Fp32TailTileHas65Elements)
{
    RunKernelCase<float>({3, 193, 128, 2, 2, 4e-3f, 1e-2f, 2e-3f, 2e-3f});
}

TEST(DeepNormGradKernelArch35, Fp32ProductionTileAndLargeD)
{
    RunKernelCase<float>({2, 8193, 4096, 2, 2, 8e-3f, 2e-2f, 3e-3f, 5e-3f});
}

TEST(DeepNormGradKernelArch35, Fp16MergeNShape)
{
    RunKernelCase<half>({5, 1000, 4096, 3, 2, 3e-2f, 5e-2f, 1e-2f, 2e-2f});
}

TEST(DeepNormGradKernelArch35, Bf16LargeNSmallDShape)
{
    RunKernelCase<bfloat16_t>({73, 257, 128, 8, 2, 6e-2f, 8e-2f, 3e-2f, 5e-2f});
}

TEST(DeepNormGradKernelArch35, Fp32SmallDRowSplit)
{
    RunKernelCase<float>({129, 7, 64, 4, 2, 5e-3f, 1e-2f, 2e-3f, 3e-3f, true});
}

TEST(DeepNormGradKernelArch35, Fp32SingleColumnRowSplit)
{
    RunKernelCase<float>({257, 1, 64, 4, 2, 5e-3f, 1e-2f, 2e-3f, 3e-3f, true});
}

TEST(DeepNormGradKernelArch35, Fp16TwoColumnRowSplit)
{
    RunKernelCase<half>({257, 2, 64, 4, 2, 3e-2f, 5e-2f, 1e-2f, 2e-2f, true});
}

TEST(DeepNormGradKernelArch35, Bf16TwoColumnRowSplit)
{
    RunKernelCase<bfloat16_t>({257, 2, 64, 4, 2, 6e-2f, 8e-2f, 3e-2f, 5e-2f, true});
}
