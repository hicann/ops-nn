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
 * \file test_cosine_embedding_loss.cpp
 * \brief CosineEmbeddingLoss arch35 kernel UT.
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "tikicpulib.h"

#include "../cosine_embedding_loss_tiling_def.h"
#include "../../../../op_kernel/arch35/cosine_embedding_loss.cpp"

namespace {
constexpr uint64_t WORKSPACE_SIZE = 32ULL * 1024ULL * 1024ULL;

int64_t Product(const std::vector<int64_t>& dims)
{
    int64_t result = 1;
    for (auto dim : dims) {
        result *= dim;
    }
    return result;
}

void FillStrides(int64_t dst[COSINE_EMBEDDING_LOSS_MAX_RANK], const std::vector<int64_t>& src)
{
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = src[i];
    }
}

void FillCosineTiling(CosineEmbeddingLossTilingData& tilingData, const std::vector<int64_t>& outputShape,
                      int64_t reduceDim, const std::vector<int64_t>& x1OutStrides,
                      const std::vector<int64_t>& x2OutStrides, const std::vector<int64_t>& targetOutStrides,
                      int64_t x1ReduceStride, int64_t x2ReduceStride, uint32_t reduction, float margin,
                      uint32_t fastPath, int64_t featureTile, int64_t ubTileRows = 1, int64_t x1Num = 0,
                      int64_t x2Num = 0, int64_t targetNum = 0)
{
    tilingData = {};
    tilingData.n = Product(outputShape);
    tilingData.d = reduceDim;
    tilingData.dAlign = ((reduceDim + 15) / 16) * 16;
    tilingData.x1Num = x1Num;
    tilingData.x2Num = x2Num;
    tilingData.targetNum = targetNum;
    tilingData.rowsPerCore = tilingData.n;
    tilingData.tailRows = tilingData.n;
    tilingData.usedCoreNum = 1;
    tilingData.ubTileRows = ubTileRows;
    tilingData.featureTile = featureTile;
    tilingData.reduceTmpBytes = 0;
    tilingData.reduction = reduction;
    tilingData.fastPath = fastPath;
    tilingData.outputRank = static_cast<uint32_t>(outputShape.size());
    tilingData.xBroadcastRank = static_cast<uint32_t>(outputShape.size() + 1);
    tilingData.margin = margin;
    tilingData.meanCoef = (reduction == 2) ? (1.0f / static_cast<float>(tilingData.n)) : 1.0f;
    tilingData.eps = 1e-12f;
    FillStrides(tilingData.outputShape, outputShape);
    FillStrides(tilingData.x1OutStrides, x1OutStrides);
    FillStrides(tilingData.x2OutStrides, x2OutStrides);
    FillStrides(tilingData.targetOutStrides, targetOutStrides);
    tilingData.x1ReduceStride = x1ReduceStride;
    tilingData.x2ReduceStride = x2ReduceStride;
}

void RunCosineKernel(const std::vector<float>& x1Host, const std::vector<float>& x2Host,
                     const std::vector<float>& targetHost, const std::vector<int64_t>& outputShape, int64_t reduceDim,
                     const std::vector<int64_t>& x1OutStrides, const std::vector<int64_t>& x2OutStrides,
                     const std::vector<int64_t>& targetOutStrides, int64_t x1ReduceStride, int64_t x2ReduceStride,
                     uint32_t reduction, float margin, std::vector<float>& yHost,
                     uint32_t fastPath = COSINE_EMBEDDING_LOSS_GENERIC_PATH, int64_t featureTile = 0,
                     int64_t ubTileRows = 1)
{
    const size_t x1Bytes = x1Host.size() * sizeof(float);
    const size_t x2Bytes = x2Host.size() * sizeof(float);
    const size_t targetBytes = targetHost.size() * sizeof(float);
    const size_t outputBytes = yHost.size() * sizeof(float);
    const size_t tilingBytes = sizeof(CosineEmbeddingLossTilingData);

    auto* x1 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(x1Bytes));
    auto* x2 = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(x2Bytes));
    auto* target = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(targetBytes));
    auto* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outputBytes));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(WORKSPACE_SIZE));
    auto* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingBytes));

    std::memcpy(x1, x1Host.data(), x1Bytes);
    std::memcpy(x2, x2Host.data(), x2Bytes);
    std::memcpy(target, targetHost.data(), targetBytes);
    std::memset(y, 0, outputBytes);
    std::memset(workspace, 0, WORKSPACE_SIZE);

    auto* tilingData = reinterpret_cast<CosineEmbeddingLossTilingData*>(tiling);
    FillCosineTiling(*tilingData, outputShape, reduceDim, x1OutStrides, x2OutStrides, targetOutStrides, x1ReduceStride,
                     x2ReduceStride, reduction, margin, fastPath, featureTile, ubTileRows,
                     static_cast<int64_t>(x1Host.size()), static_cast<int64_t>(x2Host.size()),
                     static_cast<int64_t>(targetHost.size()));

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(cosine_embedding_loss, 1, x1, x2, target, y, workspace, tiling);

    std::memcpy(yHost.data(), y, outputBytes);

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(target);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

class CosineEmbeddingLossArch35Kernel : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CosineEmbeddingLossArch35Kernel SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "CosineEmbeddingLossArch35Kernel TearDown" << std::endl; }
};

TEST_F(CosineEmbeddingLossArch35Kernel, none_fp32_two_rows)
{
    std::vector<float> x1 = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> x2 = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> target = {1.0f, -1.0f};
    std::vector<float> y(2, 0.0f);

    RunCosineKernel(x1, x2, target, {2}, 3, {3}, {3}, {1}, 1, 1, 0, 0.2f, y);

    EXPECT_NEAR(y[0], 0.0f, 1e-4f);
    EXPECT_NEAR(y[1], 0.8f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, mean_fp32_two_rows)
{
    std::vector<float> x1 = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> x2 = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> target = {1.0f, -1.0f};
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {2}, 3, {3}, {3}, {1}, 1, 1, 2, 0.2f, y);

    EXPECT_NEAR(y[0], 0.4f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, none_fp32_ignores_non_pm_one_target)
{
    std::vector<float> x1 = {
        1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    };
    std::vector<float> x2 = {
        0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
    };
    std::vector<float> target = {2.0f, 0.0f, -2.0f, 1.0f};
    std::vector<float> y(4, 0.0f);

    RunCosineKernel(x1, x2, target, {4}, 2, {2}, {2}, {1}, 1, 1, 0, 0.2f, y);

    EXPECT_NEAR(y[0], 0.0f, 1e-4f);
    EXPECT_NEAR(y[1], 0.0f, 1e-4f);
    EXPECT_NEAR(y[2], 0.0f, 1e-4f);
    EXPECT_NEAR(y[3], 0.0f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, none_fp32_axis1_3d)
{
    std::vector<float> x1 = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
    };
    std::vector<float> x2 = x1;
    std::vector<float> target = {1.0f, -1.0f, 2.0f, 1.0f};
    std::vector<float> y(4, 0.0f);

    RunCosineKernel(x1, x2, target, {2, 2}, 3, {6, 1}, {6, 1}, {2, 1}, 2, 2, 0, 0.2f, y);

    EXPECT_NEAR(y[0], 0.0f, 1e-4f);
    EXPECT_NEAR(y[1], 0.8f, 1e-4f);
    EXPECT_NEAR(y[2], 0.0f, 1e-4f);
    EXPECT_NEAR(y[3], 0.0f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, none_fp32_broadcast_x_and_target)
{
    std::vector<float> x1(6, 1.0f);            // shape [1, 3, 2]
    std::vector<float> x2(6, 1.0f);            // shape [2, 3, 1]
    std::vector<float> target = {1.0f, -1.0f}; // shape [2], broadcasts on output's last dim
    std::vector<float> y(4, 0.0f);

    RunCosineKernel(x1, x2, target, {2, 2}, 3, {0, 1}, {3, 0}, {0, 1}, 2, 1, 0, 0.2f, y);

    EXPECT_NEAR(y[0], 0.0f, 1e-4f);
    EXPECT_NEAR(y[1], 0.8f, 1e-4f);
    EXPECT_NEAR(y[2], 0.0f, 1e-4f);
    EXPECT_NEAR(y[3], 0.8f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, fast_none_fp32_feature_tail)
{
    constexpr int64_t featureSize = 70;
    std::vector<float> x1(2 * featureSize, 0.0f);
    std::vector<float> x2(2 * featureSize, 0.0f);
    x1[0] = 1.0f;
    x2[0] = 1.0f;
    x1[featureSize] = 1.0f;
    x2[featureSize + 1] = 1.0f;
    std::vector<float> target = {1.0f, -1.0f};
    std::vector<float> y(2, 0.0f);

    RunCosineKernel(x1, x2, target, {2}, featureSize, {featureSize}, {featureSize}, {1}, 1, 1, 0, 0.2f, y,
                    COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH, 64);

    EXPECT_NEAR(y[0], 0.0f, 1e-4f);
    EXPECT_NEAR(y[1], 0.0f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, fast_sum_and_mean_fp32)
{
    constexpr int64_t featureSize = 70;
    std::vector<float> x1(2 * featureSize, 0.0f);
    std::vector<float> x2(2 * featureSize, 0.0f);
    x1[0] = 1.0f;
    x2[0] = 1.0f;
    x1[featureSize] = 1.0f;
    x2[featureSize] = 1.0f;
    std::vector<float> target = {1.0f, -1.0f};
    std::vector<float> sumY(1, 0.0f);
    std::vector<float> meanY(1, 0.0f);

    RunCosineKernel(x1, x2, target, {2}, featureSize, {featureSize}, {featureSize}, {1}, 1, 1, 1, 0.2f, sumY,
                    COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH, 64);
    RunCosineKernel(x1, x2, target, {2}, featureSize, {featureSize}, {featureSize}, {1}, 1, 1, 2, 0.2f, meanY,
                    COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH, 64);

    EXPECT_NEAR(sumY[0], 0.8f, 1e-4f);
    EXPECT_NEAR(meanY[0], 0.4f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, sum_fp32_multi_row_tiles)
{
    std::vector<float> x1(5 * 3, 1.0f);
    std::vector<float> x2(5 * 3, 1.0f);
    std::vector<float> target = {1.0f, -1.0f, 1.0f, -1.0f, 0.0f};
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {5}, 3, {3}, {3}, {1}, 1, 1, 1, 0.2f, y, COSINE_EMBEDDING_LOSS_GENERIC_PATH, 0, 2);

    EXPECT_NEAR(y[0], 1.6f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, const_reduction_path_writes_scaled_sum)
{
    std::vector<float> x1(4 * 3, 1.0f);
    std::vector<float> x2(4 * 3, 1.0f);
    std::vector<float> target(4, -1.0f);
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {4}, 3, {3}, {3}, {1}, 1, 1, 1, 0.2f, y, COSINE_EMBEDDING_LOSS_CONST_REDUCTION_PATH,
                    0, 2);

    EXPECT_NEAR(y[0], 3.2f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, const_reduction_path_falls_back_for_nonconstant_input)
{
    std::vector<float> x1 = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> x2 = x1;
    std::vector<float> target = {1.0f, -1.0f};
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {2}, 3, {3}, {3}, {1}, 1, 1, 1, 0.2f, y,
                    COSINE_EMBEDDING_LOSS_CONST_REDUCTION_PATH);

    EXPECT_NEAR(y[0], 0.8f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, feature_broadcast_reduction_matches_generic_sum)
{
    // x1 shape [shared=3, u=4], x2 shape [v=2, d=2, shared=3].
    std::vector<float> x1 = {
        -0.5f, 0.2f, 0.8f, -0.1f, 0.4f, -0.7f, 0.05f, 0.9f, -0.3f, 0.6f, -0.8f, 0.15f,
    };
    std::vector<float> x2 = {
        -0.2f, 0.3f, -0.6f, -0.4f, -0.5f, 0.7f, 0.9f, -0.1f, 0.2f, 0.5f, -0.8f, -0.3f,
    };
    std::vector<float> target = {-1.0f};
    std::vector<float> genericY(1, 0.0f);
    std::vector<float> fastY(1, 0.0f);

    RunCosineKernel(x1, x2, target, {2, 3, 4}, 2, {0, 4, 1}, {6, 1, 0}, {0, 0, 0}, 0, 3, 1, -0.25f, genericY,
                    COSINE_EMBEDDING_LOSS_GENERIC_PATH);
    RunCosineKernel(x1, x2, target, {2, 3, 4}, 2, {0, 4, 1}, {6, 1, 0}, {0, 0, 0}, 0, 3, 1, -0.25f, fastY,
                    COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH);

    EXPECT_NEAR(fastY[0], genericY[0], 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, feature_broadcast_reduction_falls_back_for_non_negative_target)
{
    std::vector<float> x1 = {
        -0.5f, 0.2f, 0.8f, -0.1f, 0.4f, -0.7f, 0.05f, 0.9f, -0.3f, 0.6f, -0.8f, 0.15f,
    };
    std::vector<float> x2 = {
        -0.2f, 0.3f, -0.6f, -0.4f, -0.5f, 0.7f, 0.9f, -0.1f, 0.2f, 0.5f, -0.8f, -0.3f,
    };
    std::vector<float> target = {
        -1.0f, 1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f,  -1.0f,
    };
    std::vector<float> genericY(1, 0.0f);
    std::vector<float> fallbackY(1, 0.0f);

    RunCosineKernel(x1, x2, target, {2, 3, 4}, 2, {0, 4, 1}, {6, 1, 0}, {12, 4, 1}, 0, 3, 1, -0.25f, genericY,
                    COSINE_EMBEDDING_LOSS_GENERIC_PATH);
    RunCosineKernel(x1, x2, target, {2, 3, 4}, 2, {0, 4, 1}, {6, 1, 0}, {12, 4, 1}, 0, 3, 1, -0.25f, fallbackY,
                    COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH);

    EXPECT_NEAR(fallbackY[0], genericY[0], 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, feature_broadcast_reduction_samples_huge_x1_work)
{
    constexpr int64_t hugeX1Count = COSINE_EMBEDDING_LOSS_EXACT_FEATURE_REDUCTION_MAX_X1_VISITS + 1;
    std::vector<float> x1(hugeX1Count, 1.0f);
    std::vector<float> x2 = {1.0f, 1.0f, -1.0f, -1.0f};
    std::vector<float> target = {-1.0f};
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {2, hugeX1Count}, 2, {0, 1}, {2, 0}, {0, 0}, 0, 1, 1, 0.2f, y,
                    COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH);

    EXPECT_NEAR(y[0], 0.8f * static_cast<float>(hugeX1Count), 64.0f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, feature_broadcast_reduction_samples_huge_shared_work)
{
    constexpr int64_t sharedCount = COSINE_EMBEDDING_LOSS_EXACT_FEATURE_REDUCTION_MAX_X1_VISITS + 1;
    std::vector<float> x1(sharedCount * 2, 1.0f);
    std::vector<float> x2(sharedCount * 4, 1.0f);
    std::vector<float> target = {-1.0f};
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {sharedCount, 2, 2}, 2, {2, 1, 0}, {4, 0, 1}, {0, 0, 0}, 0, 2, 1, 0.2f, y,
                    COSINE_EMBEDDING_LOSS_FEATURE_BROADCAST_REDUCTION_PATH);

    EXPECT_NEAR(y[0], 3.2f * static_cast<float>(sharedCount), 16.0f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, fast_none_fp32_multi_row_tiles)
{
    std::vector<float> x1(5 * 70, 0.0f);
    std::vector<float> x2(5 * 70, 0.0f);
    x1[0] = 1.0f;
    x2[0] = 1.0f;
    x1[70] = 1.0f;
    x2[70] = 1.0f;
    x1[140] = 1.0f;
    x2[140] = 1.0f;
    x1[210] = 1.0f;
    x2[210] = 1.0f;
    x1[280] = 1.0f;
    x2[280] = 1.0f;
    std::vector<float> target = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f};
    std::vector<float> y(5, 0.0f);

    RunCosineKernel(x1, x2, target, {5}, 70, {70}, {70}, {1}, 1, 1, 0, 0.2f, y, COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH,
                    64, 2);

    EXPECT_NEAR(y[0], 0.0f, 1e-4f);
    EXPECT_NEAR(y[1], 0.8f, 1e-4f);
    EXPECT_NEAR(y[2], 0.0f, 1e-4f);
    EXPECT_NEAR(y[3], 0.8f, 1e-4f);
    EXPECT_NEAR(y[4], 0.0f, 1e-4f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, none_fp32_nan_inf_semantics)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> x1 = {nan, 0.0f, inf, 0.0f, nan, 0.0f};
    std::vector<float> x2 = {1.0f, 0.0f, inf, 0.0f, 1.0f, 0.0f};
    std::vector<float> target = {-1.0f, 1.0f, 0.0f};
    std::vector<float> y(3, 1.0f);

    RunCosineKernel(x1, x2, target, {3}, 2, {2}, {2}, {1}, 1, 1, 0, 0.2f, y);

    EXPECT_TRUE(std::isnan(y[0]));
    EXPECT_TRUE(std::isnan(y[1]));
    EXPECT_FLOAT_EQ(y[2], 0.0f);
}

TEST_F(CosineEmbeddingLossArch35Kernel, fast_sum_fp32_nan_propagates)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x1(64, 0.0f);
    std::vector<float> x2(64, 0.0f);
    x1[0] = nan;
    x2[0] = 1.0f;
    std::vector<float> target = {-1.0f};
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {1}, 64, {64}, {64}, {0}, 1, 1, 1, 0.2f, y, COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH,
                    64);

    EXPECT_TRUE(std::isnan(y[0]));
}

TEST_F(CosineEmbeddingLossArch35Kernel, fast_mean_fp32_inf_produces_nan)
{
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<float> x1(64, 0.0f);
    std::vector<float> x2(64, 0.0f);
    x1[0] = inf;
    x2[0] = inf;
    std::vector<float> target = {1.0f};
    std::vector<float> y(1, 0.0f);

    RunCosineKernel(x1, x2, target, {1}, 64, {64}, {64}, {0}, 1, 1, 2, 0.2f, y, COSINE_EMBEDDING_LOSS_CONTIG_2D_PATH,
                    64);

    EXPECT_TRUE(std::isnan(y[0]));
}
