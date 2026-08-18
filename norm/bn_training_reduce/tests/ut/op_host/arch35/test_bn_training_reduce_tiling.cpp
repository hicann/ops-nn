/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../../op_host/arch35/bn_training_reduce_tiling_public.h"

namespace optiling {
namespace {

constexpr int64_t kTestUbSizeBytes = 256 * 1024;

BNTrainingReducePublicInputs MakeInputs(const std::array<int64_t, 4>& shape, BNTrainingReducePublicFormat format)
{
    BNTrainingReducePublicInputs inputs;
    inputs.shape = shape;
    inputs.format = format;
    inputs.ubSize = kTestUbSizeBytes;
    const size_t channelIndex = format == BNTrainingReducePublicFormat::NCHW ? 1 : 3;
    inputs.sumDim0 = shape[channelIndex];
    inputs.squareSumDim0 = shape[channelIndex];
    return inputs;
}

TEST(BNTrainingReduceTilingTest, DispatchesGroupPathByDeterministicMode)
{
    auto inputs = MakeInputs({32, 8, 64, 64}, BNTrainingReducePublicFormat::NCHW);
    inputs.systemWorkspaceSize = 4096;

    const auto nonDeterministicResult = ComputeBNTrainingReducePublicTiling(inputs);
    ASSERT_EQ(nonDeterministicResult.status, BNTrainingReducePublicStatus::SUCCESS);
    EXPECT_EQ(nonDeterministicResult.tilingKey, static_cast<int64_t>(BNTrainingReduceTilingKey::GROUP_TAIL_R));
    EXPECT_EQ(nonDeterministicResult.workspaceSize, inputs.systemWorkspaceSize);

    inputs.deterministic = true;
    const auto deterministicResult = ComputeBNTrainingReducePublicTiling(inputs);
    ASSERT_EQ(deterministicResult.status, BNTrainingReducePublicStatus::SUCCESS);
    EXPECT_EQ(deterministicResult.tilingKey,
              static_cast<int64_t>(BNTrainingReduceTilingKey::DETERMINISTIC_GROUP_TAIL_R));
    EXPECT_EQ(deterministicResult.blockDim, nonDeterministicResult.blockDim);
    EXPECT_EQ(deterministicResult.tilingData.rGroupCnt, nonDeterministicResult.tilingData.rGroupCnt);
    EXPECT_GT(deterministicResult.workspaceSize, nonDeterministicResult.workspaceSize);
}

TEST(BNTrainingReduceTilingTest, SupportsNhwc)
{
    const auto inputs = MakeInputs({2, 4, 5, 3}, BNTrainingReducePublicFormat::NHWC);
    const auto result = ComputeBNTrainingReducePublicTiling(inputs);

    EXPECT_EQ(result.status, BNTrainingReducePublicStatus::SUCCESS);
    EXPECT_GT(result.blockDim, 0U);
}

TEST(BNTrainingReduceTilingTest, SupportsNhwcEmptyChannel)
{
    const auto inputs = MakeInputs({2, 4, 5, 0}, BNTrainingReducePublicFormat::NHWC);
    const auto result = ComputeBNTrainingReducePublicTiling(inputs);

    EXPECT_EQ(result.status, BNTrainingReducePublicStatus::SUCCESS);
    EXPECT_EQ(result.tilingKey, static_cast<int64_t>(BNTrainingReduceTilingKey::EMPTY));
    EXPECT_EQ(result.blockDim, 1U);
}

TEST(BNTrainingReduceTilingTest, SupportsNhwcEmptyReduceAxis)
{
    const auto inputs = MakeInputs({0, 4, 5, 3}, BNTrainingReducePublicFormat::NHWC);
    const auto result = ComputeBNTrainingReducePublicTiling(inputs);

    EXPECT_EQ(result.status, BNTrainingReducePublicStatus::SUCCESS);
    EXPECT_EQ(result.tilingKey, static_cast<int64_t>(BNTrainingReduceTilingKey::EMPTY));
    EXPECT_EQ(result.tilingData.axisShape[0], 3);
    EXPECT_GT(result.tilingData.usedCoreNum, 0);
}

} // namespace
} // namespace optiling
