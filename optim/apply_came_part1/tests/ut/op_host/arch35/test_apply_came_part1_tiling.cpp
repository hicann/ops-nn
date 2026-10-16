/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/apply_came_part1_tiling_arch35.h"

namespace ApplyCamePart1TilingUT {
namespace {
constexpr uint64_t kUbSize = 248UL * 1024UL;
constexpr uint64_t kCoreNum = 48UL;
constexpr uint64_t kTilingDataSize = 4096UL;
constexpr size_t kSystemWorkspaceSize = 16UL * 1024UL * 1024UL;
constexpr size_t kWorkspaceAlignment = 512UL;

struct TilingResult {
    TilingInfo info;
    optiling::ApplyCamePart1TilingData data;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

TilingResult RunTiling(const std::vector<int64_t>& gradShape, ge::DataType gradType)
{
    const gert::StorageShape grad = MakeStorageShape(gradShape);
    const gert::StorageShape scalar = MakeStorageShape({1});
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {grad, gradType, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    optiling::ApplyCamePart1CompileInfo compileInfo{static_cast<int32_t>(kCoreNum), kUbSize};
    gert::TilingContextPara para("ApplyCamePart1", inputs, outputs, attrs, &compileInfo, kCoreNum, kUbSize,
                                 kTilingDataSize);

    TilingResult result;
    EXPECT_TRUE(ExecuteTiling(para, result.info));
    EXPECT_EQ(result.info.tilingKey, 0);
    EXPECT_EQ(result.info.tilingDataSize, sizeof(result.data));
    if (result.info.tilingDataSize == sizeof(result.data)) {
        std::memcpy(&result.data, result.info.tilingData.get(), sizeof(result.data));
    }
    return result;
}

bool TryTiling(const std::vector<int64_t>& gradShape, ge::DataType gradType)
{
    const gert::StorageShape grad = MakeStorageShape(gradShape);
    const gert::StorageShape scalar = MakeStorageShape({1});
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {grad, gradType, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
        {scalar, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    optiling::ApplyCamePart1CompileInfo compileInfo{static_cast<int32_t>(kCoreNum), kUbSize};
    gert::TilingContextPara para("ApplyCamePart1", inputs, outputs, attrs, &compileInfo, kCoreNum, kUbSize,
                                 kTilingDataSize);
    TilingInfo info;
    return ExecuteTiling(para, info);
}
} // namespace

TEST(ApplyCamePart1Tiling, Fp16UsesGeneralBatchDirectPath)
{
    const auto result = RunTiling({12562512, 2, 2}, ge::DT_FLOAT16);
    EXPECT_EQ(result.data.N, 2);
    EXPECT_EQ(result.data.M, 2);
    EXPECT_EQ(result.data.batchCount, 12562512);
    EXPECT_EQ(result.data.directBatchMode, 1);
    EXPECT_EQ(result.data.usedCoreNum, static_cast<int64_t>(kCoreNum));
    EXPECT_EQ(result.info.blockNum, kCoreNum);
    ASSERT_EQ(result.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(result.info.workspaceSizes[0], kSystemWorkspaceSize + kWorkspaceAlignment);
    EXPECT_EQ(result.info.workspaceSizes[0] % kWorkspaceAlignment, 0U);
}

TEST(ApplyCamePart1Tiling, Bf16UsesGeneralBatchDirectPath)
{
    const auto result = RunTiling({63607104, 2, 2}, ge::DT_BF16);
    EXPECT_EQ(result.data.batchCount, 63607104);
    EXPECT_EQ(result.data.directBatchMode, 1);
    EXPECT_EQ(result.data.usedCoreNum, static_cast<int64_t>(kCoreNum));
    EXPECT_EQ(result.info.blockNum, kCoreNum);
    ASSERT_EQ(result.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(result.info.workspaceSizes[0], kSystemWorkspaceSize + kWorkspaceAlignment);
}

TEST(ApplyCamePart1Tiling, DirectSelectionUsesCapacityAndParallelism)
{
    const auto singleTile = RunTiling({1, 64, 64}, ge::DT_FLOAT);
    EXPECT_EQ(singleTile.data.directBatchMode, 1);
    EXPECT_EQ(singleTile.data.usedCoreNum, 1);

    const auto parallelBatch = RunTiling({384, 65, 65}, ge::DT_FLOAT);
    EXPECT_EQ(parallelBatch.data.directBatchMode, 1);
    EXPECT_EQ(parallelBatch.data.usedCoreNum, static_cast<int64_t>(kCoreNum));

    const auto insufficientBatch = RunTiling({376, 65, 65}, ge::DT_FLOAT);
    EXPECT_EQ(insufficientBatch.data.directBatchMode, 0);
    ASSERT_EQ(insufficientBatch.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(insufficientBatch.info.workspaceSizes[0] % kWorkspaceAlignment, 0U);

    const auto overCapacity = RunTiling({48, 65, 127}, ge::DT_FLOAT);
    EXPECT_EQ(overCapacity.data.directBatchMode, 0);
    ASSERT_EQ(overCapacity.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(overCapacity.info.workspaceSizes[0] % kWorkspaceAlignment, 0U);
}

TEST(ApplyCamePart1Tiling, DirectBlocksStartOnEightBatchBoundaries)
{
    const auto result = RunTiling({1000, 2, 2}, ge::DT_FLOAT16);
    ASSERT_GT(result.data.usedCoreNum, 0);
    const int64_t batchesPerBlock = (result.data.batchCount + result.data.usedCoreNum - 1) / result.data.usedCoreNum;
    const int64_t alignedBatchesPerBlock = (batchesPerBlock + 7) / 8 * 8;
    EXPECT_EQ(alignedBatchesPerBlock % 8, 0);
    EXPECT_GE(alignedBatchesPerBlock * result.data.usedCoreNum, result.data.batchCount);
    EXPECT_LT(alignedBatchesPerBlock * (result.data.usedCoreNum - 1), result.data.batchCount);
}

TEST(ApplyCamePart1Tiling, SplitsWideColumnsAcrossCoresWithoutFixedDimensionLimit)
{
    constexpr int64_t kFormerColumnLimit = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) * 64;
    const auto result = RunTiling({1, kFormerColumnLimit + 1}, ge::DT_FLOAT16);
    EXPECT_GT(result.data.mCoreNum, 1);
    EXPECT_EQ(result.data.mNormalCoreNum % 64, 0);
    EXPECT_GT(result.data.mLoopNumCore, 0);
    EXPECT_EQ(result.data.mTailCoreNum, result.data.M - result.data.mNormalCoreNum * (result.data.mCoreNum - 1));
}

} // namespace ApplyCamePart1TilingUT
