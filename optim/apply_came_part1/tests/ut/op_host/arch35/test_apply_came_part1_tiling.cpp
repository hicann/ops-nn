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
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/apply_came_part1_tiling_arch35.h"
#include "../../../../op_host/arch35/apply_came_part1_ub_plan.h"

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

TilingResult RunTiling(const std::vector<int64_t>& gradShape, ge::DataType gradType, uint64_t ubSize = kUbSize,
                       uint64_t coreNum = kCoreNum)
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
    optiling::ApplyCamePart1CompileInfo compileInfo{static_cast<int32_t>(coreNum), ubSize};
    gert::TilingContextPara para("ApplyCamePart1", inputs, outputs, attrs, &compileInfo, coreNum, ubSize,
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

TEST(ApplyCamePart1Tiling, UbPlannerSelectsTileAndBoundsLiveBuffers)
{
    for (const uint64_t dtypeBytes : {2UL, 4UL}) {
        for (const auto& config : {std::pair<uint64_t, uint64_t>{kUbSize, 128}, {64UL * 1024UL, 64}}) {
            SCOPED_TRACE(::testing::Message() << "dtypeBytes=" << dtypeBytes << " ub=" << config.first);
            const auto plan = apply_came_ub::MakePlan(config.first, dtypeBytes, kCoreNum);
            ASSERT_EQ(plan.tile, config.second);
            EXPECT_GT(plan.directElements, 0U);
            EXPECT_EQ(plan.directSumGradRQueueBytes, 2U * 64U * sizeof(float));
            EXPECT_LE(apply_came_ub::kReserveBytes + plan.persistentBytes + plan.arenaBytes, config.first);
            EXPECT_EQ(plan.persistentBytes, plan.gradQueueBytes + plan.epsQueueBytes + plan.sumGradRQueueBytes +
                                                plan.sumGradCQueueBytes + plan.sumGradRCQueueBytes +
                                                plan.syncWorkQueueBytes);
            EXPECT_EQ(plan.gradQueueBytes, plan.tile * plan.tile * dtypeBytes);
            EXPECT_EQ(plan.epsQueueBytes, dtypeBytes == sizeof(float) ? 8 * sizeof(float) : 0);
            EXPECT_EQ(plan.sumGradRQueueBytes, 2 * plan.tile * sizeof(float));
            EXPECT_EQ(plan.sumGradCQueueBytes, 2 * plan.tile * sizeof(float));
            EXPECT_LE(plan.producerPeakBytes, plan.arenaBytes);
            if (dtypeBytes == sizeof(float)) {
                EXPECT_GE(plan.producerPeakBytes, apply_came_ub::Width2ProducerPeakBytes(plan.tile));
            }
            EXPECT_LE(plan.postElements * sizeof(float) + 64, plan.arenaBytes);
            EXPECT_EQ(plan.postElements % plan.tile, 0U);
            // R high/low pair reduction is the largest post phase: six full-width live nodes.
            EXPECT_GE(plan.postElements, 6 * plan.tile);
            EXPECT_LE((plan.postElements / (6 * plan.tile)) * 6 * plan.tile * sizeof(float) + 64, plan.arenaBytes);
        }
    }
}

TEST(ApplyCamePart1Tiling, Width2ArenaAccountsForAllConcurrentBuffers)
{
    constexpr uint64_t tile = 64;
    constexpr uint64_t requiredArena = apply_came_ub::Width2ProducerPeakBytes(tile);
    const auto reference = apply_came_ub::MakePlan(64UL * 1024UL, sizeof(float), kCoreNum);
    ASSERT_EQ(reference.tile, tile);
    ASSERT_EQ(reference.producerPeakBytes, requiredArena);

    const uint64_t exactUb = apply_came_ub::kReserveBytes + reference.persistentBytes + requiredArena;
    const auto exactPlan = apply_came_ub::MakePlan(exactUb, sizeof(float), kCoreNum);
    EXPECT_EQ(exactPlan.tile, tile);
    EXPECT_EQ(exactPlan.arenaBytes, requiredArena);

    const auto undersizedPlan = apply_came_ub::MakePlan(exactUb - 1, sizeof(float), kCoreNum);
    EXPECT_EQ(undersizedPlan.tile, 0U);
}

TEST(ApplyCamePart1Tiling, Width2DirectRUsesOneSharedBoundary)
{
    EXPECT_EQ(optiling::ApplyCamePart1Width2MaxChunkRows(64), 2048);
    EXPECT_EQ(optiling::ApplyCamePart1Width2MaxChunkRows(128), 8192);
    EXPECT_FALSE(optiling::ApplyCamePart1UseDirectWidth2R(2047, 2, 1, 64));
    EXPECT_TRUE(optiling::ApplyCamePart1UseDirectWidth2R(2048, 2, 1, 64));
    EXPECT_FALSE(optiling::ApplyCamePart1UseDirectWidth2R(8191, 2, 1, 128));
    EXPECT_TRUE(optiling::ApplyCamePart1UseDirectWidth2R(8192, 2, 1, 128));
    EXPECT_FALSE(optiling::ApplyCamePart1UseDirectWidth2R(8192, 3, 1, 128));
    EXPECT_FALSE(optiling::ApplyCamePart1UseDirectWidth2R(8192, 2, 2, 128));
}

TEST(ApplyCamePart1Tiling, LargeTilesUsePlannedArenaForEachDtype)
{
    for (const auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        const auto result = RunTiling({257, 259}, dtype);
        const uint64_t dtypeBytes = dtype == ge::DT_FLOAT ? 4 : 2;
        const auto plan = apply_came_ub::MakePlan(kUbSize, dtypeBytes, kCoreNum);
        ASSERT_EQ(plan.tile, 128U);
        EXPECT_EQ(result.data.tileSize, plan.tile);
        EXPECT_EQ(result.data.ubArenaBytes, plan.arenaBytes);
        EXPECT_EQ(result.data.postUbElements, plan.postElements);
        EXPECT_EQ(result.data.gradQueueBytes, plan.gradQueueBytes);
        EXPECT_EQ(result.data.epsQueueBytes, plan.epsQueueBytes);
        EXPECT_EQ(result.data.sumGradRQueueBytes, plan.sumGradRQueueBytes);
        EXPECT_EQ(result.data.sumGradCQueueBytes, plan.sumGradCQueueBytes);
        EXPECT_EQ(result.data.sumGradRCQueueBytes, plan.sumGradRCQueueBytes);
        EXPECT_EQ(result.data.syncWorkQueueBytes, plan.syncWorkQueueBytes);
        EXPECT_EQ(result.data.directBatchMode, 0);
    }
}

TEST(ApplyCamePart1Tiling, Fp16UsesGeneralBatchDirectPath)
{
    const auto result = RunTiling({12562512, 2, 2}, ge::DT_FLOAT16);
    EXPECT_EQ(result.data.N, 2);
    EXPECT_EQ(result.data.M, 2);
    EXPECT_EQ(result.data.batchCount, 12562512);
    EXPECT_EQ(result.data.directBatchMode, 1);
    EXPECT_EQ(result.data.directGradQueueBytes, result.data.directUbElements * sizeof(uint16_t));
    EXPECT_EQ(result.data.directValueBytes, result.data.directUbElements * sizeof(float));
    EXPECT_EQ(result.data.directCastOutputBytes, (result.data.directUbElements + 2 * 32 + 3 * 8) * sizeof(float));
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
    EXPECT_EQ(insufficientBatch.data.directBatchMode, 1);
    EXPECT_GT(insufficientBatch.data.usedCoreNum, 1);
    EXPECT_LE(insufficientBatch.data.usedCoreNum, static_cast<int64_t>(kCoreNum));
    EXPECT_LE(insufficientBatch.data.usedCoreNum, insufficientBatch.data.batchCount);
    ASSERT_EQ(insufficientBatch.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(insufficientBatch.info.workspaceSizes[0] % kWorkspaceAlignment, 0U);

    const auto overCapacity = RunTiling({48, 65, 127}, ge::DT_FLOAT);
    EXPECT_EQ(overCapacity.data.directBatchMode, 0);
    ASSERT_EQ(overCapacity.info.workspaceSizes.size(), 1U);
    EXPECT_EQ(overCapacity.info.workspaceSizes[0] % kWorkspaceAlignment, 0U);
}

TEST(ApplyCamePart1Tiling, DirectBlocksCoverBatchesWithoutEmptyBlocks)
{
    const auto result = RunTiling({1000, 2, 2}, ge::DT_FLOAT16);
    ASSERT_GT(result.data.usedCoreNum, 0);
    const int64_t batchesPerBlock = (result.data.batchCount + result.data.usedCoreNum - 1) / result.data.usedCoreNum;
    EXPECT_GE(batchesPerBlock * result.data.usedCoreNum, result.data.batchCount);
    EXPECT_LT(batchesPerBlock * (result.data.usedCoreNum - 1), result.data.batchCount);
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

TEST(ApplyCamePart1Tiling, AlignsEachRCompensationPlaneBeforePairing)
{
    constexpr uint64_t kSmallUbSize = 64UL * 1024UL;
    const auto result = RunTiling({5, 575, 15}, ge::DT_FLOAT, kSmallUbSize, kCoreNum);
    EXPECT_EQ(result.data.tileSize, 64);
    EXPECT_EQ(result.data.nCoreNum, 9);
    EXPECT_EQ(result.data.mCoreNum, 1);
    EXPECT_EQ(result.data.batchCoreNum, 5);
    ASSERT_EQ(result.info.workspaceSizes.size(), 1U);
    const size_t nLoopCount = static_cast<size_t>((result.data.nCoreNum - 1) * result.data.nLoopNormCore +
                                                  result.data.nLoopTailCore);
    const size_t rcPartialCount = nLoopCount * static_cast<size_t>((result.data.M + result.data.tileSize - 1) /
                                                                   result.data.tileSize);
    const size_t rPartialCount = nLoopCount * static_cast<size_t>(result.data.mCoreNum);
    const size_t rcAlignedCount = (rcPartialCount * 8U + 127U) / 128U * 128U;
    const size_t rAlignedCount = (rPartialCount * static_cast<size_t>(result.data.tileSize) + 127U) / 128U * 128U;
    const size_t cSlotSize = result.data.M == 2 ? 8U : static_cast<size_t>(result.data.tileSize);
    const size_t cAlignedCount = (rcPartialCount * cSlotSize + 127U) / 128U * 128U;
    const size_t regionBytes = 2U * (rcAlignedCount + rAlignedCount + cAlignedCount) * sizeof(float);
    const size_t syncBytes = static_cast<size_t>(result.data.totalCoreNum) * 32U + 32U;
    const size_t userBytes = regionBytes * static_cast<size_t>(result.data.batchCoreNum) + syncBytes;
    const size_t expectedWorkspace = (kSystemWorkspaceSize + userBytes + kWorkspaceAlignment - 1U) /
                                     kWorkspaceAlignment * kWorkspaceAlignment;
    EXPECT_EQ(result.info.workspaceSizes[0], expectedWorkspace);
}

} // namespace ApplyCamePart1TilingUT
