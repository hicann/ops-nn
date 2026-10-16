/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>

#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/unique_with_counts_and_sorting_tiling_data.h"

namespace {
constexpr uint64_t CORE_COUNT = 64;
constexpr uint64_t UB_BYTES = 262144;
constexpr int64_t SIMT_RESERVE_BYTES = 32768;
constexpr int64_t SINGLE_CORE_LIMIT = 8192;
constexpr int64_t COMPACT_MIN_ELEMENTS = 65536;
constexpr int64_t LARGE_COUNT = 1000000;
constexpr int64_t VALUE_BUFFER_COUNT = 2;
constexpr int64_t COUNT_BUFFER_COUNT = 2;
constexpr int64_t RADIX_BINS = 256;
constexpr int64_t SHAPE_ELEMENTS = 27;
constexpr size_t VALUES_OUTPUT = 0;
struct CompileInfo {};
CompileInfo g_compileInfo;

class UniqueWithCountsAndSortingTilingTest : public testing::Test {
protected:
    void SetUp() override
    {
        previousPlatform_ = platform_ascendc::PlatformAscendCManager::platformInfo;
        ASSERT_EQ(fe::PlatformInfoManager::GeInstance().InitializePlatformInfo(), ge::GRAPH_SUCCESS);
        platformInfo_.Init();
        fe::OptionalInfos optionalInfo;
        optionalInfo.Init();
        ASSERT_EQ(
            fe::PlatformInfoManager::GeInstance().GetPlatformInfos("Ascend950DT_9574", platformInfo_, optionalInfo),
            ge::GRAPH_SUCCESS);
        ASSERT_EQ(platform_.GetCurNpuArch(), NpuArch::DAV_3510);
        // Use the repository UT's -fno-access-control to scope the external API platform to this test.
        // GetInstance initialization may already have cached failure in a preceding suite.
        platform_ascendc::PlatformAscendCManager::platformInfo = &platform_;
        ASSERT_EQ(platform_ascendc::PlatformAscendCManager::GetInstance(), &platform_);
    }

    void TearDown() override { platform_ascendc::PlatformAscendCManager::platformInfo = previousPlatform_; }

private:
    fe::PlatFormInfos platformInfo_;
    platform_ascendc::PlatformAscendC platform_{&platformInfo_};
    platform_ascendc::PlatformAscendC* previousPlatform_ = nullptr;
};

gert::TilingContextPara MakeContext(int64_t count, ge::DataType dtype, uint64_t ub = UB_BYTES)
{
    const gert::StorageShape input{{count}, {count}};
    const gert::StorageShape placeholder{{0}, {0}};
    return gert::TilingContextPara("UniqueWithCountsAndSorting", {{input, dtype, ge::FORMAT_ND}},
                                   {{input, dtype, ge::FORMAT_ND},
                                    {placeholder, ge::DT_INT64, ge::FORMAT_ND},
                                    {placeholder, ge::DT_INT64, ge::FORMAT_ND}},
                                   {{"return_inverse", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                    {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                    {"sorted", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                    {"out_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT64)}},
                                   &g_compileInfo, CORE_COUNT, ub);
}

TEST_F(UniqueWithCountsAndSortingTilingTest, KeepBuffersWithinBudgetAcrossPlanBoundaries)
{
    for (int64_t count :
         {SINGLE_CORE_LIMIT, SINGLE_CORE_LIMIT + 1, COMPACT_MIN_ELEMENTS, COMPACT_MIN_ELEMENTS + 1, LARGE_COUNT}) {
        for (auto dtype : {ge::DT_UINT8, ge::DT_FLOAT, ge::DT_INT64}) {
            SCOPED_TRACE(count);
            SCOPED_TRACE(dtype);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(MakeContext(count, dtype), info));
            ASSERT_GE(info.tilingDataSize, sizeof(UniqueWithCountsAndSortingTilingData));
            const auto& data = *reinterpret_cast<const UniqueWithCountsAndSortingTilingData*>(info.tilingData.get());
            const auto& unique = data.unique;
            EXPECT_EQ(unique.totalSize, count);
            EXPECT_GT(unique.tileLengthTailCore, 0);
            EXPECT_LE(unique.useCoreNums, static_cast<int64_t>(CORE_COUNT));
            EXPECT_GE(unique.shapeBufSize, SHAPE_ELEMENTS * static_cast<int64_t>(sizeof(uint64_t)));
            EXPECT_GT(unique.adjUbTileLength, 1);
            const int64_t bytes = unique.collectingCntBufSize + unique.offsetCntBufSize + unique.prevIdxBufSize +
                                  unique.shapeBufSize + unique.valueQueueSize * VALUE_BUFFER_COUNT +
                                  unique.countQueueSize * COUNT_BUFFER_COUNT;
            EXPECT_LE(bytes, static_cast<int64_t>(UB_BYTES));
            EXPECT_LE(unique.idxQueueSize, unique.countQueueSize);
            if (count >= COMPACT_MIN_ELEMENTS) {
                EXPECT_LE(bytes, static_cast<int64_t>(UB_BYTES) - SIMT_RESERVE_BYTES);
                EXPECT_GT(data.sort.numTileDataSize, 0);
                EXPECT_EQ(data.sort.numTileDataSize % RADIX_BINS, 0);
            }
            ASSERT_EQ(info.workspaceSizes.size(), 1);
            EXPECT_GT(info.workspaceSizes[VALUES_OUTPUT], 0);
        }
    }
}

TEST_F(UniqueWithCountsAndSortingTilingTest, RejectTileCountBeyondKernelRange)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext(std::numeric_limits<int64_t>::max(), ge::DT_INT64), info));
}

TEST_F(UniqueWithCountsAndSortingTilingTest, RejectInsufficientUbBeforeBufferSubtraction)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(MakeContext(COMPACT_MIN_ELEMENTS, ge::DT_FLOAT, SIMT_RESERVE_BYTES), info));
    EXPECT_FALSE(ExecuteTiling(MakeContext(COMPACT_MIN_ELEMENTS, ge::DT_FLOAT, SIMT_RESERVE_BYTES + 1), info));
}
} // namespace
