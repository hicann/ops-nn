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
#include <string>
#include <vector>
#include "index/kth_value/op_host/arch35/kth_value_tiling_arch35.h"
#include "index/kth_value/op_kernel/arch35/kth_value_tiling_data.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
constexpr size_t WORK_SPACE_SIZE = 16777216;
optiling::KthValueCompileInfo g_compileInfo = {64};

gert::TilingContextPara MakeTilingContext(const gert::StorageShape& xShape, const gert::StorageShape& outputShape,
                                          ge::DataType dtype, int64_t dim)
{
    return gert::TilingContextPara(
        "Median",
        {
            {xShape, dtype, ge::FORMAT_ND},
        },
        {
            {outputShape, dtype, ge::FORMAT_ND},
            {outputShape, ge::DT_INT64, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)),
        },
        &g_compileInfo);
}

const KthValueTilingData* GetTilingData(const TilingInfo& tilingInfo)
{
    EXPECT_GE(tilingInfo.tilingDataSize, sizeof(KthValueTilingData));
    return reinterpret_cast<const KthValueTilingData*>(tilingInfo.tilingData.get());
}
} // namespace

TEST(MedianTilingTest, FloatMedianUsesNanPropagatingMode)
{
    auto context = MakeTilingContext({{4, 6}, {4, 6}}, {{4, 1}, {4, 1}}, ge::DT_FLOAT, -1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 264);
    const auto* tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData->medianMode, 1U);
    EXPECT_EQ(tilingData->kthIndex, 2);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST(MedianTilingTest, IntegerMedianKeepsStaticRank)
{
    auto context = MakeTilingContext({{3, 8}, {3, 8}}, {{3, 1}, {3, 1}}, ge::DT_INT64, -1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    const auto* tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData->medianMode, 0U);
    EXPECT_EQ(tilingData->kthIndex, 3);
}

TEST(MedianTilingTest, AxisLengthOneUsesCopyRoute)
{
    auto context = MakeTilingContext({{100, 1}, {100, 1}}, {{100, 1}, {100, 1}}, ge::DT_FLOAT16, -1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 263);
    const auto* tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData->medianMode, 1U);
    EXPECT_EQ(tilingData->kthIndex, 0);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], WORK_SPACE_SIZE);
}

TEST(MedianTilingTest, RejectsOutOfRangeDimension)
{
    auto context = MakeTilingContext({{2, 8}, {2, 8}}, {{2, 1}, {2, 1}}, ge::DT_FLOAT, 2);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

struct MedianRouteCase {
    const char* name;
    int64_t rows;
    int64_t axis;
    ge::DataType dtype;
    uint64_t expectedTilingKey;
};

class MedianRouteTilingTest : public testing::TestWithParam<MedianRouteCase> {};

TEST_P(MedianRouteTilingTest, CoversKthValueLastAxisRoute)
{
    const auto& testCase = GetParam();
    auto context = MakeTilingContext({{testCase.rows, testCase.axis}, {testCase.rows, testCase.axis}},
                                     {{testCase.rows, 1}, {testCase.rows, 1}}, testCase.dtype, -1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, testCase.expectedTilingKey);
}

INSTANTIATE_TEST_SUITE_P(LastAxisRoutes, MedianRouteTilingTest,
                         testing::Values(MedianRouteCase{"merge_sort", 2, 1024, ge::DT_FLOAT, 256},
                                         MedianRouteCase{"merge_more_core", 2, 8192, ge::DT_FLOAT, 259},
                                         MedianRouteCase{"merge_intra_core", 64, 16384, ge::DT_FLOAT, 260},
                                         MedianRouteCase{"small_axis_insertion", 1024, 4, ge::DT_FLOAT, 261},
                                         MedianRouteCase{"small_axis_two_stage", 4096, 17, ge::DT_FLOAT, 262},
                                         MedianRouteCase{"axis_one_copy", 100, 1, ge::DT_FLOAT16, 263},
                                         MedianRouteCase{"sort32_small_axis", 2, 24, ge::DT_FLOAT, 264},
                                         MedianRouteCase{"radix_select", 1, 1000000, ge::DT_FLOAT, 267},
                                         MedianRouteCase{"small_axis_short_rank_select", 16936, 15, ge::DT_INT64, 268}),
                         [](const testing::TestParamInfo<MedianRouteCase>& info) {
                             return std::string(info.param.name);
                         });
