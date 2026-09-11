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
#include "index/kth_value/op_host/arch35/kth_value_tiling_arch35.h"
#include "index/kth_value/op_kernel/arch35/kth_value_tiling_data.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
optiling::KthValueCompileInfo g_compileInfo = {64};

gert::TilingContextPara MakeTilingContext(const gert::StorageShape& xShape, const gert::StorageShape& outputShape,
                                          ge::DataType dtype, int64_t dim)
{
    return gert::TilingContextPara(
        "NanMedian",
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

TEST(NanMedianTilingTest, UsesIgnoreNanMode)
{
    auto context = MakeTilingContext({{8, 7}, {8, 7}}, {{8, 1}, {8, 1}}, ge::DT_BF16, -1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 264);
    const auto* tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData->medianMode, 2U);
    EXPECT_EQ(tilingData->kthIndex, 3);
}

TEST(NanMedianTilingTest, IntegerInputUsesUnchangedStaticK)
{
    auto context = MakeTilingContext({{2, 8}, {2, 8}}, {{2, 1}, {2, 1}}, ge::DT_INT64, -1);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(context, tilingInfo));
    const auto* tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData->medianMode, 0U);
    EXPECT_EQ(tilingData->kthIndex, 3);
}

TEST(NanMedianTilingTest, RejectsOutOfRangeDimension)
{
    auto context = MakeTilingContext({{2, 8}, {2, 8}}, {{2, 1}, {2, 1}}, ge::DT_FLOAT, 2);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
