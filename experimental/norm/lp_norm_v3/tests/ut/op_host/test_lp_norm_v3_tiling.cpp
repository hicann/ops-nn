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
 * \file test_lp_norm_v3_tiling.cpp
 * \brief LpNormV3 host tiling UT.
 */
#include <gtest/gtest.h>

#include "lp_norm_v3_tiling_data.h"
#include "tiling_case_executor.h"

namespace {

struct LpNormV3CompileInfo {};

gert::TilingContextPara MakeTilingPara(const std::vector<int64_t>& shape, int64_t axis,
                                       LpNormV3CompileInfo* compileInfo)
{
    gert::StorageShape inputShape(shape, shape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {inputShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {inputShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"p", Ops::NN::AnyValue::CreateFrom<float>(2.0f)},
        {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
    };
    return gert::TilingContextPara("LpNormV3", inputs, outputs, attrs, compileInfo);
}

const LpNormV3TilingData* GetTilingData(const TilingInfo& info)
{
    EXPECT_GE(info.tilingDataSize, sizeof(LpNormV3TilingData));
    return reinterpret_cast<const LpNormV3TilingData*>(info.tilingData.get());
}

TEST(LpNormV3Tiling, AxisZeroUsesRowsForTwoDimensionalInput)
{
    LpNormV3CompileInfo compileInfo;
    auto para = MakeTilingPara({2, 3}, 0, &compileInfo);
    TilingInfo info;

    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.tilingKey, 1);
    const auto* tilingData = GetTilingData(info);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->rows, 2U);
    EXPECT_EQ(tilingData->cols, 3U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_GE(info.workspaceSizes[0], 2U * 16U * sizeof(float));
}

TEST(LpNormV3Tiling, AxisOneUsesColumnsForTwoDimensionalInput)
{
    LpNormV3CompileInfo compileInfo;
    auto para = MakeTilingPara({2, 3}, 1, &compileInfo);
    TilingInfo info;

    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_EQ(info.tilingKey, 2);
    const auto* tilingData = GetTilingData(info);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->rows, 2U);
    EXPECT_EQ(tilingData->cols, 3U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_GE(info.workspaceSizes[0], 3U * 16U * sizeof(float));
}

TEST(LpNormV3Tiling, AxisZeroRejectsNonTwoDimensionalInput)
{
    LpNormV3CompileInfo compileInfo;
    auto para = MakeTilingPara({2, 3, 4}, 0, &compileInfo);
    TilingInfo info;

    EXPECT_FALSE(ExecuteTiling(para, info));
}

TEST(LpNormV3Tiling, AxisOneRejectsNonTwoDimensionalInput)
{
    LpNormV3CompileInfo compileInfo;
    auto para = MakeTilingPara({2, 3, 4}, 1, &compileInfo);
    TilingInfo info;

    EXPECT_FALSE(ExecuteTiling(para, info));
}

} // namespace
