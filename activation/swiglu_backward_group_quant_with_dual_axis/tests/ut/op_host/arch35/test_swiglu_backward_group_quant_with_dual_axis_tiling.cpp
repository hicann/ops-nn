/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swiglu_backward_group_quant_with_dual_axis_ut_common.h"

namespace {
SwigluTilingCase MakeNoWeightCase(gert::StorageShape& gradY, gert::StorageShape& x, gert::StorageShape& y,
                                  gert::StorageShape& scale1, gert::StorageShape& scale2,
                                  gert::StorageShape& gradWeight)
{
    return {{&gradY, &x},
            {&y, &scale1, &y, &scale2, &gradWeight},
            {1, 1, 0, 0, 0},
            {ge::DT_BF16, ge::DT_BF16},
            {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT}};
}

TEST_F(SwigluBackwardGroupQuantWithDualAxisTilingTest, BasicBf16WithoutWeight)
{
    gert::StorageShape gradY = {{128, 256}, {128, 256}};
    gert::StorageShape x = {{128, 512}, {128, 512}};
    gert::StorageShape y = {{128, 512}, {128, 512}};
    gert::StorageShape scale1 = {{128, 8, 2}, {128, 8, 2}};
    gert::StorageShape scale2 = {{2, 512, 2}, {2, 512, 2}};
    gert::StorageShape gradWeight = {{0}, {0}};
    auto testCase = MakeNoWeightCase(gradY, x, y, scale1, scale2, gradWeight);
    auto result = Run(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.totalRows, 128U);
    EXPECT_EQ(result.data.dimN, 256U);
    EXPECT_EQ(result.data.nTiles, 2U);
    EXPECT_EQ(result.data.numGroups, 1U);
}

TEST_F(SwigluBackwardGroupQuantWithDualAxisTilingTest, GroupedBf16WithFp32Weight)
{
    gert::StorageShape gradY = {{128, 256}, {128, 256}};
    gert::StorageShape x = {{128, 512}, {128, 512}};
    gert::StorageShape weight = {{128, 1}, {128, 1}};
    gert::StorageShape yOrigin = {{128, 256}, {128, 256}};
    gert::StorageShape groupIndex = {{3}, {3}};
    gert::StorageShape y = {{128, 512}, {128, 512}};
    gert::StorageShape scale1 = {{128, 8, 2}, {128, 8, 2}};
    gert::StorageShape scale2 = {{5, 512, 2}, {5, 512, 2}};
    gert::StorageShape gradWeight = {{128, 1}, {128, 1}};
    SwigluTilingCase testCase{
        {&gradY, &x, &weight, &yOrigin, &groupIndex},
        {&y, &scale1, &y, &scale2, &gradWeight},
        {1, 1, 1, 1, 1},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT64},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT}};
    testCase.clampLimit = 7.0f;
    testCase.bias = 1.0f;
    auto result = Run(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.numGroups, 3U);
    EXPECT_EQ(result.data.gradWeightTileH, 256U);
    EXPECT_EQ(result.data.gradWeightTileTokens, 32U);
    EXPECT_FLOAT_EQ(result.data.clampLimit, 7.0f);
}

TEST_F(SwigluBackwardGroupQuantWithDualAxisTilingTest, Fp16E5M2WithClamp)
{
    gert::StorageShape gradY = {{65, 128}, {65, 128}};
    gert::StorageShape x = {{65, 256}, {65, 256}};
    gert::StorageShape y = {{65, 256}, {65, 256}};
    gert::StorageShape scale1 = {{65, 4, 2}, {65, 4, 2}};
    gert::StorageShape scale2 = {{2, 256, 2}, {2, 256, 2}};
    gert::StorageShape gradWeight = {{0}, {0}};
    auto testCase = MakeNoWeightCase(gradY, x, y, scale1, scale2, gradWeight);
    testCase.inputTypes = {ge::DT_FLOAT16, ge::DT_FLOAT16};
    testCase.outputTypes[0] = ge::DT_FLOAT8_E5M2;
    testCase.outputTypes[2] = ge::DT_FLOAT8_E5M2;
    testCase.clampLimit = 10.0f;
    testCase.dstType = 35;
    auto result = Run(testCase);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.data.dimM, 65U);
    EXPECT_EQ(result.data.dimN, 128U);
    EXPECT_FLOAT_EQ(result.data.clampLimit, 10.0f);
}

TEST_F(SwigluBackwardGroupQuantWithDualAxisTilingTest, RejectRankThreeInput)
{
    gert::StorageShape gradY = {{2, 64, 256}, {2, 64, 256}};
    gert::StorageShape x = {{2, 64, 512}, {2, 64, 512}};
    gert::StorageShape y = {{2, 64, 512}, {2, 64, 512}};
    gert::StorageShape scale1 = {{2, 64, 8, 2}, {2, 64, 8, 2}};
    gert::StorageShape scale2 = {{2, 1, 512, 2}, {2, 1, 512, 2}};
    gert::StorageShape gradWeight = {{0}, {0}};
    auto testCase = MakeNoWeightCase(gradY, x, y, scale1, scale2, gradWeight);
    EXPECT_EQ(Run(testCase).status, ge::GRAPH_FAILED);
}

TEST_F(SwigluBackwardGroupQuantWithDualAxisTilingTest, RejectWeightWithoutGroupIndex)
{
    gert::StorageShape gradY = {{64, 128}, {64, 128}};
    gert::StorageShape x = {{64, 256}, {64, 256}};
    gert::StorageShape weight = {{64, 1}, {64, 1}};
    gert::StorageShape yOrigin = {{64, 128}, {64, 128}};
    gert::StorageShape y = {{64, 256}, {64, 256}};
    gert::StorageShape scale1 = {{64, 4, 2}, {64, 4, 2}};
    gert::StorageShape scale2 = {{1, 256, 2}, {1, 256, 2}};
    gert::StorageShape gradWeight = {{64, 1}, {64, 1}};
    SwigluTilingCase testCase{
        {&gradY, &x, &weight, &yOrigin},
        {&y, &scale1, &y, &scale2, &gradWeight},
        {1, 1, 1, 1, 0},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT}};
    EXPECT_EQ(Run(testCase).status, ge::GRAPH_FAILED);
}
} // namespace
