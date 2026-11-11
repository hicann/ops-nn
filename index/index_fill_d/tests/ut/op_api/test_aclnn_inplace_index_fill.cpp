/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_index_fill.h"
#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using FillParam = std::tuple<op::SocVersion, aclDataType, aclDataType, int64_t>;
class InplaceIndexFillTypes : public testing::TestWithParam<FillParam> {};

TEST_P(InplaceIndexFillTypes, AcceptsTargetTypesOnEveryAxis)
{
    const auto [soc, dtype, indexType, dim] = GetParam();
    op::SocVersionManager platform(soc);
    auto self = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND).ValueRange(0, 1);
    auto index = TensorDesc({3}, indexType, ACL_FORMAT_ND).Value(std::vector<int64_t>{0, -1, 0});
    auto value = ScalarDesc(1.0000000000000002);
    auto ut = OP_API_UT(aclnnInplaceIndexFill, INPUT(self, dim, index, value), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_P(InplaceIndexFillTypes, HandlesEmptyIndex)
{
    const auto [soc, dtype, indexType, dim] = GetParam();
    op::SocVersionManager platform(soc);
    auto self = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND).ValueRange(0, 1);
    auto index = TensorDesc({0}, indexType, ACL_FORMAT_ND);
    auto value = ScalarDesc(1.0);
    auto ut = OP_API_UT(aclnnInplaceIndexFill, INPUT(self, dim, index, value), OUTPUT());
    uint64_t workspaceSize = UINT64_MAX;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_NE(workspaceSize, UINT64_MAX);
}

TEST_P(InplaceIndexFillTypes, HandlesNoncontiguousSelf)
{
    const auto [soc, dtype, indexType, dim] = GetParam();
    op::SocVersionManager platform(soc);
    auto self = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND, {1, 10, 2}).ValueRange(0, 1);
    auto index = TensorDesc({1}, indexType, ACL_FORMAT_ND).Value(std::vector<int64_t>{0});
    auto value = ScalarDesc(1.0);
    auto ut = OP_API_UT(aclnnInplaceIndexFill, INPUT(self, dim, index, value), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_P(InplaceIndexFillTypes, AcceptsOutOfPlaceTargetTypes)
{
    const auto [soc, dtype, indexType, dim] = GetParam();
    op::SocVersionManager platform(soc);
    auto self = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND).ValueRange(0, 1);
    auto index = TensorDesc({1}, indexType, ACL_FORMAT_ND).Value(std::vector<int64_t>{0});
    auto value = ScalarDesc(1.0);
    auto out = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, value), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_P(InplaceIndexFillTypes, HandlesOutOfPlaceEmptyIndex)
{
    const auto [soc, dtype, indexType, dim] = GetParam();
    op::SocVersionManager platform(soc);
    auto self = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND, {1, 10, 2}).ValueRange(0, 1);
    auto index = TensorDesc({0}, indexType, ACL_FORMAT_ND);
    auto value = ScalarDesc(1.0);
    auto out = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND, {30, 10, 2});
    auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, value), OUTPUT(out));
    uint64_t workspaceSize = UINT64_MAX;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_NE(workspaceSize, UINT64_MAX);
}

TEST_P(InplaceIndexFillTypes, HandlesOutOfPlaceNoncontiguousViews)
{
    const auto [soc, dtype, indexType, dim] = GetParam();
    op::SocVersionManager platform(soc);
    auto self = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND, {1, 10, 2}).ValueRange(0, 1);
    auto index = TensorDesc({1}, indexType, ACL_FORMAT_ND).Value(std::vector<int64_t>{0});
    auto value = ScalarDesc(1.0000000000000002);
    auto out = TensorDesc({2, 3, 5}, dtype, ACL_FORMAT_ND, {30, 10, 2});
    auto ut = OP_API_UT(aclnnIndexFill, INPUT(self, dim, index, value), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(A2AndA3, InplaceIndexFillTypes,
                         testing::Combine(testing::Values(op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93),
                                          testing::Values(ACL_INT8, ACL_UINT8, ACL_DOUBLE, ACL_FLOAT16),
                                          testing::Values(ACL_INT32, ACL_INT64), testing::Values(0L, 1L, 2L, -1L)));

TEST(InplaceIndexFillValidation, RejectsUnrequestedTypes)
{
    for (auto soc : {op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SocVersionManager platform(soc);
        for (auto dtype : {ACL_INT16, ACL_UINT64}) {
            auto self = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND).ValueRange(0, 1);
            auto index = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Value(std::vector<int64_t>{0});
            auto value = ScalarDesc(1.0);
            auto ut = OP_API_UT(aclnnInplaceIndexFill, INPUT(self, 0L, index, value), OUTPUT());
            uint64_t workspaceSize = 0;
            EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
            auto out = TensorDesc({2, 3}, dtype, ACL_FORMAT_ND);
            auto outUt = OP_API_UT(aclnnIndexFill, INPUT(self, 0L, index, value), OUTPUT(out));
            EXPECT_EQ(outUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
        }
    }
}
