/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "level2/aclnn_hardswish.h"
#include "level2/aclnn_fill_scalar.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class HardswishContiguousTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "HardswishContiguousTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "HardswishContiguousTest TearDown" << std::endl;
    }
};

TEST_F(HardswishContiguousTest, Contiguous)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2});
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2});

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(HardswishContiguousTest, Transpose)
{
    auto selfDesc = TensorDesc({3, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND, {210, 42, 1, 7}, 0, {3, 5, 6, 7});
    auto outDesc = TensorDesc({3, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND, {210, 42, 1, 7}, 0, {3, 5, 6, 7});

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(HardswishContiguousTest, BroadcastTo)
{
    // TODO
    //  auto selfDesc = TensorDesc({4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND, {6, 0, 1, 0}, 0, {4, 1, 6, 1});
    //  auto outDesc = TensorDesc({4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND, {6, 0, 1, 0}, 0, {4, 1, 6, 1});
    //
    //  auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));
    //
    //  uint64_t workspaceSize = 0;
    //  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    //  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    //
    //  ut.TestPrecision();
}

TEST_F(HardswishContiguousTest, aclnnInplaceHardswishStridedSlice)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {4, 2}, 0, {2, 4});

    auto ut = OP_API_UT(aclnnInplaceHardswish, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(HardswishContiguousTest, aclnnInplaceFillStridedSlice)
{
    // TODO
    //  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND, {4, 2}, 0, {8});
    //  auto scalarDesc = ScalarDesc(2);
    //
    //  auto ut = OP_API_UT(aclnnInplaceFillScalar, INPUT(selfDesc, scalarDesc), OUTPUT());
    //
    //  uint64_t workspaceSize = 0;
    //  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    //  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    //
    //  ut.TestPrecision();
}
