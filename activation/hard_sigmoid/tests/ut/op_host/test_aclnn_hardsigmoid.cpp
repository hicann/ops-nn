/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_hardsigmoid.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_hardsigmoid_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_hardsigmoid_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_hardsigmoid_test TearDown" << std::endl;
    }
};

// self的数据类型不在支持范围内
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_001)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self和out的shape类型不一致
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_002)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self和out的shape类型不一致
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_003)
{
    auto selfDesc = TensorDesc({2, 3, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_004)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_005)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常路径，float16
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_006)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int32
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_007)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，bfloat16
TEST_F(l2_hardsigmoid_test, ascend910B2_hardsigmoid_test_008)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 输入dtype不支持
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_008)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入dtype不支持
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_009)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入输出shape不同
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_010)
{
    auto selfDesc = TensorDesc({2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32，9维张量
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_011)
{
    auto selfDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32，8维张量
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_012)
{
    auto selfDesc = TensorDesc({5, 2, 3, 4, 6, 1, 7, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({5, 2, 3, 4, 6, 1, 7, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 0维场景
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_013)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT(selfDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
// 空指针
TEST_F(l2_hardsigmoid_test, l2_hardsigmoid_test_014)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(aclnnHardsigmoid, INPUT((aclTensor*)nullptr), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}