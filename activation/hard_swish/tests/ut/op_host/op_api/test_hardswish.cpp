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
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_hardswish_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_hardswish_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_hardswish_test TearDown" << std::endl;
    }
};

// self的数据类型不在支持范围内(INT64)
TEST_F(l2_hardswish_test, l2_hardswish_test_001)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self的数据类型不在支持范围内(INT32)
TEST_F(l2_hardswish_test, l2_hardswish_test_002)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// shape不一致
TEST_F(l2_hardswish_test, l2_hardswish_test_003)
{
    auto selfDesc = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_hardswish_test, l2_hardswish_test_004)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_hardswish_test, l2_hardswish_test_005)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常路径，float16
TEST_F(l2_hardswish_test, l2_hardswish_test_006)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    // ut.TestPrecision();
}

// 正常路径，float32，5维张量
TEST_F(l2_hardswish_test, l2_hardswish_test_007)
{
    auto selfDesc = TensorDesc({2, 4, 3, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 3, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// shape不一致
TEST_F(l2_hardswish_test, l2_hardswish_test_008)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32，NHWC
TEST_F(l2_hardswish_test, l2_hardswish_test_009)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32，HWCN
TEST_F(l2_hardswish_test, l2_hardswish_test_010)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32，9维张量
TEST_F(l2_hardswish_test, l2_hardswish_test_011)
{
    auto selfDesc = TensorDesc({2, 4, 1, 3, 4, 6, 5, 3, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 1, 3, 4, 6, 5, 3, 7}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32，0维张量
TEST_F(l2_hardswish_test, l2_hardswish_test_012)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，bfloat16
TEST_F(l2_hardswish_test, ascend910B2_hardswish_bf16_precision)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswish, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}