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
#include "../../../op_host/op_api/aclnn_hardsigmoid_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_hardsigmoid_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_hardsigmoid_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_hardsigmoid_backward_test TearDown" << std::endl;
    }
};

TEST_F(l2_hardsigmoid_backward_test, grad_nullptr)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT((aclTensor*)nullptr, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_hardsigmoid_backward_test, self_nullptr)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, (aclTensor*)nullptr), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_hardsigmoid_backward_test, out_nullptr)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_hardsigmoid_backward_test, gradOutput_invalid_dtype)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, self_invalid_dtype_int32)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, self_invalid_dtype_bool)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, self_invalid_shape_9_dim)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, gradOutput_invalid_shape_9_dim)
{
    auto gradOutputDesc = TensorDesc({2, 3, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, gradOutput_self_shape_not_equal)
{
    auto gradOutputDesc = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, self_out_shape_not_equal)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_hardsigmoid_backward_test, empty_tensor_2_0_3)
{
    auto gradOutputDesc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_hardsigmoid_backward_test, float32_nd)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, float16_nd)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, ascend910B2_bfloat16_nd)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, grad_float16_self_float_nd)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, self_transpose)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {2, 1}, 0, {3, 2});
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}
