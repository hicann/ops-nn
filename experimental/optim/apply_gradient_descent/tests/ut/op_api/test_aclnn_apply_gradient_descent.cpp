/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_apply_gradient_descent.h"
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class TestAclnnApplyGradientDescent : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "apply_gradient_descent_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "apply_gradient_descent_test TearDown" << std::endl; }
};

// 正常场景_FLOAT_ND (inplace: var 既是输入也是输出)
TEST_F(TestAclnnApplyGradientDescent, l2_agd_normal_FLOAT_ND)
{
    auto varDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_FLOAT16_ND
TEST_F(TestAclnnApplyGradientDescent, l2_agd_normal_FLOAT16_ND)
{
    auto varDesc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_BF16_ND
TEST_F(TestAclnnApplyGradientDescent, l2_agd_normal_BF16_ND)
{
    auto varDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常场景_3D
TEST_F(TestAclnnApplyGradientDescent, l2_agd_normal_3d)
{
    auto varDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空tensor场景
TEST_F(TestAclnnApplyGradientDescent, l2_agd_normal_empty_tensor)
{
    auto varDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常场景_var为空指针
TEST_F(TestAclnnApplyGradientDescent, l2_agd_abnormal_var_nullptr)
{
    auto varDesc = nullptr;
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景_alpha为空指针
TEST_F(TestAclnnApplyGradientDescent, l2_agd_abnormal_alpha_nullptr)
{
    auto varDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = nullptr;
    auto deltaDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景_delta为空指针
TEST_F(TestAclnnApplyGradientDescent, l2_agd_abnormal_delta_nullptr)
{
    auto varDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = nullptr;

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景_非法dtype(INT32)
TEST_F(TestAclnnApplyGradientDescent, l2_agd_abnormal_dtype_int32)
{
    auto varDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_dtype不一致(var fp32, delta fp16)
TEST_F(TestAclnnApplyGradientDescent, l2_agd_abnormal_dtype_mismatch)
{
    auto varDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_var/delta shape 不一致
TEST_F(TestAclnnApplyGradientDescent, l2_agd_abnormal_shape_mismatch)
{
    auto varDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_alpha 非标量(numel != 1)
TEST_F(TestAclnnApplyGradientDescent, l2_agd_abnormal_alpha_not_scalar)
{
    auto varDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto deltaDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnApplyGradientDescent, INPUT(varDesc, alphaDesc, deltaDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
