/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_apply_adam_w_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_apply_adam_w_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_apply_adam_w_v2_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_apply_adam_w_v2_test TearDown" << std::endl;
    }
};

TEST_F(l2_apply_adam_w_v2_test, case_nullptr)
{
    auto var = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto m = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto v = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxGradNorm = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = true;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(nullptr, m, v, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, nullptr, v, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, nullptr, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(
        aclnnApplyAdamWV2, INPUT(var, m, v, nullptr, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, v, maxGradNorm, nullptr, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize), OUTPUT());
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut5 = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, v, maxGradNorm, grad, nullptr, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize), OUTPUT());
    aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_apply_adam_w_v2_test, case_null_tensor)
{
    auto var = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto m = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto v = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxGradNorm = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = true;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, v, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_apply_adam_w_v2_test, case_error_dtype)
{
    auto var = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto m = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto v = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto grad = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = true;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, v, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_apply_adam_w_v2_test, case_error_shape)
{
    auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto m = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto vERROR = TensorDesc({4, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto stepERROR = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = true;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, vERROR, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut1 = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, v, maxGradNorm, grad, stepERROR, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_apply_adam_w_v2_test, case_maxGradNorm_is_not_null)
{
    auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto m = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = true;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, v, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_apply_adam_w_v2_test, case_maxGradNorm_is_null)
{
    auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto m = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = false;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2, INPUT(var, m, v, nullptr, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_apply_adam_w_v2_test, case_diff_dtype)
{
    auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto m = TensorDesc({4, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = true;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2,
        INPUT(var, m, v, maxGradNorm, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_apply_adam_w_v2_test, case_diff_dtype_and_maxGradNorm_is_null)
{
    auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto m = TensorDesc({4, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto step = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    float lr = 0.001f;
    float beta1 = 0.01f;
    float beta2 = 0.09f;
    float weightDecay = 0.0001f;
    float eps = 0.000001f;
    bool amsgrad = false;
    bool maximize = true;
    auto ut = OP_API_UT(
        aclnnApplyAdamWV2, INPUT(var, m, v, nullptr, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize),
        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}