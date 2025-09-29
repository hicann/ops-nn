/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_scaled_masked_softmax_backward.cpp
 * \brief
 */
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "level2/aclnn_scaled_masked_softmax_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_scaled_masked_softmax_backward_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_scaled_masked_softmax_backward_v2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_scaled_masked_softmax_backward_v2_test TearDown" << endl;
    }
};

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_null_y_grad)
{
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnScaledMaskedSoftmaxBackward, INPUT((aclTensor*)nullptr, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_null_y)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, (aclTensor*)nullptr, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_null_mask)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, (aclTensor*)nullptr, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_null_output)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT((aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_dtype_y_grad)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_INT32, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_dtype_mask)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_format)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_NCHW);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_shape_y_grad)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, -1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_dim_y_grad)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_dim_y_grad_and_y)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_dim_mask)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_shape_mask)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, -1}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_error_broadcast)
{
    TensorDesc yGradDesc = TensorDesc({4, 4, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({4, 4, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({4, 4, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, false), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_backward_v2_test, test_scaled_masked_softmax_backward_wrong_fix_triu_mask)
{
    TensorDesc yGradDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc yDesc = TensorDesc({1, 2, 4, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc maskDesc = TensorDesc({1, 2, 4, 128}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc outDesc = TensorDesc({1, 2, 4, 128}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut =
        OP_API_UT(aclnnScaledMaskedSoftmaxBackward, INPUT(yGradDesc, yDesc, maskDesc, 1.0, true), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
