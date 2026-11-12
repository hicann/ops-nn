/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "level2/aclnn_threshold_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_threshold_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_threshold_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_threshold_backward_test TearDown" << std::endl; }
};

// 计算的数据类型不在支持范围内
TEST_F(l2_threshold_backward_test, l2_test_unsupport_dtype)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalar_desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入的数据类型不匹配
TEST_F(l2_threshold_backward_test, l2_test_unmatch_dtype)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalar_desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输入的shape不一致场景
TEST_F(l2_threshold_backward_test, l2_test_unmatch_shape)
{
    auto gradOutputDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalar_desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 输出shape不可用场景
TEST_F(l2_threshold_backward_test, l2_test_invalid_out_shape)
{
    auto gradOutputDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalar_desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空指针
TEST_F(l2_threshold_backward_test, l2_test_nullptr)
{
    auto gradOutputDesc = TensorDesc({2, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(nullptr, selfDesc, scalar_desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensor
TEST_F(l2_threshold_backward_test, l2_test_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({2, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_INT32, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.0f);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalar_desc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_threshold_backward_test, ascend950PR_9589_l2_test_relu_grad_int32_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_INT32, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, ascend950PR_9589_l2_test_relu_grad_int8_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_INT8, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_INT8, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, ascend950PR_9589_l2_test_relu_grad_fp16_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, ascend950PR_9589_l2_test_relu_grad_fp32_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, ascend910B2_l2_test_threshold_grad_v2_d_positive_threshold_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_INT32, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, ascend910B2_l2_test_threshold_grad_v2_d_negative_threshold_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_INT32, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(-1.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, ascend910B2_l2_test_relu_grad_bfp16_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 校验维度超过8维的场景
TEST_F(l2_threshold_backward_test, ascend910B2_l2_test_relu_grad_check_max_dim)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarSesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarSesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_threshold_backward_test, l2_test_relu_grad_uint8_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_UINT8, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_UINT8, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_relu_grad_int64_unsupported_on_stub)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_INT64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_INT64, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_threshold_backward_test, l2_test_int64_nonzero_threshold_fail)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_INT64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_INT64, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_threshold_backward_test, l2_test_2d_shape_success)
{
    auto gradOutputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.5f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_4d_shape_success)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_8d_shape_success)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_threshold_backward_test, l2_test_large_shape_success)
{
    auto gradOutputDesc = TensorDesc({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1024, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_threshold_backward_test, l2_test_bf16_threshold_zero_precision)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_uint8_nonzero_threshold_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_UINT8, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_UINT8, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(1.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_fp32_nonzero_threshold_precision)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(2.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_scalar_tensor_success)
{
    auto gradOutputDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalarDesc = ScalarDesc(0.0f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_threshold_backward_test, l2_test_threshold_nullptr)
{
    auto gradOutputDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            10,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, (const aclScalar*)nullptr),
                        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// issue #8230：threshold是host侧标量，判零须精确比较，|threshold|<=FLT_EPSILON的非零阈值
// 不得被静默当作0走ReluGrad分支（丢弃threshold），应按公式 self(i) > threshold 走
// ThresholdGradV2D分支。以下用例覆盖容差判零边界（UT为host第一段校验，精度由ST/repro覆盖）
TEST_F(l2_threshold_backward_test, l2_test_threshold_flt_epsilon_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    // 1.1920928955078125e-07f 即 2^-23（FLT_EPSILON），恰在原容差判零分界点上，
    // 修复前会被当作0静默路由到ReluGrad
    auto scalarDesc = ScalarDesc(1.1920928955078125e-07f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_threshold_nextafter_epsilon_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    // nextafter(FLT_EPSILON)在原容差外，修复前后行为一致（对照组）
    auto scalarDesc = ScalarDesc(1.1920930e-7f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_threshold_tiny_positive_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    // 远小于FLT_EPSILON的非零正阈值，修复前同样被容差判零吞掉
    auto scalarDesc = ScalarDesc(1e-9f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_threshold_backward_test, l2_test_threshold_tiny_negative_success)
{
    auto gradOutputDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    // 负侧小阈值：修复前|threshold|<=FLT_EPSILON同样被归零路由到ReluGrad，丢失负阈值语义
    auto scalarDesc = ScalarDesc(-1e-7f);
    auto outDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnThresholdBackward, INPUT(gradOutputDesc, selfDesc, scalarDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    ut.TestPrecision();
}
