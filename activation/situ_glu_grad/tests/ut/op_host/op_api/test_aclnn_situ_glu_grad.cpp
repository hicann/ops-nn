/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_situ_glu_grad.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_situ_glu_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "situ_glu_grad_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "situ_glu_grad_test TearDown" << std::endl; }
};

// 正常场景_float32_nd_最后维切分_activate_left
TEST_F(l2_situ_glu_grad_test, normal_dtype_float32_format_nd)
{
    auto gradYDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_float16_nd_activate_left为false
TEST_F(l2_situ_glu_grad_test, normal_dtype_float16_format_nd_activate_right)
{
    auto gradYDesc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 1.0;
    bool activateLeftDesc = false;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_bfloat16_nd_dim为0
TEST_F(l2_situ_glu_grad_test, normal_dtype_bfloat16_format_nd_dim_zero)
{
    auto gradYDesc = TensorDesc({2, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto xDesc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int64_t dimDesc = 0;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_非连续出参视图_strides与连续布局不一致
TEST_F(l2_situ_glu_grad_test, normal_noncontiguous_out_strided_view)
{
    auto gradYDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto xDesc = TensorDesc({2, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    // 出参为非连续视图：view[2,32]，strides{64,2}，storage[2,64]
    auto outDesc = TensorDesc({2, 32}, ACL_FLOAT, ACL_FORMAT_ND, {64, 2}, 0, {2, 64}).Precision(0.0001, 0.0001);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_非连续出参视图_带padding的storage
TEST_F(l2_situ_glu_grad_test, normal_noncontiguous_out_padded_view)
{
    auto gradYDesc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto xDesc = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    // 出参为带padding的非连续视图：view[4,8]，strides{16,1}，storage[4,16]
    auto outDesc = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {16, 1}, 0, {4, 16}).Precision(0.001, 0.001);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_空tensor
TEST_F(l2_situ_glu_grad_test, normal_empty_tensor)
{
    auto gradYDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常场景_gradY为空指针
TEST_F(l2_situ_glu_grad_test, abnormal_grady_nullptr)
{
    auto gradYDesc = nullptr;
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 异常场景_out数据类型与x不一致
TEST_F(l2_situ_glu_grad_test, abnormal_out_dtype_mismatch)
{
    auto gradYDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 异常场景_dim超出范围
TEST_F(l2_situ_glu_grad_test, abnormal_dim_out_of_range)
{
    auto gradYDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = 2;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 异常场景_切分维度为奇数
TEST_F(l2_situ_glu_grad_test, abnormal_split_dim_odd)
{
    auto gradYDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 15}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 15}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 异常场景_gradY非切分维度与x不一致
TEST_F(l2_situ_glu_grad_test, abnormal_grady_shape_mismatch)
{
    auto gradYDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 异常场景_out shape与x不一致
TEST_F(l2_situ_glu_grad_test, abnormal_out_shape_mismatch)
{
    auto gradYDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    double betaDesc = 1.0;
    double linearBetaDesc = 0.0;
    bool activateLeftDesc = true;

    auto ut = OP_API_UT(aclnnSituGluGrad, INPUT(gradYDesc, xDesc, dimDesc, betaDesc, linearBetaDesc, activateLeftDesc),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// 异常场景_workspaceSize为空指针，第一段直接返回161001不解引用（参照PR !14239检视意见）
TEST_F(l2_situ_glu_grad_test, abnormal_workspacesize_nullptr)
{
    auto gradYDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* gradY = gradYDesc.ToAclTypeRawPtr();
    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* out = outDesc.ToAclTypeRawPtr();
    ASSERT_NE(gradY, nullptr);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(out, nullptr);

    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = aclnnSituGluGradGetWorkspaceSize(gradY, x, -1, 1.0, 0.0, true, out, nullptr, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    Release(gradY);
    Release(x);
    Release(out);
}

// 异常场景_executor为空指针，第一段直接返回161001不使用
TEST_F(l2_situ_glu_grad_test, abnormal_executor_nullptr)
{
    auto gradYDesc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* gradY = gradYDesc.ToAclTypeRawPtr();
    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* out = outDesc.ToAclTypeRawPtr();
    ASSERT_NE(gradY, nullptr);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(out, nullptr);

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = aclnnSituGluGradGetWorkspaceSize(gradY, x, -1, 1.0, 0.0, true, out, &workspaceSize, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    Release(gradY);
    Release(x);
    Release(out);
}
