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
#include "../../../op_host/op_api/aclnn_hardswish_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_hardswish_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_hardswish_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_hardswish_backward_test TearDown" << std::endl;
    }
};

// 正常场景_float32_nd
TEST_F(l2_hardswish_backward_test, normal_dtype_float32_format_nd)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_float32_nhwc
TEST_F(l2_hardswish_backward_test, normal_dtype_float32_format_nhwc)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_float32_nchw
TEST_F(l2_hardswish_backward_test, normal_dtype_float32_format_nchw)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_float32_hwcn
TEST_F(l2_hardswish_backward_test, normal_dtype_float32_format_hwcn)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_float32_ncdhw
TEST_F(l2_hardswish_backward_test, normal_dtype_float32_format_ncdhw)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 正常场景_float32_ndhwc
TEST_F(l2_hardswish_backward_test, normal_dtype_float32_format_ndhwc)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_hardswish_backward_test, normal_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// CheckNotNull_1
TEST_F(l2_hardswish_backward_test, abnormal_grad_output_nullptr)
{
    auto gradOutputDesc = nullptr;
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_2
TEST_F(l2_hardswish_backward_test, abnormal_self_nullptr)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = nullptr;
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_3
TEST_F(l2_hardswish_backward_test, abnormal_out_nullptr)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_1
TEST_F(l2_hardswish_backward_test, abnormal_dtype_grad_output_int64)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_2
TEST_F(l2_hardswish_backward_test, abnormal_dtype_self_int64)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_3
TEST_F(l2_hardswish_backward_test, abnormal_dtype_grad_output_self_unequal)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_1
TEST_F(l2_hardswish_backward_test, abnormal_shape_grad_output_self_unequal)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_2
TEST_F(l2_hardswish_backward_test, abnormal_shape_dim_greater_than_threshold)
{
    auto gradOutputDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景_bfloat16_nd
TEST_F(l2_hardswish_backward_test, ascend910B2_hardswish_grad_bf16_precision)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardswishBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}