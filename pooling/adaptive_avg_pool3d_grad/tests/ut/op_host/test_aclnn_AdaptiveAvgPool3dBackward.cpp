/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_adaptive_avg_pool3d_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_adaptive_avg_pool3d_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "_adaptive_avg_pool3d_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "_adaptive_avg_pool3d_backward_test TearDown" << endl;
    }
};

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_outputsize_one_one_NCDHW_01)
{
    aclDataType dtype = ACL_FLOAT;

    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, ascend910B2_adaptive_avg_pool3d_backward_outputsize_allone_NCDHW_01)
{
    aclDataType dtype = ACL_FLOAT;

    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_outputsize_one_one_CDHW_03)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, ascend910B2_adaptive_avg_pool3d_backward_outputsize_common_CDHW_04)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({2, 1, 2, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_outputsize_one_one_NCDHW_fp16_05)
{
    aclDataType dtype = ACL_FLOAT16;

    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, ascend910B2_adaptive_avg_pool3d_backward_outputsize_one_one_NCDHW_fp16_05)
{
    aclDataType dtype = ACL_FLOAT16;

    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_input_one_one_shape_notmatch_notsupport_07)
{
    aclDataType dtype = ACL_FLOAT;

    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_dtype_not_support_10)
{
    aclDataType dtype = ACL_INT32;
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_dtype_allone_support_11)
{
    aclDataType dtype = ACL_DOUBLE;
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, ascend910B_adaptive_avg_pool3d_backward_dtype_allone_support_11)
{
    aclDataType dtype = ACL_DOUBLE;
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, ascend910B_adaptive_avg_pool3d_backward_common_not_support_11)
{
    aclDataType dtype = ACL_DOUBLE;
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 2}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_dtype_not_support_12)
{
    aclDataType dtype = ACL_COMPLEX64;
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_outsize_one_one_dtype_notmatch_notsupport_13)
{
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_input_one_one_dtype_notmatch_notsupport_14)
{
    auto grad_output_tensor_desc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_FLOAT_CDHW_15)
{
    auto grad_output_tensor_desc = TensorDesc({4, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({4, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({4, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_FLOAT_NCDHW1_14)
{
    auto grad_output_tensor_desc = TensorDesc({4, 4, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({4, 4, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({4, 4, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_FLOAT_CDHW2_15)
{
    auto grad_output_tensor_desc = TensorDesc({5, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({5, 5, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({5, 5, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_FLOAT_NCDHW2_16)
{
    auto grad_output_tensor_desc = TensorDesc({5, 5, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({5, 5, 5, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({5, 5, 5, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

// TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_input_null_tensor_17) {
//   auto grad_output_tensor_desc = TensorDesc({4, 8, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-50, 50);
//   auto input_tensor_desc = TensorDesc({1, 1, 0, 0, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-50, 50);
//   auto output_tensor = TensorDesc({1, 1, 0, 0, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

//   auto ut = OP_API_UT(aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc),
//                                                       OUTPUT(output_tensor));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   ut.TestPrecision();
// }

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_FLOAT_shape_not_equal_18)
{
    auto grad_output_tensor_desc = TensorDesc({3, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_FLOAT_shape_not_4or5_19)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({3, 3, 3, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({3, 3, 3, 3, 3, 3}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({3, 3, 3, 3, 3, 3}, dtype, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_dtype_not_match_grad_input_21)
{
    auto grad_output_tensor_desc = TensorDesc({3, 3, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_dtype_not_match_grad_out_22)
{
    auto grad_output_tensor_desc = TensorDesc({3, 3, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_gradoutput_null_tensor_23)
{
    auto grad_output_tensor_desc = TensorDesc({4, 4, 0, 0, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({4, 4, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({4, 4, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_gradoutput_dim_invalid_24)
{
    auto grad_output_tensor_desc = TensorDesc({2, 2, -1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_self_dim_invalid_25)
{
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto input_tensor_desc = TensorDesc({2, 2, -1, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto output_tensor = TensorDesc({2, 2, -1, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_self_nullptr_26)
{
    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto input_tensor_desc = nullptr;

    auto output_tensor = TensorDesc({2, 2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, adaptive_avg_pool3d_backward_gradoutput_nullptr_27)
{
    auto grad_output_tensor_desc = nullptr;
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);

    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_adaptive_avg_pool3d_backward_test, ascend910B2_adaptive_avg_pool3d_backward_outshape_not_match_selfshape_29)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({2, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 1, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(
    l2_adaptive_avg_pool3d_backward_test, ascend910B2_adaptive_avg_pool3d_backward_gradoutput__n_not_match_selfshape_30)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({3, 1, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool3dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
