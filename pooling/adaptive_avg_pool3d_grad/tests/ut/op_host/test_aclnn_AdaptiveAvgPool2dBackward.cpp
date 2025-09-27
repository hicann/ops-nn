/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_adaptive_avg_pool2d_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_adaptive_avg_pool2d_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "_adaptive_avg_pool2d_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "_adaptive_avg_pool2d_backward_test TearDown" << endl;
    }
};

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_NCHW)
{
    aclDataType dtype = ACL_FLOAT;

    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}
TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_CHW)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({2, 1, 1}, dtype, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2}, dtype, ACL_FORMAT_NCL).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2}, dtype, ACL_FORMAT_NCL);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}
TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_NCHW_fp16)
{
    aclDataType dtype = ACL_FLOAT16;

    auto grad_output_tensor_desc = TensorDesc({2, 2, 1, 1}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2}, dtype, ACL_FORMAT_NCHW).Precision(0.03, 0.03);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}
TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_CHW_fp16)
{
    aclDataType dtype = ACL_FLOAT16;
    auto grad_output_tensor_desc = TensorDesc({2, 1, 1}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2}, dtype, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2}, dtype, ACL_FORMAT_NCHW).Precision(0.03, 0.03);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));

    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_shape_not_support)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({2, 2, 2, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    // ut.TestPrecision();
}

TEST_F(
    l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_shape_not_match_not_support)
{
    aclDataType dtype = ACL_FLOAT;

    auto grad_output_tensor_desc = TensorDesc({2, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_dtype_not_support)
{
    aclDataType dtype = ACL_DOUBLE;
    auto grad_output_tensor_desc = TensorDesc({2, 1, 1}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, dtype, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    // ut.TestPrecision();
}

TEST_F(
    l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_outputsize_one_one_dtype_not_match_not_support)
{
    auto grad_output_tensor_desc = TensorDesc({2, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto output_tensor = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_FLOAT_NCHW)
{
    auto grad_output_tensor_desc =
        TensorDesc({2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{0.3500, 0.3500, 0.3500, 0.2500, 0.2500,
                                                                                 0.2500, 0.3500, 0.3500, 0.3500, 0.2500,
                                                                                 0.2500, 0.2500, 0.3500, 0.3500, 0.3500,
                                                                                 0.2500, 0.2500, 0.2500, 0.3500, 0.3500,
                                                                                 0.3500, 0.2500, 0.2500, 0.2500});
    auto input_tensor_desc =
        TensorDesc({2, 2, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW)
            .Value(vector<float>{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1, 0.1,
                                 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
                                 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.1, 0.1,
                                 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3});
    auto output_tensor = TensorDesc({2, 2, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.03, 0.03);
    // auto output_tensor = TensorDesc({1,1,1,1,16,16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_FLOAT_CHW)
{
    auto grad_output_tensor_desc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({2, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({2, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.03, 0.03);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_FLOAT_NCHW1)
{
    auto grad_output_tensor_desc = TensorDesc({1, 1, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 32, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 32, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.03, 0.03);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_FLOAT_CHW1)
{
    auto grad_output_tensor_desc = TensorDesc({1, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 32, 16}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 32, 16}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.03, 0.03);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}
TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_Fp16_NCHW1)
{
    aclDataType dtype = ACL_FLOAT16;
    auto grad_output_tensor_desc = TensorDesc({1, 1, 16, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 32, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 32, 16}, dtype, ACL_FORMAT_NCHW).Precision(0.03, 0.03);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_Fp16_CHW1)
{
    aclDataType dtype = ACL_FLOAT16;
    auto grad_output_tensor_desc = TensorDesc({1, 16, 16}, dtype, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 32, 16}, dtype, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 32, 16}, dtype, ACL_FORMAT_NCL).Precision(0.03, 0.03);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, ascend910B2_adaptive_avg_pool2d_backward_Fp16_CHW1)
{
    aclDataType dtype = ACL_BF16;
    auto grad_output_tensor_desc = TensorDesc({1, 16, 16}, dtype, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 32, 16}, dtype, ACL_FORMAT_NCL).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 32, 16}, dtype, ACL_FORMAT_NCL).Precision(0.03, 0.03);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_input_null_tensor)
{
    auto grad_output_tensor_desc = TensorDesc({1, 1, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 0, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 0, 16}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_FLOAT_shape_not_support)
{
    auto grad_output_tensor_desc = TensorDesc({1, 1, 1, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 1, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_FLOAT_shape_not_match_not_support)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({1, 16, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 1, 32, 16}, dtype, ACL_FORMAT_ND).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 1, 32, 16}, dtype, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_FLOAT_shape_not_match_not_support1)
{
    aclDataType dtype = ACL_FLOAT;
    auto grad_output_tensor_desc = TensorDesc({1, 16, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 32, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 32, 16}, dtype, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_dtype_not_support)
{
    vector<aclDataType> dtypes{ACL_INT64, ACL_UINT64, ACL_BF16};
    for (auto dtype : dtypes) {
        auto grad_output_tensor_desc = TensorDesc({1, 1, 16, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
        auto input_tensor_desc = TensorDesc({1, 16, 32, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
        auto output_tensor = TensorDesc({1, 16, 32, 16}, dtype, ACL_FORMAT_NCHW);

        auto ut = OP_API_UT(
            aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, ascend910B2_adaptive_avg_pool2d_backward_dtype_not_support)
{
    vector<aclDataType> dtypes{ACL_INT64, ACL_UINT64};
    for (auto dtype : dtypes) {
        auto grad_output_tensor_desc = TensorDesc({1, 1, 16, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
        auto input_tensor_desc = TensorDesc({1, 16, 32, 16}, dtype, ACL_FORMAT_NCHW).ValueRange(-50, 50);
        auto output_tensor = TensorDesc({1, 16, 32, 16}, dtype, ACL_FORMAT_NCHW);

        auto ut = OP_API_UT(
            aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_dtype_not_match_not_support)
{
    auto grad_output_tensor_desc = TensorDesc({1, 1, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 32, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 32, 16}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_gradoutput_null_tensor)
{
    auto grad_output_tensor_desc = TensorDesc({1, 1, 0, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto input_tensor_desc = TensorDesc({1, 1, 32, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({1, 1, 32, 16}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_gradoutput_discontinue)
{
    auto grad_output_tensor_desc =
        TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);

    auto input_tensor_desc = TensorDesc({2, 2, 6, 7}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);
    auto output_tensor = TensorDesc({2, 2, 6, 7}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_self_discontinue)
{
    auto grad_output_tensor_desc = TensorDesc({2, 2, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-50, 50);

    auto input_tensor_desc =
        TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);

    auto output_tensor = TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    ut.TestPrecision();
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_self_not_support_format)
{
    auto grad_output_tensor_desc = TensorDesc({2, 2, 4, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-50, 50);

    auto input_tensor_desc =
        TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NHWC, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);

    auto output_tensor = TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_adaptive_avg_pool2d_backward_test, adaptive_avg_pool2d_backward_self_not_support_format2)
{
    auto grad_output_tensor_desc = TensorDesc({2, 2, 4, 3}, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(-50, 50);

    auto input_tensor_desc =
        TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NCL, {40, 20, 1, 5}, 0, {2, 2, 4, 5}).ValueRange(-50, 50);

    auto output_tensor = TensorDesc({2, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnAdaptiveAvgPool2dBackward, INPUT(grad_output_tensor_desc, input_tensor_desc), OUTPUT(output_tensor));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
