/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../../op_host/op_api/aclnn_layer_norm_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_layer_norm_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "layer_norm_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "layer_norm_quant_test TearDown" << endl; }
};

TEST_F(l2_layer_norm_quant_test, ascend950_case_static_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_layer_norm_quant_test, ascend910b_case_static_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_layer_norm_quant_test, ascend910b_case_float16_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_layer_norm_quant_test, ascend910b_case_non_aligned_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 63}, ACL_FLOAT16, ACL_FORMAT_ND); // 非对齐形状
    auto tensor_desc_gamma = TensorDesc({1, 63}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 63}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 63}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gamma为1维的正向回归，覆盖2维首维校验的旁路分支
TEST_F(l2_layer_norm_quant_test, case_static_gamma_1d_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gamma为2维时第一维必须为1，否则其余行会被静默丢弃
TEST_F(l2_layer_norm_quant_test, case_static_err_gamma_first_dim_not_1)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// zeroPointsOptional的shape需要与scale保持一致（即[1]），否则其余元素会被静默丢弃
TEST_F(l2_layer_norm_quant_test, case_static_err_zero_points_shape_invalid)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            5,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scaleOut的dtype必须为FLOAT
TEST_F(l2_layer_norm_quant_test, case_static_err_scale_out_dtype_invalid)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            8,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scaleOut的shape为x的shape剔除最后一维
TEST_F(l2_layer_norm_quant_test, case_static_err_scale_out_shape_invalid)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = TensorDesc(
        {
            9,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scaleOut不允许为空指针
TEST_F(l2_layer_norm_quant_test, case_static_err_scale_out_nullptr)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_s = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o = TensorDesc(
        {
            1,
        },
        ACL_INT8, ACL_FORMAT_ND);

    auto tensor_desc_y = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_os = nullptr;
    int quantMode = 0;
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, quantMode, eps),
        OUTPUT(tensor_desc_y, tensor_desc_os));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
