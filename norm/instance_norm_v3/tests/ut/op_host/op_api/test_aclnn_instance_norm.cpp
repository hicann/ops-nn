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

#include "../../../../op_host/op_api/aclnn_instance_norm.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_instance_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "instance_norm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "instance_norm_test TearDown" << endl;
    }
};

TEST_F(l2_instance_norm_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> invalidList = {ACL_DOUBLE, ACL_UINT8, ACL_INT8,   ACL_INT16,     ACL_INT32,
                                       ACL_INT64,  ACL_BOOL,  ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    const char* dataFormat = "NCHW";
    double eps = 1e-5;
    int length = invalidList.size();

    for (int i = 0; i < length; i++) {
        auto xDesc = TensorDesc({2, 8, 8, 16}, invalidList[i], ACL_FORMAT_ND);
        auto gammaDesc = TensorDesc({8}, invalidList[i], ACL_FORMAT_ND);
        auto betaDesc = TensorDesc({8}, invalidList[i], ACL_FORMAT_ND);
        auto yDesc = TensorDesc({2, 8, 8, 16}, invalidList[i], ACL_FORMAT_ND);
        auto meanDesc = TensorDesc({2, 8, 1, 1}, invalidList[i], ACL_FORMAT_ND);
        auto varDesc = TensorDesc({2, 8, 1, 1}, invalidList[i], ACL_FORMAT_ND);

        auto ut = OP_API_UT(
            aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_instance_norm_test, case_x_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT((aclTensor*)nullptr, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_gamma_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, (aclTensor*)nullptr, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_beta_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, (aclTensor*)nullptr, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_y_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT((aclTensor*)nullptr, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_mean_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, (aclTensor*)nullptr, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_var_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, (aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_dataformat_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, (const char*)nullptr, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_gamma_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_beta_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_y_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_mean_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_var_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_x_not_4d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_x_5d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_gamma_not_1d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_beta_not_1d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_gamma_shape_ne_c_nchw_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_gamma_shape_ne_c_nhwc_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_beta_shape_ne_c_nchw_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_beta_shape_ne_c_nhwc_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_dataformat_invalid_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "INVALID";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_dataformat_empty_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}