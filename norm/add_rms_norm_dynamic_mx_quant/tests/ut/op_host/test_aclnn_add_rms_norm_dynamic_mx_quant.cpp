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

#include "../../../op_host/op_api/aclnn_add_rms_norm_dynamic_mx_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace std;

class l2_add_rms_norm_dynamic_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_dynamic_mx_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "add_rms_norm_dynamic_mx_quant_test TearDown" << endl;
    }
};

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_001)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 36;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_002)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 35;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, nullptr, epsilon, scale_alg, round_mode, dst_type,
            output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_003)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 36;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_004)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 40;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_005)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 41;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_006)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("floor");
    int64_t dst_type = 40;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_007)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 1;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 36;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_008)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 36;
    bool output_rstd = false;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_009)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            128,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            128,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({8, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({8, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 36;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_010)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({2, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({2, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            32,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({2, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({2, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({2, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 36;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, nullptr, epsilon, scale_alg, round_mode, dst_type,
            output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_011)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({1, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({1, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 35;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_mx_quant_test, ascend950_case_012)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x_out = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_mxscale_out = TensorDesc({16, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({16, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    int64_t scale_alg = 0;
    char* round_mode = const_cast<char*>("rint");
    int64_t dst_type = 36;
    bool output_rstd = true;

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicMxQuant,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, epsilon, scale_alg, round_mode,
            dst_type, output_rstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_x_out, tensor_desc_mxscale_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}