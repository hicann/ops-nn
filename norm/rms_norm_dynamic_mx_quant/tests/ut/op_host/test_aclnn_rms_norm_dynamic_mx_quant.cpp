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

#include "../../../op_host/op_api/aclnn_rms_norm_dynamic_mx_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace std;

constexpr int64_t GE_DT_FLOAT8_E5M2 = 35;
constexpr int64_t GE_DT_FLOAT8_E4M3FN = 36;
constexpr int64_t GE_DT_FLOAT4_E2M1 = 40;
constexpr int64_t GE_DT_FLOAT4_E1M2 = 41;

class l2_rms_norm_dynamic_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "rms_norm_dynamic_mx_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "rms_norm_dynamic_mx_quant_test TearDown" << endl;
    }
};

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e5m2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E5M2;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp4_e2m1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT4_E2M1;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp4_e1m2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT4_E1M2;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_bf16_fp8_e4m3fn)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_bf16_fp8_e5m2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E5M2;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_bf16_fp4_e2m1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT4_E2M1;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_bf16_fp4_e1m2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT4_E1M2;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_with_beta)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
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
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_beta, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_no_rstd)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = false;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_scale_alg_1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 1;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e5m2_scale_alg_1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 1;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E5M2;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp4_e2m1_floor)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("floor");
    int64_t dstType = GE_DT_FLOAT4_E2M1;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp4_e2m1_round)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("round");
    int64_t dstType = GE_DT_FLOAT4_E2M1;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp4_e1m2_floor)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("floor");
    int64_t dstType = GE_DT_FLOAT4_E1M2;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp4_e1m2_round)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({8, 64}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("round");
    int64_t dstType = GE_DT_FLOAT4_E1M2;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_large_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            256,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({256, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({256, 4, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({256, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_3d)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({4, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            4,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({4, 3, 4}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({4, 3, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({4, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_4d)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            5,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({2, 3, 4, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({2, 3, 4, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({2, 3, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_unaligned)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({33, 17}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            17,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({33, 17}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({33, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({33, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_fp16_fp8_e4m3fn_single_element)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_dynamic_mx_quant_test, ascend950_case_bf16_fp8_e4m3fn_large_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({1024}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            1024,
        },
        ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y = TensorDesc({1024}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_mxscale = TensorDesc({16, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    int64_t scaleAlg = 0;
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormDynamicMxQuant,
        INPUT(tensor_desc_x, tensor_desc_gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd),
        OUTPUT(tensor_desc_y, tensor_desc_mxscale, tensor_desc_rstd));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}