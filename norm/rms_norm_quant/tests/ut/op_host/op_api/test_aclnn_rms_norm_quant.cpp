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

#include "../../../../op_host/op_api/aclnn_rms_norm_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_rms_norm_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "rms_norm_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "rms_norm_quant_test TearDown" << endl; }

    struct TensorCase {
        array<vector<int64_t>, 6> shapes = {{{2, 16}, {16}, {16}, {1}, {1}, {2, 16}}};
        array<aclDataType, 6> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, ACL_INT8};
    };

    aclnnStatus RunTensorCase(const TensorCase& testCase)
    {
        auto xDesc = TensorDesc(testCase.shapes[0], testCase.dtypes[0], ACL_FORMAT_ND);
        auto gammaDesc = TensorDesc(testCase.shapes[1], testCase.dtypes[1], ACL_FORMAT_ND);
        auto betaDesc = TensorDesc(testCase.shapes[2], testCase.dtypes[2], ACL_FORMAT_ND);
        auto scaleDesc = TensorDesc(testCase.shapes[3], testCase.dtypes[3], ACL_FORMAT_ND);
        auto offsetDesc = TensorDesc(testCase.shapes[4], testCase.dtypes[4], ACL_FORMAT_ND);
        auto yDesc = TensorDesc(testCase.shapes[5], testCase.dtypes[5], ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRmsNormQuant, INPUT(xDesc, gammaDesc, betaDesc, scaleDesc, offsetDesc, 1e-6, yDesc),
                            OUTPUT());
        uint64_t workspaceSize = 0;
        return ut.TestGetWorkspaceSize(&workspaceSize);
    }
};

TEST_F(l2_rms_norm_quant_test, case_fp16_001)
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
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnRmsNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, eps, tensor_desc_y),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_test, case_bf16_002)
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
    double eps = 1e-6;

    auto ut = OP_API_UT(
        aclnnRmsNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_gamma, tensor_desc_beta, tensor_desc_s, tensor_desc_o, eps, tensor_desc_y),
        OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_test, ascend950_rank2_scale_returns_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto xDesc = TensorDesc({2, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 16}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRmsNormQuant, INPUT(xDesc, gammaDesc, betaDesc, scaleDesc, offsetDesc, 1e-6, yDesc),
                        OUTPUT());

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rms_norm_quant_test, ascend950_invalid_norm_leading_dim_returns_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (size_t tensorIndex : {1U, 2U}) {
        for (int64_t lastDim : {1, 16}) {
            SCOPED_TRACE("tensor index: " + to_string(tensorIndex) + ", last dim: " + to_string(lastDim));
            TensorCase testCase;
            testCase.shapes[0] = {2, lastDim};
            testCase.shapes[1] = {lastDim};
            testCase.shapes[2] = {lastDim};
            testCase.shapes[5] = {2, lastDim};
            testCase.shapes[tensorIndex] = {2, lastDim};
            EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
        }
    }
}

TEST_F(l2_rms_norm_quant_test, ascend950_invalid_quant_parameter_shapes_return_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorCase testCase;
    testCase.shapes[3] = {15};
    testCase.shapes[4] = {15};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);

    testCase.shapes[3] = {16};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);

    testCase.shapes[4] = {1, 16};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rms_norm_quant_test, ascend950_leading_one_and_quant_scale_modes_remain_valid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (bool gammaLeadingOne : {false, true}) {
        for (bool betaLeadingOne : {false, true}) {
            for (int64_t scaleLength : {1, 16}) {
                SCOPED_TRACE("gamma leading one: " + to_string(gammaLeadingOne) + ", beta leading one: " +
                             to_string(betaLeadingOne) + ", scale length: " + to_string(scaleLength));
                TensorCase testCase;
                testCase.shapes[1] = gammaLeadingOne ? vector<int64_t>{1, 16} : vector<int64_t>{16};
                testCase.shapes[2] = betaLeadingOne ? vector<int64_t>{1, 16} : vector<int64_t>{16};
                testCase.shapes[3] = {scaleLength};
                testCase.shapes[4] = {scaleLength};
                EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_rms_norm_quant_test, ascend950_null_gamma_and_beta_remain_param_nullptr)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto xDesc = TensorDesc({2, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto normDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto offsetDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto nullGammaUt = OP_API_UT(aclnnRmsNormQuant, INPUT(xDesc, nullptr, normDesc, scaleDesc, offsetDesc, 1e-6, yDesc),
                                 OUTPUT());
    auto nullBetaUt = OP_API_UT(aclnnRmsNormQuant, INPUT(xDesc, normDesc, nullptr, scaleDesc, offsetDesc, 1e-6, yDesc),
                                OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(nullGammaUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(nullBetaUt.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rms_norm_quant_test, legacy_norm_preprocessing_behavior_is_unchanged)
{
    for (auto socVersion : {op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93}) {
        op::SocVersionManager versionManager(socVersion);
        SCOPED_TRACE("soc version: " + to_string(static_cast<int>(socVersion)));
        TensorCase testCase;
        testCase.dtypes[3] = ACL_FLOAT16;
        testCase.dtypes[4] = ACL_INT8;
        testCase.shapes[1] = {1, 16};
        testCase.shapes[2] = {1, 16};
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);

        // The new early shape check is RegBase-only; legacy reshape failures keep their previous status.
        testCase.shapes[1] = {2, 16};
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_INNER_NULLPTR);
        testCase.shapes[1] = {1, 16};
        testCase.shapes[2] = {2, 16};
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_INNER_NULLPTR);
    }
}
