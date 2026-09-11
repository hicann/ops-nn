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

#include "norm/rms_norm_quant_v3/op_api/aclnn_rms_norm_quant_v3.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace std;

class l2_rms_norm_quant_v3_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "rms_norm_quant_v3_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "rms_norm_quant_v3_test TearDown" << endl; }

    aclnnStatus RunEmptyTensorCase(size_t emptyTensorIndex, bool outputRstd = true)
    {
        op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
        const vector<int64_t> xShape = emptyTensorIndex == 0 ? vector<int64_t>{0, 64} : vector<int64_t>{8, 64};
        const vector<int64_t> gammaShape = emptyTensorIndex == 1 ? vector<int64_t>{0} : vector<int64_t>{64};
        const vector<int64_t> scaleShape = emptyTensorIndex == 2 ? vector<int64_t>{0} : vector<int64_t>{1};
        const vector<int64_t> offsetShape = emptyTensorIndex == 3 ? vector<int64_t>{0} : vector<int64_t>{1};
        const vector<int64_t> betaShape = emptyTensorIndex == 4 ? vector<int64_t>{0} : vector<int64_t>{64};
        const vector<int64_t> yShape = emptyTensorIndex == 5 ? vector<int64_t>{0, 64} : vector<int64_t>{8, 64};
        const vector<int64_t> rstdShape = emptyTensorIndex == 6 ? vector<int64_t>{0, 1} : vector<int64_t>{8, 1};

        auto xDesc = TensorDesc(xShape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto gammaDesc = TensorDesc(gammaShape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto scaleDesc = TensorDesc(scaleShape, ACL_FLOAT, ACL_FORMAT_ND);
        auto offsetDesc = TensorDesc(offsetShape, ACL_FLOAT, ACL_FORMAT_ND);
        auto betaDesc = TensorDesc(betaShape, ACL_FLOAT16, ACL_FORMAT_ND);
        auto yDesc = TensorDesc(yShape, ACL_INT8, ACL_FORMAT_ND);
        auto rstdDesc = TensorDesc(rstdShape, ACL_FLOAT, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnRmsNormQuantV3,
                            INPUT(xDesc, gammaDesc, scaleDesc, offsetDesc, betaDesc, 1e-5, true, outputRstd),
                            OUTPUT(yDesc, rstdDesc));
        uint64_t workspaceSize = 0;
        return ut.TestGetWorkspaceSize(&workspaceSize);
    }
};

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_001)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    bool divMode = true;
    bool outputRstd = true;

    auto ut = OP_API_UT(aclnnRmsNormQuantV3,
                        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, tensor_desc_offset, tensor_desc_beta,
                              epsilon, divMode, outputRstd),
                        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_002)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    bool divMode = true;
    bool outputRstd = false;

    auto ut = OP_API_UT(
        aclnnRmsNormQuantV3,
        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, nullptr, nullptr, epsilon, divMode, outputRstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_003)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    bool divMode = true;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormQuantV3,
        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, nullptr, nullptr, epsilon, divMode, outputRstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_004)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    bool divMode = true;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormQuantV3,
        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, nullptr, nullptr, epsilon, divMode, outputRstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_005)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    // int4 output is packed into int32
    auto tensor_desc_y_out = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    bool divMode = true;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormQuantV3,
        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, nullptr, nullptr, epsilon, divMode, outputRstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_006)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    bool divMode = false;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormQuantV3,
        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, nullptr, nullptr, epsilon, divMode, outputRstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_007)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({1, 128}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-6;
    bool divMode = true;
    bool outputRstd = true;

    auto ut = OP_API_UT(aclnnRmsNormQuantV3,
                        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, tensor_desc_offset, nullptr, epsilon,
                              divMode, outputRstd),
                        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_case_008)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_y_out = TensorDesc({16, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_rstd_out = TensorDesc({16, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double epsilon = 1e-5;
    bool divMode = true;
    bool outputRstd = true;

    auto ut = OP_API_UT(
        aclnnRmsNormQuantV3,
        INPUT(tensor_desc_x, tensor_desc_gamma, tensor_desc_scale, nullptr, nullptr, epsilon, divMode, outputRstd),
        OUTPUT(tensor_desc_y_out, tensor_desc_rstd_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_empty_tensor_returns_param_invalid)
{
    constexpr size_t tensorCount = 7;
    for (size_t emptyTensorIndex = 0; emptyTensorIndex < tensorCount; ++emptyTensorIndex) {
        SCOPED_TRACE("empty tensor index: " + std::to_string(emptyTensorIndex));
        EXPECT_EQ(RunEmptyTensorCase(emptyTensorIndex), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_disabled_rstd_allows_empty_placeholder)
{
    EXPECT_EQ(RunEmptyTensorCase(6, false), ACLNN_SUCCESS);
}
