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

    struct TensorCase {
        array<vector<int64_t>, 7> shapes = {{{2, 16}, {16}, {1}, {1}, {16}, {2, 16}, {2, 1}}};
        array<aclDataType, 7> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
                                        ACL_FLOAT16, ACL_INT8,    ACL_FLOAT};
    };

    aclnnStatus RunTensorCase(const TensorCase& testCase, bool outputRstd = true)
    {
        auto xDesc = TensorDesc(testCase.shapes[0], testCase.dtypes[0], ACL_FORMAT_ND);
        auto gammaDesc = TensorDesc(testCase.shapes[1], testCase.dtypes[1], ACL_FORMAT_ND);
        auto scaleDesc = TensorDesc(testCase.shapes[2], testCase.dtypes[2], ACL_FORMAT_ND);
        auto offsetDesc = TensorDesc(testCase.shapes[3], testCase.dtypes[3], ACL_FORMAT_ND);
        auto betaDesc = TensorDesc(testCase.shapes[4], testCase.dtypes[4], ACL_FORMAT_ND);
        auto yDesc = TensorDesc(testCase.shapes[5], testCase.dtypes[5], ACL_FORMAT_ND);
        auto rstdDesc = TensorDesc(testCase.shapes[6], testCase.dtypes[6], ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnRmsNormQuantV3,
                            INPUT(xDesc, gammaDesc, scaleDesc, offsetDesc, betaDesc, 1e-6, true, outputRstd),
                            OUTPUT(yDesc, rstdDesc));
        uint64_t workspaceSize = 0;
        return ut.TestGetWorkspaceSize(&workspaceSize);
    }

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

TEST_F(l2_rms_norm_quant_v3_test, ascend950_scalar_x_returns_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto xDesc = TensorDesc(std::vector<int64_t>{}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto rstdDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRmsNormQuantV3, INPUT(xDesc, gammaDesc, scaleDesc, nullptr, nullptr, 1e-6, true, false),
                        OUTPUT(yDesc, rstdDesc));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_scalar_y_returns_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto xDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc(std::vector<int64_t>{}, ACL_INT8, ACL_FORMAT_ND);
    auto rstdDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnRmsNormQuantV3, INPUT(xDesc, gammaDesc, scaleDesc, nullptr, nullptr, 1e-6, true, false),
                        OUTPUT(yDesc, rstdDesc));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_invalid_tensor_shapes_return_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    struct ShapeCase {
        const char* name;
        size_t tensorIndex;
        vector<int64_t> shape;
    };
    const vector<ShapeCase> shapeCases = {
        {"x_rank_nine", 0, {1, 1, 1, 1, 1, 1, 1, 2, 16}},
        {"gamma_scalar", 1, {}},
        {"gamma_rank_three", 1, {1, 1, 16}},
        {"gamma_leading_dim_not_one", 1, {2, 16}},
        {"gamma_last_dim_mismatch", 1, {15}},
        {"scale_scalar", 2, {}},
        {"scale_rank_two", 2, {1, 1}},
        {"scale_length_mismatch", 2, {15}},
        {"offset_scalar", 3, {}},
        {"offset_rank_two", 3, {1, 1}},
        {"offset_length_mismatch", 3, {15}},
        {"beta_scalar", 4, {}},
        {"beta_rank_three", 4, {1, 1, 16}},
        {"beta_leading_dim_not_one", 4, {2, 16}},
        {"beta_last_dim_mismatch", 4, {15}},
        {"y_front_dim_mismatch", 5, {3, 16}},
        {"y_last_dim_mismatch", 5, {2, 15}},
        {"y_rank_mismatch", 5, {1, 2, 16}},
        {"y_rank_nine", 5, {1, 1, 1, 1, 1, 1, 1, 2, 16}},
        {"rstd_scalar", 6, {}},
        {"rstd_rank_mismatch", 6, {2}},
        {"rstd_front_dim_mismatch", 6, {3, 1}},
        {"rstd_last_dim_not_one", 6, {2, 2}},
    };
    for (const auto& shapeCase : shapeCases) {
        SCOPED_TRACE(shapeCase.name);
        TensorCase testCase;
        testCase.shapes[shapeCase.tensorIndex] = shapeCase.shape;
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_invalid_norm_shape_with_unit_last_dim_returns_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (size_t tensorIndex : {1U, 4U}) {
        SCOPED_TRACE("tensor index: " + to_string(tensorIndex));
        TensorCase testCase;
        testCase.shapes[0] = {2, 1};
        testCase.shapes[1] = {1};
        testCase.shapes[4] = {1};
        testCase.shapes[5] = {2, 1};
        testCase.shapes[tensorIndex] = {2, 1};
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_matching_scale_offset_with_invalid_length_returns_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorCase testCase;
    testCase.shapes[2] = {15};
    testCase.shapes[3] = {15};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_scalar_boundaries_preserve_rank_one_and_leading_one)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorCase testCase;
    testCase.shapes[2] = {1};
    testCase.shapes[3] = {};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);

    testCase.shapes[0] = {2, 1};
    testCase.shapes[1] = {1};
    testCase.shapes[3] = {1};
    testCase.shapes[4] = {};
    testCase.shapes[5] = {2, 1};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);

    for (const auto& normShape : vector<vector<int64_t>>{{1}, {1, 1}}) {
        SCOPED_TRACE(testing::PrintToString(normShape));
        testCase.shapes[1] = normShape;
        testCase.shapes[4] = normShape;
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_invalid_packed_output_shapes_return_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (const auto& outputShape : vector<vector<int64_t>>{{3, 2}, {2, 3}, {1, 2, 2}}) {
        SCOPED_TRACE("packed output shape: " + testing::PrintToString(outputShape));
        TensorCase testCase;
        testCase.dtypes[5] = ACL_INT32;
        testCase.shapes[5] = outputShape;
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
    }

    TensorCase testCase;
    testCase.shapes[0] = {2, 15};
    testCase.shapes[1] = {15};
    testCase.shapes[4] = {15};
    testCase.dtypes[5] = ACL_INT32;
    testCase.shapes[5] = {2, 1};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);

    testCase.dtypes[5] = ACL_INT4;
    testCase.shapes[5] = {2, 15};
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_invalid_tensor_dtypes_return_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    const array<aclDataType, 7> invalidDtypes = {ACL_INT32, ACL_FLOAT, ACL_INT32,  ACL_INT64,
                                                 ACL_BF16,  ACL_FLOAT, ACL_FLOAT16};
    for (size_t tensorIndex = 0; tensorIndex < invalidDtypes.size(); ++tensorIndex) {
        SCOPED_TRACE("tensor index: " + to_string(tensorIndex));
        TensorCase testCase;
        testCase.dtypes[tensorIndex] = invalidDtypes[tensorIndex];
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_invalid_quant_dtype_combinations_return_param_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    // Each entry is {x dtype, scale dtype, offset dtype}; gamma and beta follow x.
    const vector<array<aclDataType, 3>> invalidCombinations = {
        {ACL_FLOAT16, ACL_BF16, ACL_BF16},    {ACL_BF16, ACL_FLOAT16, ACL_FLOAT16}, {ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT},
        {ACL_FLOAT, ACL_FLOAT, ACL_INT32},    {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16},  {ACL_FLOAT16, ACL_FLOAT, ACL_INT8},
        {ACL_FLOAT16, ACL_FLOAT16, ACL_BF16}, {ACL_BF16, ACL_BF16, ACL_FLOAT},
    };
    for (const auto& dtypes : invalidCombinations) {
        SCOPED_TRACE("dtype combination: " + testing::PrintToString(dtypes));
        TensorCase testCase;
        testCase.dtypes[0] = dtypes[0];
        testCase.dtypes[1] = dtypes[0];
        testCase.dtypes[4] = dtypes[0];
        testCase.dtypes[2] = dtypes[1];
        testCase.dtypes[3] = dtypes[2];
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_leading_one_and_quant_scale_modes_remain_valid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (bool gammaLeadingOne : {false, true}) {
        for (bool betaLeadingOne : {false, true}) {
            for (int64_t scaleLength : {1, 16}) {
                SCOPED_TRACE("gamma leading one: " + to_string(gammaLeadingOne) + ", beta leading one: " +
                             to_string(betaLeadingOne) + ", scale length: " + to_string(scaleLength));
                TensorCase testCase;
                testCase.shapes[1] = gammaLeadingOne ? vector<int64_t>{1, 16} : vector<int64_t>{16};
                testCase.shapes[4] = betaLeadingOne ? vector<int64_t>{1, 16} : vector<int64_t>{16};
                testCase.shapes[2] = {scaleLength};
                testCase.shapes[3] = {scaleLength};
                EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_supported_quant_dtype_combinations_remain_valid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    const vector<array<aclDataType, 3>> supportedCombinations = {
        {ACL_FLOAT, ACL_FLOAT, ACL_FLOAT}, {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16},
        {ACL_BF16, ACL_BF16, ACL_BF16},    {ACL_FLOAT16, ACL_FLOAT16, ACL_INT8},
        {ACL_BF16, ACL_BF16, ACL_INT8},    {ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT},
        {ACL_BF16, ACL_FLOAT, ACL_FLOAT},  {ACL_FLOAT16, ACL_FLOAT, ACL_INT32},
        {ACL_BF16, ACL_FLOAT, ACL_INT32},
    };
    const vector<aclDataType> outputDtypes = {ACL_INT8,          ACL_INT4,        ACL_INT32,
                                              ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_HIFLOAT8};
    for (const auto& dtypes : supportedCombinations) {
        for (auto outputDtype : outputDtypes) {
            SCOPED_TRACE("dtype combination: " + testing::PrintToString(dtypes) +
                         ", output dtype: " + to_string(static_cast<int>(outputDtype)));
            TensorCase testCase;
            testCase.dtypes = {dtypes[0], dtypes[0], dtypes[1], dtypes[2], dtypes[0], outputDtype, ACL_FLOAT};
            if (outputDtype == ACL_INT32) {
                testCase.shapes[5] = {2, 2};
            }
            EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
        }
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_rank_one_and_rank_eight_remain_valid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (size_t rank : {1U, 8U}) {
        SCOPED_TRACE("rank: " + to_string(rank));
        TensorCase testCase;
        testCase.shapes[0] = vector<int64_t>(rank, 1);
        testCase.shapes[0].back() = 16;
        testCase.shapes[5] = testCase.shapes[0];
        testCase.shapes[6] = vector<int64_t>(rank, 1);
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_rank_one_allows_scalar_rstd)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (auto dtype : {ACL_FLOAT16, ACL_BF16, ACL_FLOAT}) {
        for (int64_t lastDim : {1, 16}) {
            for (bool leadingOne : {false, true}) {
                SCOPED_TRACE("dtype: " + to_string(dtype) + ", last dim: " + to_string(lastDim) +
                             ", leading one: " + to_string(leadingOne));
                TensorCase testCase;
                testCase.shapes[0] = {lastDim};
                testCase.shapes[1] = leadingOne ? vector<int64_t>{1, lastDim} : vector<int64_t>{lastDim};
                testCase.shapes[4] = testCase.shapes[1];
                testCase.shapes[5] = {lastDim};
                testCase.dtypes[0] = dtype;
                testCase.dtypes[1] = dtype;
                testCase.dtypes[4] = dtype;
                testCase.shapes[6] = {};
                EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
                testCase.shapes[6] = {1};
                EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_scalar_rstd_does_not_relax_other_checks)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    for (const auto& xShape : vector<vector<int64_t>>{{1, 16}, {2, 16}, {1, 1, 16}, {2, 3, 16}}) {
        SCOPED_TRACE("x shape: " + testing::PrintToString(xShape));
        TensorCase testCase;
        testCase.shapes[0] = xShape;
        testCase.shapes[5] = xShape;
        testCase.shapes[6] = {};
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
        testCase.shapes[6] = xShape;
        testCase.shapes[6].back() = 1;
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_SUCCESS);
    }

    TensorCase testCase;
    testCase.shapes[0] = {16};
    testCase.shapes[5] = {16};
    testCase.shapes[6] = {};
    testCase.dtypes[6] = ACL_FLOAT16;
    EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
    testCase.dtypes[6] = ACL_FLOAT;
    for (const auto& rstdShape : vector<vector<int64_t>>{{0}, {2}, {1, 1}}) {
        SCOPED_TRACE("rstd shape: " + testing::PrintToString(rstdShape));
        testCase.shapes[6] = rstdShape;
        EXPECT_EQ(RunTensorCase(testCase), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_disabled_rstd_allows_scalar_placeholder)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorCase testCase;
    testCase.shapes[6] = {};
    testCase.dtypes[6] = ACL_INT32;
    EXPECT_EQ(RunTensorCase(testCase, false), ACLNN_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_disabled_rstd_ignores_placeholder_shape_and_dtype)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorCase testCase;
    testCase.shapes[6] = {3, 2, 1};
    testCase.dtypes[6] = ACL_INT32;
    EXPECT_EQ(RunTensorCase(testCase, false), ACLNN_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_disabled_rstd_and_optional_inputs_allow_nullptr)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto xDesc = TensorDesc({2, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({1, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRmsNormQuantV3, INPUT(xDesc, gammaDesc, scaleDesc, nullptr, nullptr, 1e-6, true, false),
                        OUTPUT(yDesc, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_rms_norm_quant_v3_test, ascend950_requested_rstd_requires_non_nullptr)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto xDesc = TensorDesc({2, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRmsNormQuantV3, INPUT(xDesc, gammaDesc, scaleDesc, nullptr, nullptr, 1e-6, true, true),
                        OUTPUT(yDesc, nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}
