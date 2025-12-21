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

#include "../../../op_host/op_api/aclnn_group_norm_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_group_norm_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "group_norm_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "group_norm_backward_test TearDown" << endl;
    }
};

TEST_F(l2_group_norm_backward_test, case_float_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, Ascend910B2_case_float_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const int64_t N = 2;
        const int64_t C = 3;
        const int64_t HxW = 4;
        const int64_t group = 1;
        auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
        auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
        auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
        auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
        auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
        auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
        auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
        auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
        auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

        auto ut = OP_API_UT(
            aclnnGroupNormBackward,
            INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
            OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_group_norm_backward_test, case_float16_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, ascend910B2_case_bfloat16_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_BF16, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_BF16, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_BF16, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_BF16, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_BF16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_BF16, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_BF16, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> ValidList = {ACL_DOUBLE, ACL_UINT8, ACL_INT8,   ACL_INT16,     ACL_INT32,
                                     ACL_INT64,  ACL_BOOL,  ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    int length = ValidList.size();

    for (int i = 0; i < length; i++) {
        auto grad_out_desc = TensorDesc({N, C, HxW}, ValidList[i], ACL_FORMAT_ND);
        auto input_desc = TensorDesc({N, C, HxW}, ValidList[i], ACL_FORMAT_ND);
        auto gamma_desc = TensorDesc({C}, ValidList[i], ACL_FORMAT_ND);
        auto mean_desc = TensorDesc({N, group}, ValidList[i], ACL_FORMAT_ND);
        auto rstd_desc = TensorDesc({N, group}, ValidList[i], ACL_FORMAT_ND);
        auto grad_input_desc = TensorDesc({N, C, HxW}, ValidList[i], ACL_FORMAT_ND);
        auto dgamma_desc = TensorDesc({C}, ValidList[i], ACL_FORMAT_ND);
        auto dbeta_desc = TensorDesc({C}, ValidList[i], ACL_FORMAT_ND);

        auto ut = OP_API_UT(
            aclnnGroupNormBackward,
            INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
            OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_group_norm_backward_test, case_gradout_not_1d_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N * C * HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_1d_normal)
{
    const int64_t N = 2;
    const int64_t C = 1;
    const int64_t HxW = 1;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_2d_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 1;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_8d_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_9d_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_backward_test, case_gradout_nullptr_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT((aclTensor*)nullptr, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_backward_test, case_input_nullptr_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, (aclTensor*)nullptr, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_backward_test, case_mean_nullptr_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, (aclTensor*)nullptr, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_backward_test, case_shape_not_same_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({1, C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({1, C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({1, C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_rstd_nullptr_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, (aclTensor*)nullptr, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_backward_test, case_gamma_nullptr_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, (aclTensor*)nullptr, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_gradinput_nullptr_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT((aclTensor*)nullptr, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_dgamma_nullptr_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, (aclTensor*)nullptr, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_dbeta_nullptr_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, (aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, case_diff_dtype_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_backward_test, case_group_not_gt0_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 0;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_backward_test, case_gradout_numel_ne_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_backward_test, case_gradout_shape_ne_gradinput_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_backward_test, case_C_not_div_group_abnormal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 2;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_backward_test, Ascend910_9599_case_float_normal)
{
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_backward_test, Ascend910_9599_case_float_normal_2)
{
    const int64_t N = 2;
    const int64_t C = 4;
    const int64_t HxW = 128;
    const int64_t group = 2;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, Ascend910_9599_case_float16_normal_3)
{
    const int64_t N = 2;
    const int64_t C = 4;
    const int64_t HxW = 128;
    const int64_t group = 2;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_backward_test, Ascend910_9599_case_bfloat16_normal_4)
{
    const int64_t N = 2;
    const int64_t C = 4;
    const int64_t HxW = 128;
    const int64_t group = 2;
    auto grad_out_desc = TensorDesc({N, C, HxW}, ACL_BF16, ACL_FORMAT_ND);
    auto input_desc = TensorDesc({N, C, HxW}, ACL_BF16, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({C}, ACL_BF16, ACL_FORMAT_ND);
    auto mean_desc = TensorDesc({N, group}, ACL_BF16, ACL_FORMAT_ND);
    auto rstd_desc = TensorDesc({N, group}, ACL_BF16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto grad_input_desc = TensorDesc({N, C, HxW}, ACL_BF16, ACL_FORMAT_ND);
    auto dgamma_desc = TensorDesc({C}, ACL_BF16, ACL_FORMAT_ND);
    auto dbeta_desc = TensorDesc({C}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNormBackward,
        INPUT(grad_out_desc, input_desc, mean_desc, rstd_desc, gamma_desc, N, C, HxW, group, output_mask),
        OUTPUT(grad_input_desc, dgamma_desc, dbeta_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
