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

#include "../../../op_host/op_api/aclnn_group_norm.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_group_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "group_norm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "group_norm_test TearDown" << endl;
    }
};

// TEST_F(l2_group_norm_test, case_float_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_group_norm_test, case_float16_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_group_norm_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> ValidList = {ACL_DOUBLE, ACL_UINT8, ACL_INT8,   ACL_INT16,     ACL_INT32,
                                     ACL_INT64,  ACL_BOOL,  ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    int length = ValidList.size();

    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({2, 3, 4}, ValidList[i], ACL_FORMAT_ND);
        auto gamma_desc = TensorDesc({3}, ValidList[i], ACL_FORMAT_ND);
        auto beta_desc = TensorDesc({3}, ValidList[i], ACL_FORMAT_ND);
        auto out_desc = TensorDesc({2, 3, 4}, ValidList[i], ACL_FORMAT_ND);
        auto mean_out_desc = TensorDesc({2, 1}, ValidList[i], ACL_FORMAT_ND);
        auto rstd_out_desc = TensorDesc({2, 1}, ValidList[i], ACL_FORMAT_ND);
        auto ut = OP_API_UT(
            aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
            OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_group_norm_test, case_gamma_not_1d_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_beta_not_1d_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_1d_abnormal)
{
    auto self_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 1;
    const int64_t HxW = 1;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_group_norm_test, case_2d_normal)
// {
//     auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 1;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_group_norm_test, case_4d_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_group_norm_test, case_5d_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_group_norm_test, case_6d_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_group_norm_test, case_7d_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_group_norm_test, case_8d_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_group_norm_test, case_9d_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_self_nullptr_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT((aclTensor*)nullptr, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// TEST_F(l2_group_norm_test, case_gamma_nullptr_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, (aclTensor*)nullptr, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_group_norm_test, case_beta_nullptr_normal)
// {
//     auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 4;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, (aclTensor*)nullptr, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_group_norm_test, case_out_nullptr_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT((aclTensor*)nullptr, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_test, case_mean_out_nullptr_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, (aclTensor*)nullptr, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_test, case_rstd_out_nullptr_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, (aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_test, case_diff_dtype_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_group_not_gt0_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 0;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_self_numel_ne_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 5;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_gamma_ne_C_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_beta_ne_C_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_self_shpe_ne_gn_out_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_mean_shpe_ne_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_rstd_shpe_ne_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_empty_normal)
{
    auto self_desc = TensorDesc({2, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 0;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_test, case_C_not_div_group_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 2;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_group_norm_test, ascend910B2_case_bfloat16_normal)
// {
//     auto self_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
//     auto gamma_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
//     auto beta_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
//     const int64_t N = 2;
//     const int64_t C = 3;
//     const int64_t HxW = 1;
//     const int64_t group = 1;
//     const double eps = 0.00001;
//     auto out_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
//     auto mean_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);
//     auto rstd_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(
//         aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
//         OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     // EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

TEST_F(l2_group_norm_test, case_self_ne_N_abnormal)
{
    auto self_desc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({3, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_self_ne_C_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 4;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_self_ne_HxW_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 2;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_self_format_ne_out_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_test, case_self_private_format_abnormal)
{
    auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t HxW = 4;
    const int64_t group = 1;
    const double eps = 0.00001;
    auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(self_desc, gamma_desc, beta_desc, N, C, HxW, group, eps),
        OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
