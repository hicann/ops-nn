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
#include "norm/group_norm_silu/op_host/op_api/aclnn_group_norm_silu.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_group_norm_silu_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "group_norm_silu_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "group_norm_silu_test TearDown" << endl; }
};

TEST_F(l2_group_norm_silu_test, ascend910B2_case_float16_normal) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSiluV2,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps, activateSilu),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_bfloat16_normal) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_invalid_dtype_abnormal) {
  vector<aclDataType> ValidList = {ACL_DOUBLE, ACL_UINT8, ACL_INT8, ACL_INT16, ACL_INT32, ACL_INT64,
                                   ACL_BOOL, ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  int length = ValidList.size();

  for (int i = 0; i < 1; i++) {
    auto self_desc = TensorDesc({2, 3, 4}, ValidList[i], ACL_FORMAT_ND);
    auto gamma_desc = TensorDesc({3}, ValidList[i], ACL_FORMAT_ND);
    auto beta_desc = TensorDesc({3}, ValidList[i], ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 3, 4}, ValidList[i], ACL_FORMAT_ND);
    auto mean_out_desc = TensorDesc({2, 1}, ValidList[i], ACL_FORMAT_ND);
    auto rstd_out_desc = TensorDesc({2, 1}, ValidList[i], ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_gamma_not_1d_abnormal) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_beta_not_1d_abnormal) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_1d_abnormal) {
  auto self_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_2d_normal) {
  auto self_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_4d_normal) {
  auto self_desc = TensorDesc({2, 3, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_5d_normal) {
  auto self_desc = TensorDesc({2, 3, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_4d_normal_no_gamma) {
  auto self_desc = TensorDesc({2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  const int64_t group = 2;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}


TEST_F(l2_group_norm_silu_test, ascend910B2_case_self_nullptr_abnormal) {
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT((aclTensor*)nullptr, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_gamma_nullptr_normal) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, (aclTensor*)nullptr, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_beta_nullptr_normal) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, (aclTensor*)nullptr, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_empty_tensor_normal) {
  auto self_desc = TensorDesc({2,3,0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_out_nullptr_abnormal) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT((aclTensor*)nullptr, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_dtype_normal0) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_dtype_normal1) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_dtype_abnormal0) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_dtype_abnormal1) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_dtype_abnormal2) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_dtype_abnormal3) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSilu,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_group_norm_silu_test, ascend910B2_case_float16_empty) {
  auto self_desc = TensorDesc({2, 3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSiluV2,
                      INPUT(self_desc, gamma_desc, beta_desc, group, eps, activateSilu),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}