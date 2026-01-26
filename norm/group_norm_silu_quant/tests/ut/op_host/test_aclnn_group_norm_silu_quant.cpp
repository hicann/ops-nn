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
#include "../../../op_host/op_api/aclnn_group_norm_silu_quant.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_group_norm_silu_quant_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "group_norm_silu_quant_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "group_norm_silu_quant_test TearDown" << endl; }
};

TEST_F(l2_group_norm_silu_quant_test, ascend910B2_case_float16_perchannel) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto quantScale_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSiluQuant,
                      INPUT(self_desc, gamma_desc, beta_desc, quantScale_desc, group, eps, activateSilu),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_quant_test, ascend910B2_case_bfloat16_perchannel) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScale_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;

  auto out_desc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSiluQuant,
                      INPUT(self_desc, gamma_desc, beta_desc, quantScale_desc, group, eps, activateSilu),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_quant_test, ascend910B2_case_float16_pertensor) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto quantScale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;
  auto out_desc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSiluQuant,
                      INPUT(self_desc, gamma_desc, beta_desc, quantScale_desc, group, eps, activateSilu),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_group_norm_silu_quant_test, ascend910B2_case_bfloat16_pertensor) {
  auto self_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto gamma_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  auto beta_desc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  const int64_t group = 1;
  const double eps = 0.00001;
  const bool activateSilu = true;

  auto out_desc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto mean_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);
  auto rstd_out_desc = TensorDesc({2, 1}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGroupNormSiluQuant,
                      INPUT(self_desc, gamma_desc, beta_desc, quantScale_desc, group, eps, activateSilu),
                      OUTPUT(out_desc, mean_out_desc, rstd_out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}