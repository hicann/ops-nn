/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <vector>

#include "../../../op_host/op_api/aclnn_mish_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

class l2_mish_backward_v2_test : public testing::Test {
 protected:
  static void SetUpTestCase()
  {
    std::cout << "mish_backward_v2_test SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "mish_backward_v2_test TearDown" << std::endl;
  }
};

TEST_F(l2_mish_backward_v2_test, case_001_float)
{
  auto gradDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-4, 4);
  auto xGradDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_mish_backward_v2_test, case_002_float16)
{
  auto gradDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-4, 4);
  auto xGradDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_mish_backward_v2_test, case_003_bfloat16)
{
  auto gradDesc = TensorDesc({2, 2, 3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 2, 3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-4, 4);
  auto xGradDesc = TensorDesc({2, 2, 3, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
  } else {
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_mish_backward_v2_test, case_004_invalid_type_double)
{
  auto gradDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xGradDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mish_backward_v2_test, case_005_invalid_type_int32)
{
  auto gradDesc = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xGradDesc = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mish_backward_v2_test, case_006_mixed_dtype_not_supported)
{
  auto gradDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xGradDesc = TensorDesc({2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mish_backward_v2_test, case_007_empty_tensor)
{
  auto gradDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto xDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto xGradDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_mish_backward_v2_test, case_008_not_contiguous)
{
  auto gradDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto xDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-4, 4);
  auto xGradDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_mish_backward_v2_test, case_009_nullptr)
{
  auto tensorDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspaceSize = 0;

  auto ut1 = OP_API_UT(aclnnMishBackward, INPUT(nullptr, tensorDesc), OUTPUT(tensorDesc));
  aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnMishBackward, INPUT(tensorDesc, nullptr), OUTPUT(tensorDesc));
  aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut3 = OP_API_UT(aclnnMishBackward, INPUT(tensorDesc, tensorDesc), OUTPUT(nullptr));
  aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mish_backward_v2_test, case_010_max_dim)
{
  auto gradDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto xDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto xGradDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mish_backward_v2_test, case_011_shape_mismatch)
{
  auto gradDesc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto xDesc = TensorDesc({1, 3, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto xGradDesc = TensorDesc({1, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mish_backward_v2_test, case_012_output_shape_mismatch)
{
  auto gradDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xGradDesc = TensorDesc({2, 2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mish_backward_v2_test, case_013_output_dtype_mismatch)
{
  auto gradDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto xGradDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMishBackward, INPUT(gradDesc, xDesc), OUTPUT(xGradDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
