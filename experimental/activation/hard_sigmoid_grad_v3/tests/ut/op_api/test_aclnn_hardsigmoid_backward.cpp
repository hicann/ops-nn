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

#include "../../../op_host/op_api/aclnn_hardsigmoid_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;

class l2_hardsigmoid_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase()
  {
    std::cout << "hardsigmoid_backward_test SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "hardsigmoid_backward_test TearDown" << std::endl;
  }
};

TEST_F(l2_hardsigmoid_backward_test, case_001_float)
{
  auto gradOutputDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, case_002_float16)
{
  auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, case_003_bfloat16)
{
  auto gradOutputDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto selfDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910E) {
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
  } else {
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_hardsigmoid_backward_test, case_004_mixed_dtype)
{
  auto gradOutputDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, case_005_empty_tensor)
{
  auto gradOutputDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, case_006_not_contiguous)
{
  auto gradOutputDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
  auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, case_007_scalar_input)
{
  auto gradOutputDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_backward_test, case_008_invalid_grad_dtype)
{
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, case_009_invalid_out_dtype)
{
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, case_010_shape_mismatch)
{
  auto gradOutputDesc = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, case_011_invalid_dim)
{
  auto gradOutputDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_backward_test, case_012_nullptr)
{
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardsigmoidBackward, INPUT(static_cast<aclTensor *>(nullptr), selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
