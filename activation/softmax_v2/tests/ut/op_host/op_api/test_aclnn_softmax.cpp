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

#include "../../../../op_host/op_api/aclnn_softmax.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_softmax_test : public testing::Test {
  protected:
  static void SetUpTestCase() { cout << "softmax_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "softmax_test TearDown" << endl; }
};

TEST_F(l2_softmax_test, case_001_bf_abnormal) {
  auto self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_test, case_002_float16_normal) {
  auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_softmax_test, case_003_float32_normal) {
  auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_004_double_normal) {
  auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 1;// tf算子只支持lastdim

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_005_8_dim_input) {
  auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_006_max_dim) {
  auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 7;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_007_dim_invalid) {
  auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 8;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_test, case_008_format_NHWC_normal) {
  auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_010_emptytensor_normal) {
  auto self = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_softmax_test, case_011_input_shape_diff_with_out_normal) {
  auto self = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
//   ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_012_input_dtype_diff_with_out_normal) {
  auto self = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
//   ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_013_input_contigous_normal) {
  auto self = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);
  auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_softmax_test, case_014_9_dim_input) {
  auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_test, case_015_nullptr) {
  auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(nullptr, dim), OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(nullptr));
  workspace_size = 0;
  aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_softmax_test, case_016_bf16) {
  auto self = TensorDesc({3, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B) {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      ut.TestPrecision();
  } else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_softmax_test, case_017_platforminfo_ascend910B) {
  auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_softmax_test, case_zero_dim_num) {
  auto self = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmax, INPUT(self, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}
