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
#include "../../../../op_host/op_api/aclnn_softmax_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_softmax_backward_test : public testing::Test {
  protected:
  static void SetUpTestCase() { cout << "softmax_backward_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "softmax_backward_test TearDown" << endl; }
};

TEST_F(l2_softmax_backward_test, case_001_float32_normal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_002_float16_normal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_softmax_backward_test, case_003_double_abnormal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_backward_test, case_004_bfloat_abnormal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_backward_test, case_005_diff_input_dtype_abnormal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_backward_test, case_006_diff_input_format_normal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_007_diff_input_shape_abnormal) {
  auto gradOutput = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_008_diff_in_out_dtype_normal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_softmax_backward_test, case_009_diff_in_out_format_normal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_010_diff_in_out_shape_normal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_backward_test, case_011_8_dim_input) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_012_max_dim) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 7;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_013_dim_invalid) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 8;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_backward_test, case_014_format_NHWC_normal) {
  auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_016_input_emptytensor_normal) {
  auto gradOutput = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_softmax_backward_test, case_017_input_contigous_normal) {
  auto gradOutput = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);
  auto output = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 10);
  auto out = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_softmax_backward_test, case_018_9_dim_input) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_softmax_backward_test, case_019_nullptr) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(nullptr, output, dim), OUTPUT(out));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);


  auto ut2 = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, nullptr, dim), OUTPUT(out));
  workspaceSize = 0;
  aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut3 = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(nullptr));
  workspaceSize = 0;
  aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_softmax_backward_test, case_020_bf16) {
  auto gradOutput = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto out = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B) {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      ut.TestPrecision();
  } else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_softmax_backward_test, case_zero_dim_num) {
  auto gradOutput = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto output = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto out = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnSoftmaxBackward, INPUT(gradOutput, output, dim), OUTPUT(out));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
