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

#include "../../../../op_host/op_api/aclnn_logsoftmax.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_logsoftmax_test : public testing::Test {
  protected:
  static void SetUpTestCase() { cout << "logsoftmax_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "logsoftmax_test TearDown" << endl; }
};

TEST_F(l2_logsoftmax_test, case_000_workspace) {
  auto tensorDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  int64_t dim = 0;

  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(tensorDesc, dim), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_001_float32_normal) {
  auto selfTensor = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto outTensor = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_logsoftmax_test, case_002_float16_normal) {
  auto selfTensor = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto outTensor = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_003_double_normal) {
  auto selfTensor = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto outTensor = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_logsoftmax_test, case_008_1_dim_input_tensor) {
  auto selfTensor = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();

  selfTensor = TensorDesc({8}, ACL_DOUBLE, ACL_FORMAT_ND);
  outTensor = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_009_3_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();

  selfTensor = TensorDesc({1, 2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
  outTensor = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_010_5_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();

  selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_011_8_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();

  selfTensor = TensorDesc({1, 2, 3, 4, 5, 7, 8}, ACL_DOUBLE, ACL_FORMAT_ND);
  outTensor = TensorDesc({1, 2, 3, 4, 5, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_012_dim_coverage1) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 4;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_013_dim_coverage2) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = -5;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_014_NHWC) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  selfTensor = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NHWC);
  outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_015_NCHW) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  selfTensor = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NCHW);
  outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_016_HWCN) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  selfTensor = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_HWCN);
  outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_017_NDHWC) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);
  auto outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  selfTensor = TensorDesc({2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NDHWC);
  outTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_018_NCDHW) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_NCDHW);
  outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);
  auto utDouble = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  workspaceSize = 0;
  aclRet = utDouble.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_019_invalid_dtype) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_UINT32, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_UINT32, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logsoftmax_test, case_020_invalid_shape) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logsoftmax_test, case_021_invalid_dim) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 9;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logsoftmax_test, case_023_empty_tensor) {
  auto selfTensor = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_024_uncontiguous) {
  auto selfTensor = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
  auto outTensor = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5});
  int64_t dim = 1;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_logsoftmax_test, case_025_nullptr) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto ut1 = OP_API_UT(aclnnLogSoftmax, INPUT((aclTensor*)nullptr, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(nullptr));

  aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_logsoftmax_test, case_026_9_dim_input_tensor) {
  auto selfTensor = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_logsoftmax_test, case_027_0_dim_input_tensor) {
  auto selfTensor = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_logsoftmax_test, ascend910B2_case_BF16_910b) {
  auto selfTensor = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_logsoftmax_test, case_BF16_910) {
  auto selfTensor = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnLogSoftmax, INPUT(selfTensor, dim), OUTPUT(outTensor));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}