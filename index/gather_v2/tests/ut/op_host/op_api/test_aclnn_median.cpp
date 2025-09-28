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

#include "level2/aclnn_median.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_median_test : public testing::Test {
  protected:
  static void SetUpTestCase() { cout << "median_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "median_test TearDown" << endl; }
};

TEST_F(l2_median_test, case_000_workspace) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_001_float32_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_002_float16_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_median_test, case_003_int32_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_004_int8_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_005_uint8_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_006_int16_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_007_int64_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_008_complex64_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_test, case_009_complex128_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_test, ascend910B2_case_010_bfloat16_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B) {
      EXPECT_EQ(aclRet, ACL_SUCCESS);
      // ut.TestPrecision();
  } else {
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(l2_median_test, case_011_bool_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_test, case_012_double_normal) {
  auto selfTensor = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_test, case_013_1_dim_input_tensor) {
  auto selfTensor = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_014_3_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_015_4_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_016_8_dim_input_tensor) {
  auto selfTensor = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_017_NHWC) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NHWC);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_018_NCHW) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_019_HWCN) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_HWCN);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_020_NDHWC) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NDHWC);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_021_NCDHW) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_median_test, case_022_empty_tensor) {
  auto selfTensor = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_median_test, case_023_uncontiguous) {
  auto selfTensor = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_median_test, case_024_nullptr) {
  auto selfTensor = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut1 = OP_API_UT(aclnnMedian, INPUT((aclTensor *)nullptr), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT((aclTensor *)nullptr));
  workspaceSize = 0;
  aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_median_test, case_025_9_dim_input_tensor) {
  auto selfTensor = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outTensor = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}