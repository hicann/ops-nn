/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_inverse.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


using namespace std;

class l2_inverse_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_inverse_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "l2_inverse_test TearDown" << endl; }
};

TEST_F(l2_inverse_test, normal_float) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inverse_test, normal_double) {
  auto selfDesc = TensorDesc({1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCHW).Value(vector<double>{1, 2, 3, 4});
  auto outDesc = TensorDesc({1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inverse_test, normal_complex64) {
  auto selfDesc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_NHWC).Value(vector<float>{1, 2, 3, 4});
  auto outDesc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_NHWC);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inverse_test, normal_complex128) {
  auto selfDesc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX128, ACL_FORMAT_HWCN).Value(vector<float>{1, 2, 3, 4});
  auto outDesc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX128, ACL_FORMAT_HWCN);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inverse_test, normal_float16) {
  auto selfDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC).Value(vector<float>{1, 2, 3, 4});
  auto outDesc = TensorDesc({1, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景：空tensor
TEST_F(l2_inverse_test, normal_empty_tensor) {
    auto selfDesc = TensorDesc({0, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_inverse_test, abnormal_self_int) {
  auto selfDesc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inverse_test, normal_self_can_cast_out) {
  auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto outDesc = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_inverse_test, abnormal_self_can_not_cast_out) {
  auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self与out的shape不一致
TEST_F(l2_inverse_test, abnormal_diff_shape_self_out) {
  auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self为空
TEST_F(l2_inverse_test, abnormal_self_null) {
  auto selfDesc = nullptr;
  auto outDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：out为空
TEST_F(l2_inverse_test, abnormal_out_null) {
  auto selfDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = nullptr;

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_inverse_test, abnormal_not_square) {
  auto selfDesc = TensorDesc({3, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
  auto outDesc = TensorDesc({3, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：维度大于8
TEST_F(l2_inverse_test, abnormal_over_max_dim) {
  auto selfDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({3, 2, 1, 2, 1, 3, 2, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：维度小于2
TEST_F(l2_inverse_test, abnormal_less_min_dim) {
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inverse_test, normal_valuerange) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1).Value(vector<float>{-1, 0, 0, 1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_inverse_test, normal_non_contiguous) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2}).Value(vector<float>{1, 2, 3, 4});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {2, 2});

  auto ut = OP_API_UT(aclnnInverse, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}