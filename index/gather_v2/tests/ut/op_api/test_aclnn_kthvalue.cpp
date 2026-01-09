/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_kthvalue.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_kthvalue_test : public testing::Test {
 protected:
  static void SetUpTestCase() {std::cout << "kthvalue_test SetUp" << std::endl;}

  static void TearDownTestCase() { std::cout << "kthvalue_test TearDown" << std::endl; }
};

// self为空指针
TEST_F(l2_kthvalue_test, l2_kthvalue_test_nullptr_self0) {
  auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(nullptr, 1, 0, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// values为空指针
TEST_F(l2_kthvalue_test, l2_kthvalue_test_nullptr_values0) {
  auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 1, 0, 1), OUTPUT(nullptr, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// indices为空指针
TEST_F(l2_kthvalue_test, l2_kthvalue_test_nullptr_indices0) {
  auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 1, 0, 1), OUTPUT(valDesc, nullptr));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// self为空tensor
TEST_F(l2_kthvalue_test, l2_kthvalue_test_nullptr_self) {
  auto selfDesc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({1, 1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 1, 0, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32
TEST_F(l2_kthvalue_test, l2_kthvalue_test_fp32) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 2, 1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 1, 2, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32，dim为-1
TEST_F(l2_kthvalue_test, l2_kthvalue_test_fp32_dim_neg1) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 2, 1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 2, -1, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32, dim为0
TEST_F(l2_kthvalue_test, l2_kthvalue_test_fp32_dim_neg0) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({1, 2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 1, 0, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,k为负数
TEST_F(l2_kthvalue_test, l2_kthvalue_test_fp32_k_neg1) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, -1, -1, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32,k大于self在dim上的size
TEST_F(l2_kthvalue_test, l2_kthvalue_test_fp32_k_10) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 10, 2, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float16
TEST_F(l2_kthvalue_test, l2_kthvalue_test_fp16) {
  auto selfDesc = TensorDesc({2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({1, 2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 2, 0, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int8
TEST_F(l2_kthvalue_test, l2_kthvalue_test_int8) {
  auto selfDesc = TensorDesc({2, 4, 5}, ACL_INT8, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 4, 1}, ACL_INT8, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 4, 1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 2, 2, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，uint8
TEST_F(l2_kthvalue_test, l2_kthvalue_test_uint8) {
  auto selfDesc = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 1}, ACL_UINT8, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 4, 1, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int16
TEST_F(l2_kthvalue_test, l2_kthvalue_test_int16) {
  auto selfDesc = TensorDesc({2, 4, 5}, ACL_INT16, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 4, 1}, ACL_INT16, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 4, 1}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 3, 2, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 用例不支持，complex64
TEST_F(l2_kthvalue_test, l2_kthvalue_test_complex16) {
  auto selfDesc = TensorDesc({2, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 2, 1, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，complex128
TEST_F(l2_kthvalue_test, l2_kthvalue_test_complex128) {
  auto selfDesc = TensorDesc({2, 4, 5}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({2, 2, 5}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 2, 1, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，shape=9
TEST_F(l2_kthvalue_test, l2_kthvalue_test_shape9) {
  auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT32, ACL_FORMAT_ND);
  auto valDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT32, ACL_FORMAT_ND);
  auto indDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnKthvalue, INPUT(selfDesc, 1, 1, 1), OUTPUT(valDesc, indDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
