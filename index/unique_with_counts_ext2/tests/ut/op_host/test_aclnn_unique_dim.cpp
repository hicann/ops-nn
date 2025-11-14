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

#include "level2/aclnn_unique_dim.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_unique_dim_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "unique_dim_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "unique_dim_test TearDown" << endl;
  }
};

// 测试空Tensor
TEST_F(l2_unique_dim_test, case_016_EMPTY) {
  auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto indices = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
  auto counts = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);

  bool sorted = true;
  bool return_inverse = true;
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                      OUTPUT(output, indices, counts));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试空指针
TEST_F(l2_unique_dim_test, case_017_NULLPTR) {
  vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
  auto self = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
  auto output = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto indices = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
  auto counts = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  bool sorted = true;
  bool return_inverse = true;
  int64_t dim = 0;
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet;

  auto ut1 = OP_API_UT(aclnnUniqueDim, INPUT(nullptr, sorted, return_inverse, dim),
                       OUTPUT(output, indices, counts));
  aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                       OUTPUT(nullptr, indices, counts));
  aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut3 = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                       OUTPUT(output, nullptr, counts));
  aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut4 = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                       OUTPUT(output, indices, nullptr));
  aclRet = ut4.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试inverseOut数据类型
TEST_F(l2_unique_dim_test, case_018_DTYPE_INVERSE_OUT) {
  vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
  auto self = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
  auto output = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto indices = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);  // invalid type of inverseOut
  auto counts = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  bool sorted = true;
  bool return_inverse = true;
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                      OUTPUT(output, indices, counts));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试countsOut数据类型
TEST_F(l2_unique_dim_test, case_019_DTYPE_COUNTS_OUT) {
  vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
  auto self = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
  auto output = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto indices = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
  auto counts = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);  // invalid type of countsOut

  bool sorted = true;
  bool return_inverse = true;
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                      OUTPUT(output, indices, counts));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试self和valueOut数据类型不一致
TEST_F(l2_unique_dim_test, case_020_DTYPE_UNCONSISTENT_BETWEEN_SELF_AND_VALUE_OUT) {
  vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
  auto self = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
  auto output = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto indices = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
  auto counts = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  bool sorted = true;
  bool return_inverse = true;
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                      OUTPUT(output, indices, counts));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试inverseOut和countsOut数据类型不一致
TEST_F(l2_unique_dim_test, case_021_DTYPE_UNCONSISTENT_BETWEEN_INVERSE_OUT_AND_COUNTS_OUT) {
  vector<float> raw_value = {0, 1, 2, 2, 3, 1, 1, 1, 4, 4};
  auto self = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(raw_value);
  auto output = TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto indices = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
  auto counts = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  bool sorted = true;
  bool return_inverse = true;
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                      OUTPUT(output, indices, counts));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试shape维度大于8
TEST_F(l2_unique_dim_test, case_022_OVERSIZE_SHAPE) {
  auto self = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10);
  auto output = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto indices = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_INT64, ACL_FORMAT_ND);
  auto counts = TensorDesc({1, 2, 2, 1, 2, 2, 3, 2, 1, 1}, ACL_INT64, ACL_FORMAT_ND);

  bool sorted = true;
  bool return_inverse = true;
  int64_t dim = 0;
  auto ut = OP_API_UT(aclnnUniqueDim, INPUT(self, sorted, return_inverse, dim),
                      OUTPUT(output, indices, counts));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
