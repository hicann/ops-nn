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

#include "../../../op_host/op_api/aclnn_index_add_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_index_add_v2_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "aclnnIndexAddV2_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "aclnnIndexAddV2_test TearDown" << std::endl; }
};

TEST_F(l2_index_add_v2_test, case_NULLPTR_0) {
  int64_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT((aclTensor*)nullptr, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT((aclTensor*)nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_v2_test, case_NULLPTR_1) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int64_t dims = 0;
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, (aclTensor*)nullptr, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_v2_test, case_NULLPTR_2) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int64_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, (aclTensor*)nullptr, scalar_desc, mode), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_FLOAT) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  // EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_DOUBLE) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(static_cast<double>(1.0f));
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  // EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_FLOAT_self_dim3) {
  auto tensor_desc = TensorDesc({5, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(static_cast<float>(1.0f));
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  // EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_type_self) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_COMPLEX64, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_type_index) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_type_source) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_COMPLEX64, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_type_source_self) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_shape_dim) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 10;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_shape_self_source) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_shape_index) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({5, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(0, 10);
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_error_shape_index_source) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_FLOAT_mode_0) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 0;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_FLOAT_mode_0_scalar_2) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(2.0f);
  int64_t mode = 0;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_FLOAT_mode_0_int32) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 0;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1);
  int64_t mode = 0;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_index_add_v2_test, Ascend910B_case_FLOAT_mode_0_int32_dim_1) {
  auto tensor_desc = TensorDesc({5, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  int32_t dims = 1;
  auto index_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_NCHW).Value(vector<int32_t>{0, 2, 4});
  auto source_desc = TensorDesc({3, 3}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-10, 10);
  auto scalar_desc = ScalarDesc(1.0f);
  int64_t mode = 0;

  auto ut = OP_API_UT(aclnnIndexAddV2, INPUT(tensor_desc, dims, index_desc, source_desc, scalar_desc, mode), OUTPUT(tensor_desc));


  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}