/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "../../../op_host/op_api/aclnn_nonzero.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace op;
using namespace std;

class l2_nonzero_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "nonzero_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "nonzero_test TearDown" << std::endl;
  }
};

TEST_F(l2_nonzero_test, aclnnNonzero_01_int8_nd) {
  auto selfDesc = TensorDesc({5}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{0, 1, 2, 3, 4});
  auto outDesc = TensorDesc({4, 1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, aclnnNonzero_02_float32_nhwc) {
  auto selfDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto outDesc = TensorDesc({5, 1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, aclnnNonzero_03_float16_nchw) {
  auto selfDesc = TensorDesc({2, 3, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto outDesc = TensorDesc({36, 4}, ACL_INT64, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, aclnnNonzero_08_uint8_hwcn) {
  auto selfDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_HWCN);
  auto outDesc = TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, aclnnNonzero_09_bool_hwcn) {
  auto selfDesc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_HWCN);
  auto outDesc = TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, aclnnNonzero_10_empty_tensor) {
  auto selfDesc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_NCDHW);
  auto outDesc = TensorDesc({1, 0}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, aclnnNonzero_11_input_not_contiguous) {
  auto selfDesc = TensorDesc({5, 4}, ACL_INT8, ACL_FORMAT_HWCN, {1, 5}, 0, {4, 5});
  auto outDesc = TensorDesc({20, 2}, ACL_INT64, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, aclnnNonzero_12_input_out_nullptr) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_INT64, ACL_FORMAT_ND);

  auto ut_l = OP_API_UT(aclnnNonzero, INPUT(nullptr), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_o = OP_API_UT(aclnnNonzero, INPUT(tensor_desc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nonzero_test, aclnnNonzero_13_error_input_dtype) {
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nonzero_test, aclnnNonzero_14_error_output_dtype) {
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nonzero_test, aclnnNonzero_15_input_error_shape_len) {
  auto tensorDesc9 = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut_self = OP_API_UT(aclnnNonzero, INPUT(tensorDesc9), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nonzero_test, aclnnNonzero_16_input_error_shape_dim) {
  auto tensorDesc9 = TensorDesc({2147484638,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto outDesc = TensorDesc({2147483638,}, ACL_INT64, ACL_FORMAT_ND);

  auto ut_self = OP_API_UT(aclnnNonzero, INPUT(tensorDesc9), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nonzero_test, ascend910B2_aclnnNonzero_16_bf16) {
  auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_NCHW);
  auto outDesc = TensorDesc({6, 2}, ACL_INT64, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_test, Ascend910_95_aclnnNonzero_11_input_not_contiguous) {
  auto selfDesc = TensorDesc({5, 4}, ACL_INT8, ACL_FORMAT_HWCN, {1, 5}, 0, {4, 5});
  auto outDesc = TensorDesc({20, 2}, ACL_INT64, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnNonzero, INPUT(selfDesc), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
