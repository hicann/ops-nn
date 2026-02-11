/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_log_sigmoid.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;

class l2_log_sigmoid_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "log_sigmoid_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "log_sigmoid_test TearDown" << endl;
  }
};

TEST_F(l2_log_sigmoid_test, case_nullptr_input) {
  auto tensor_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT((aclTensor*)nullptr), OUTPUT(tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_log_sigmoid_test, case_nullptr_output) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT((aclTensor*)nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_log_sigmoid_test, case_dtype_notsupport_double) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, case_dtype_notsupport_complex64) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, case_dtype_notsupport_int32) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, case_dtype_fp32) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_forward) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);
  auto buffer_tensor_desc = TensorDesc(self_tensor_desc);

  auto ut = OP_API_UT(aclnnLogSigmoidForward, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc, buffer_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_dtype_fp16) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_format_HWCN) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_format_NDHWC) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_format_NCDHW) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_shape_1d) {
  auto self_tensor_desc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_shape_3d) {
  auto self_tensor_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_shape_4d) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_shape_8d) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_shape_9d) {
  auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_empty_empty) {
  auto self_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_log_sigmoid_test, case_noempty_empty) {
  auto self_tensor_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, case_empty_noempty) {
  auto self_tensor_desc = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, case_different_shape) {
  auto self_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, case_different_dtype) {
  auto self_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, case_different_dim) {
  auto self_tensor_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_sigmoid_test, ascend910B2_case_bf16) {
  auto self_tensor_desc = TensorDesc({1, 2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto out_tensor_desc = TensorDesc({1, 2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);

  auto ut = OP_API_UT(aclnnLogSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}