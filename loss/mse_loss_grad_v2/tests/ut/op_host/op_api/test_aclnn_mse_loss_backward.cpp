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

#include "../../../../op_host/op_api/aclnn_mse_loss_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace op;
using namespace std;

class l2_mse_loss_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "mse_loss_backward_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "mse_loss_backward_test TearDown" << std::endl;
  }
};

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_01_float_nd_none) {
  auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_02_float16_nchw_mean) {
  auto gradOutputDesc = TensorDesc({1, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto selfDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_03_float_float16_nhwc_sum) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  int64_t reduction = 2;

  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_04_float16_float_nhwc_mean) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_05_float_nd_empty_tensor_none) {
  auto gradOutputDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({1, 0, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_06_float_nd_empty_tensor_mean) {
  auto gradOutputDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({1, 0, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_07_float_nd_empty_tensor_sum) {
  auto gradOutputDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 2;

  auto outDesc = TensorDesc({1, 0, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_11_float_hwcn_input_not_contiguous) {
  auto gradOutputDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  //EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_12_float_ncdhw_out_not_contiguous) {
  auto gradOutputDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  //EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_13_reduction_error) {
  auto gradOutputDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 3;

  auto outDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_14_input_out_nullptr) {
  auto tensorDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 0;

  auto ut_l = OP_API_UT(aclnnMseLossBackward, INPUT(nullptr, tensorDesc, tensorDesc, reduction), OUTPUT(tensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_r = OP_API_UT(aclnnMseLossBackward, INPUT(tensorDesc, nullptr, tensorDesc, reduction), OUTPUT(tensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_r.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_ta = OP_API_UT(aclnnMseLossBackward, INPUT(tensorDesc, tensorDesc, nullptr, reduction), OUTPUT(tensorDesc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_ta.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_o = OP_API_UT(aclnnMseLossBackward, INPUT(tensorDesc, tensorDesc, tensorDesc, reduction), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_15_aclnnMseLossBackward_input_error_shape) {
  auto gradOutputDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({123, 8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_16_aclnnMseLossBackward_error_output_dtype) {
  auto gradOutputDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_17_aclnnMseLossBackward_error_input_dtype) {
  auto gradOutputDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_18_aclnnMseLossBackward_diff_input_dtype) {
  auto gradOutputDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  auto ut_2 = OP_API_UT(aclnnMseLossBackward, INPUT(targetDesc, selfDesc, gradOutputDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_19_aclnnMseLossBackward_input_error_shape_len) {
  auto tensorDesc9 = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto tensorDesc1 = TensorDesc({7, 8, 9, 10}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto tensorDesc2 = TensorDesc({7, 8, 9, 10}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto ut_grad = OP_API_UT(aclnnMseLossBackward, INPUT(tensorDesc9, tensorDesc1, tensorDesc1, reduction),
                           OUTPUT(tensorDesc2));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_grad.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  auto ut_self = OP_API_UT(aclnnMseLossBackward, INPUT(tensorDesc1, tensorDesc9, tensorDesc1, reduction),
                           OUTPUT(tensorDesc2));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  auto ut_tar = OP_API_UT(aclnnMseLossBackward, INPUT(tensorDesc1, tensorDesc1, tensorDesc9, reduction),
                           OUTPUT(tensorDesc2));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_tar.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_20_float_hwcn_ndhwc_mean) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_21_aclnnMseLossBackward_output_error_shape_none) {
  auto gradOutputDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({123, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({123, 11, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({123, 8, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, aclnnMseLossBackward_22_aclnnMseLossBackward_input_error_shape) {
  auto gradOutputDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({123, 8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_backward_test, ascend910B2_aclnnMseLossBackward_02_bfloat16_nchw_mean) {
  auto gradOutputDesc = TensorDesc({1, 3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto selfDesc = TensorDesc({2, 3, 2}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({2, 3, 2}, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMseLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}
