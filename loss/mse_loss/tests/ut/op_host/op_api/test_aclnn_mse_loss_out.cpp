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

#include "../../../../op_host/op_api/aclnn_mse_loss_out.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

#include <unistd.h>

using namespace op;
using namespace std;

class l2_mse_loss_out_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "mse_loss_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "mse_loss_test TearDown" << std::endl;
  }
};

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_01_float_nd_none) {
  auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_02_float16_nchw_mean) {
  auto selfDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_03_float_float16_nhwc_sum) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-1, 1);
  int64_t reduction = 2;

  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_04_float16_float_nhwc_mean) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_05_float_nd_empty_tensor_none) {
  auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_06_float_nd_empty_tensor_mean) {
  auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_07_float_nd_empty_tensor_sum) {
  auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 2;

  auto outDesc = TensorDesc({0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_11_float_hwcn_input_not_contiguous) {
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_12_float_ncdhw_out_not_contiguous) {
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_13_reduction_error) {
  auto selfDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 3;

  auto outDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_14_input_out_nullptr) {
  auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t reduction = 0;

  auto ut_l = OP_API_UT(aclnnMseLossOut, INPUT(nullptr, tensor_desc, reduction), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_r = OP_API_UT(aclnnMseLossOut, INPUT(tensor_desc, nullptr, reduction), OUTPUT(tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_r.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_o = OP_API_UT(aclnnMseLossOut, INPUT(tensor_desc, tensor_desc, reduction), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_15_aclnnMseLossOut_input_error_shape) {
  auto selfDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_16_aclnnMseLossOut_error_output_dtype) {
  auto selfDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_17_aclnnMseLossOut_error_input_dtype) {
  auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({6, 2, 1, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_18_aclnnMseLossOut_input_error_shape_len) {
  auto tensorDesc9 = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto tensorDesc1 = TensorDesc({7, 8, 9, 10}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto tensorDesc2 = TensorDesc({7, 8, 9, 10}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto ut_self = OP_API_UT(aclnnMseLossOut, INPUT(tensorDesc9, tensorDesc1, reduction), OUTPUT(tensorDesc2));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  auto ut_tar = OP_API_UT(aclnnMseLossOut, INPUT(tensorDesc1, tensorDesc9, reduction), OUTPUT(tensorDesc2));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_tar.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_19_float_hwcn_ndhwc_mean) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_20_aclnnMseLossOut_output_error_shape_none) {
  auto selfDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({123, 8, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_out_test, aclnnMseLossOut_21_aclnnMseLossOut_output_error_shape_mean) {
  auto selfDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto targetDesc = TensorDesc({8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_mse_loss_out_test, ascend910B2_aclnnMseLossOut_bf16_float_nhwc_mean) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMseLossOut, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
