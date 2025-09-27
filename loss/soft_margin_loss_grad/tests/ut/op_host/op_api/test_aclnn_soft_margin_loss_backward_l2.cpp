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

#include "../../../../op_host/op_api/aclnn_soft_margin_loss_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace op;
using namespace std;

class l2_soft_margin_loss_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "soft_margin_loss_backward_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "soft_margin_loss_backward_test TearDown" << std::endl;
  }
};

// grad_output nullptr
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_grad_output_nullptr) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(nullptr, selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// self nullptr
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_self_nullptr) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, nullptr, targetDesc, reduction),
                      OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// target nullptr
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_target_nullptr) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, nullptr, reduction),
                      OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_out_nullptr) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                      OUTPUT(nullptr));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// empty tensor
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_empty_tensor) {
  auto gradOutputDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// dtype float32
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_float32) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// dtype double
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_double) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// reduction mean
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_reduction_mean) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 1;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// reduction mean
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_reduction_sum) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 2;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// shape 9d
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_shape_9d) {
  auto gradOutputDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// target shape broadcast
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_diff_target_shape) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 2;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// dtype self float16 target float32
TEST_F(l2_soft_margin_loss_backward_test, soft_margin_loss_backward_float16_float32) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype bfloat16
TEST_F(l2_soft_margin_loss_backward_test, ascend910B2_soft_margin_loss_backward_bfloat16) {
  auto gradOutputDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLossBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction),
                                                         OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
