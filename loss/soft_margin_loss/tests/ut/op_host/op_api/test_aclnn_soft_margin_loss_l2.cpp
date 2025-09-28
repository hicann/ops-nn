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

#include "../../../../op_host/op_api/aclnn_soft_margin_loss.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace op;
using namespace std;

class l2_soft_margin_loss_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "soft_margin_loss_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "soft_margin_loss_test TearDown" << std::endl;
  }
};

// input nullptr
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_self_nullptr) {
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(nullptr, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// target nullptr
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_target_nullptr) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, nullptr, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_out_nullptr) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(nullptr));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dtype float32
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_float32) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// dtype double
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_double) {
  auto selfDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype bfloat16
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_bfloat16) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// empty tensor, reduction none
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_self_empty_none) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// empty tensor, reduction mean
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_self_empty_mean) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 1;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// empty tensor, reduction sum
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_self_empty_sum) {
  auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 2;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// reduction none
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_reduction_none) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// reduction mean
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_reduction_mean) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 1;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// reduction sum
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_reduction_sum) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 2;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// reduction other(3)
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_reduction_3) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 3;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

// reduction other(-4)
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_reduction_negative_4) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = -4;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  //ut.TestPrecision();
}

/*
// target broadcast
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_target_broadcast) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}

// self broadcast
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_self_broadcast) {
  auto selfDesc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_not_contiguous) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  ut.TestPrecision();
}
*/
// shape 9d
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_shape_9d) {
  auto selfDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype self float16 target float32
TEST_F(l2_soft_margin_loss_test, soft_margin_loss_float16_float32) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype bfloat16
TEST_F(l2_soft_margin_loss_test, ascend910B2_soft_margin_loss_bfloat16) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto targetDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{-1, 1, 1, -1});
  auto outDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  int64_t reduction = 0;

  auto ut = OP_API_UT(aclnnSoftMarginLoss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
