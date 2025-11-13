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
#include <float.h>
#include "gtest/gtest.h"
#include "opdev/op_log.h"

#include "../../../../op_host/op_api/aclnn_binary_cross_entropy_with_logits.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

enum Reduction {
  None,
  Mean,
  Sum,
  END
};

class l2BinaryCrossEntropyWithLogitsTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "binary_cross_entropy_with_logits_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "binary_cross_entropy_with_logits_test TearDown" << std::endl;
  }
};

// *** tensor dtype test ***
// test invalid input type
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_invalid_double_type) {
  auto self_tensor_desc = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::None;
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** bceLoss with logits reduction test ***
// default reduction, mean
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_default_reduction) {
  auto self_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::Mean;
  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// sum reduction
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_sum_reduction) {
  auto self_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::Sum;
  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// *** tensor relationship constraint test ***
// test out dtype is not same with target
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_diff_dtype_of_target_and_out) {
  auto self_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::None;
  auto out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test target and self have different shape
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_diff_shape_between_target_and_self) {
  auto self_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::None;
  auto out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test posweight shape cannot broadcast to target shape
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_unable_broadcast_between_posweight_and_target) {
  auto self_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::None;
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** tensor rank range ***
// empty tensor, with reduction is none
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_empty_tensor_under_none_reduction) {
  auto self_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::None;
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty tensor, with reduction is mean
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_empty_tensor_under_default_reduction) {
  auto self_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::Sum;
  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty tensor, with reduction is sum
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_empty_tensor_under_sum_reduction) {
  auto self_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::Sum;
  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test input tensor size is over limit, nine
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_input_size_overlimit) {
  auto self_tensor_desc = TensorDesc({5, 1, 3, 2, 4, 6, 3, 5, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 1, 3, 2, 4, 6, 3, 5, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 1, 3, 2, 4, 6, 3, 5, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 1, 3, 2, 4, 6, 3, 5, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::Sum;
  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** nullptr test ***
// test nullptr input
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_null_input) {
  auto target_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::Sum;
  auto out_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT((aclTensor*)nullptr, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr output
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_null_output) {
  auto self_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::Sum;

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT((aclTensor*)nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test illegal reduction
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_illegal_reduction) {
  auto self_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = 3;
  auto out_tensor_desc = TensorDesc(self_tensor_desc).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test wrong output shape with reduction
TEST_F(l2BinaryCrossEntropyWithLogitsTest, case_bcelosswithlogits_for_wrong_output_shape) {
  auto self_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto pos_weight_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = 1;
  auto out_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogits,
                INPUT(self_tensor_desc, target_tensor_desc, weight_tensor_desc, pos_weight_tensor_desc, reduction),
                OUTPUT(out_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
