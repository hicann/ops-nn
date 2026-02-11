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

#include "level2/aclnn_binary_cross_entropy_with_logits_target_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

enum Reduction { None, Mean, Sum, END };

class l2_binary_cross_entropy_with_logits_target_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "binary_cross_entropy_with_logits_target_backward_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "binary_cross_entropy_with_logits_target_backward_test TearDown" << endl;
  }
};

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_f16_reduce_valid) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  auto weight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  auto posWeight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
  }
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_f32_reduce_valid) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  auto weight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(weight_value);
  auto posWeight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
  }
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_f16_reduce_valid_no_weight) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  aclTensor* weight = nullptr;
  auto posWeight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
  }
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_f32_reduce_none_no_weight) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  aclTensor* weight = nullptr;
  auto posWeight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(weight_value);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
  }
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_f16_reduce_valid_no_posWeight) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  aclTensor* posWeight = nullptr;
  auto weight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
  }
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_f32_reduce_valid_no_posWeight) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  aclTensor* posWeight = nullptr;
  auto weight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(weight_value);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
  }
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_contiguous) {
  auto gradOutput = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 1);
  auto input = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(20, 1.0);
  auto target = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(20, 2.0);
  auto weight = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Value(weight_value);
  auto posWeight = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Value(weight_value);
  auto output = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
  }
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_empty) {
  auto gradOutput = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto weight = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto posWeight = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gradTarget = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Mean;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                      INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(gradTarget));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_dtype_promotion) {
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  auto weight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  auto posWeight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  int64_t reduction = None;
  uint64_t workspace_size = 0;

  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                      INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_self_float32_type_promotion) {
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  auto weight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  auto posWeight = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(weight_value);
  int64_t reduction = None;
  uint64_t workspace_size = 0;

  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                      INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_NC1HWC0) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 16}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 16}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(384, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 16}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(384, 2.0);
  auto weight = TensorDesc({1, 2, 3, 4, 16}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Value(weight_value);
  auto posWeight = TensorDesc({1, 2, 3, 4, 16}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Value(weight_value);
  int64_t reduction = Mean;
  auto output = TensorDesc({1, 2, 3, 4, 16}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                      INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_FRACTAL_Z) {
  auto gradOutput = TensorDesc({16, 16, 16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z).ValueRange(0, 1);
  auto input = TensorDesc({16, 16, 16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(65536, 1.0);
  auto target = TensorDesc({16, 16, 16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(65536, 2.0);
  auto weight = TensorDesc({16, 16, 16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z).Value(weight_value);
  auto posWeight = TensorDesc({16, 16, 16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z).Value(weight_value);
  int64_t reduction = Mean;
  auto output = TensorDesc({16, 16, 16, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_Z).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                      INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// abnormal
TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_unsupport_dtype) {
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Value(target_value);
  int64_t reduction = None;
  uint64_t workspace_size = 0;

  auto gradOutput1 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input1 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);
  auto output1 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_1 = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput1, input1, target, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction),
                        OUTPUT(output1));

  aclnnStatus aclRet_1 = ut_1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_1, ACLNN_ERR_PARAM_INVALID);

  auto gradOutput2 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input2 = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target2 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Value(target_value);
  auto output2 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_2 = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput2, input2, target2, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction),
                        OUTPUT(output2));

  aclnnStatus aclRet_2 = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_2, ACLNN_ERR_PARAM_INVALID);

  auto gradOutput3 = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input3 = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target3 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Value(target_value);
  auto output3 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_3 = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput3, input3, target3, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction),
                        OUTPUT(output3));

  aclnnStatus aclRet_3 = ut_3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_3, ACLNN_ERR_PARAM_INVALID);

  auto gradOutput4 = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input4 = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target4 = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  auto output4 = TensorDesc({1, 2, 3, 4, 5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut_4 = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput4, input4, target4, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction),
                        OUTPUT(output4));

  aclnnStatus aclRet_4 = ut_4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet_4, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_dimension8) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(362880, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  int64_t reduction = Sum;
  auto output = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                INPUT(gradOutput, input, target, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction), OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_out_format_invalid) {
  auto gradOutput = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(362880, 1.0);
  auto target = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  int64_t reduction = Sum;
  auto output = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                INPUT(gradOutput, input, target, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction), OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_unsupport_shape) {
  auto gradOutput = TensorDesc({3, 1, 4, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({5, 4, 3, 2, 1}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  int64_t reduction = Sum;
  auto output = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                INPUT(gradOutput, input, target, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction), OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, case_null) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(target_value);
  int64_t reduction = Sum;
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut1 = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                       INPUT((aclTensor*)nullptr, input, target, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction),
                       OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                INPUT(gradOutput, (aclTensor*)nullptr, target, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction),
                OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut3 =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                INPUT(gradOutput, input, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction),
                OUTPUT(output));

  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut4 =
      OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                INPUT(gradOutput, input, target, (aclTensor*)nullptr, (aclTensor*)nullptr, reduction), OUTPUT(nullptr));

  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_binary_cross_entropy_with_logits_target_backward_test, Ascend910B2_case_bf16_reduce_valid) {
  auto gradOutput = TensorDesc({1, 2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto input = TensorDesc({1, 2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
  vector<float> target_value;
  target_value.resize(120, 1.0);
  auto target = TensorDesc({1, 2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).Value(target_value);
  vector<float> weight_value;
  weight_value.resize(120, 2.0);
  auto weight = TensorDesc({1, 2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).Value(weight_value);
  auto posWeight = TensorDesc({1, 2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).Value(weight_value);
  auto output = TensorDesc({1, 2, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  vector<int64_t> reductions{None, Mean, Sum};
  for (auto reduction : reductions) {
    auto ut = OP_API_UT(aclnnBinaryCrossEntropyWithLogitsTargetBackward,
                        INPUT(gradOutput, input, target, weight, posWeight, reduction), OUTPUT(output));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}
