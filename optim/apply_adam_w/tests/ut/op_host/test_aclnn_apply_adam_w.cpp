/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_apply_adam_w.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_apply_adam_w_test : public testing::Test {
protected:
  static void SetUpTestCase() { std::cout << "l2_apply_adam_w_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "l2_apply_adam_w_test TearDown" << std::endl; }
};

TEST_F(l2_apply_adam_w_test, case_nullptr) {
  auto var = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto m = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto v = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto maxGradNorm = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto grad = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = true;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(nullptr, m, v, beta1Power, beta2Power, lr, weightDecay,
                                             beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut1 = OP_API_UT(aclnnApplyAdamW, INPUT(var, nullptr, v, beta1Power, beta2Power, lr, weightDecay,
                                              beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, nullptr, beta1Power, beta2Power, lr, weightDecay,
                                              beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut3 = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                              beta1, beta2, eps, grad, nullptr, amsgrad, maximize), OUTPUT());
  aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut4 = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                              beta1, beta2, eps, nullptr, maxGradNorm, amsgrad, maximize), OUTPUT());
  aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut5 = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, nullptr,
                                              beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_apply_adam_w_test, case_null_tensor) {
  auto var = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto m = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto v = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto maxGradNorm = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto grad = TensorDesc({4, 3, 0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = true;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                             beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_apply_adam_w_test, case_error_dtype) {
  auto var = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto m = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto v = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto grad = TensorDesc({4, 3, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = true;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                             beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_apply_adam_w_test, case_error_shape) {
  auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto m = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto vERROR = TensorDesc({4, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gradError = TensorDesc({4, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = true;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, vERROR, beta1Power, beta2Power, lr, weightDecay, 
                                             beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  auto ut1 = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                              beta1, beta2, eps, gradError, maxGradNorm, amsgrad, maximize), OUTPUT());
  aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_apply_adam_w_test, case_maxGradNorm_is_not_null) {
  auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto m = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = true;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                             beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_apply_adam_w_test, case_maxGradNorm_is_null) {
  auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto m = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = false;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                             beta1, beta2, eps, grad, nullptr, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_apply_adam_w_test, case_diff_dtype) {
  auto var = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto m = TensorDesc({4, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto v = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto maxGradNorm = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto grad = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = true;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                             beta1, beta2, eps, grad, maxGradNorm, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_apply_adam_w_test, case_not_contiguous) {
  auto var = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  auto m = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  auto v = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  auto grad = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  auto lr = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta1Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto beta2Power = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto weightDecay = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  auto eps = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.0, 1.0);
  bool amsgrad = false;
  bool maximize = true;
  auto ut = OP_API_UT(aclnnApplyAdamW, INPUT(var, m, v, beta1Power, beta2Power, lr, weightDecay,
                                             beta1, beta2, eps, grad, nullptr, amsgrad, maximize), OUTPUT());
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}