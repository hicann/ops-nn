/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file test_fused_cross_entropy_loss_with_max_sum.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_fused_cross_entropy_loss_with_max_sum.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"


using namespace std;
using namespace op;

class l2_fused_cross_entropy_loss_with_max_sum_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "fused_cross_entropy_loss_with_max_sum_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "fused_cross_entropy_loss_with_max_sum_test TearDown" << endl; }
};

TEST_F(l2_fused_cross_entropy_loss_with_max_sum_test, case_0) {
  auto logitsMax = TensorDesc({4}, ACL_FLOAT,ACL_FORMAT_ND).ValueRange(-1,1);
  auto sumExpLogits = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
  auto predictedLogits = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,5);
  auto inputOptional = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,5);
  auto weightOptional = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,5);
  auto vocabParallelLogitsOptional = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,5);
  float labelSmoothing = 0.0f;
  auto lossOut = TensorDesc({4}, ACL_FLOAT,ACL_FORMAT_ND).ValueRange(-1,1);
  auto softMaxOutOptional = TensorDesc({4, 2}, ACL_FLOAT,ACL_FORMAT_ND).ValueRange(0,5);
  auto ut = OP_API_UT(aclnnFusedCrossEntropyLossWithMaxSum,
                       INPUT(logitsMax,sumExpLogits,predictedLogits,labelSmoothing,inputOptional,weightOptional,
                             vocabParallelLogitsOptional),
                       OUTPUT(lossOut,softMaxOutOptional));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_fused_cross_entropy_loss_with_max_sum_test, case_1) {
  auto logitsMax = TensorDesc({4}, ACL_FLOAT,ACL_FORMAT_ND).ValueRange(-1,1);
  auto sumExpLogits = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 5);
  auto predictedLogits = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0,5);
  float labelSmoothing = 0.0f;
  auto lossOut = TensorDesc({4}, ACL_FLOAT,ACL_FORMAT_ND).ValueRange(-1,1);
  auto ut = OP_API_UT(aclnnFusedCrossEntropyLossWithMaxSum,
                       INPUT(logitsMax,sumExpLogits,predictedLogits,labelSmoothing,nullptr,nullptr,
                             nullptr),
                       OUTPUT(lossOut,nullptr));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  // ut.TestPrecision();
}
