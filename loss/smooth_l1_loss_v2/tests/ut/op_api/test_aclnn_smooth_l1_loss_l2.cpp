/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "../../../op_host/op_api/aclnn_smooth_l1_loss.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace std;

class smooth_l1_loss : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "smooth_l1_loss_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "smooth_l1_loss_test TearDown" << endl;
  }
};

TEST_F(smooth_l1_loss, smooth_l1_loss_fp16_none) {
  auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  int64_t reduction = 0;
  float beta = 2.0;

  auto outDesc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(smooth_l1_loss, smooth_l1_loss_fp32_none) {
  auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  int64_t reduction = 0;
  float beta = 2.0;

  auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(smooth_l1_loss, smooth_l1_loss_fp16_mean) {
  auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  int64_t reduction = 1;
  float beta = 2.0;

  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(smooth_l1_loss, smooth_l1_loss_fp16_none_NCL) {
  auto selfDesc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  int64_t reduction = 0;
  float beta = 2.0;

  auto outDesc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(smooth_l1_loss, smooth_l1_loss_fp16_none_boardcast_NCL) {
  auto selfDesc = TensorDesc({5, 7}, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  int64_t reduction = 0;
  float beta = 2.0;

  auto outDesc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(smooth_l1_loss, ascend910B2_smooth_l1_loss_bf16_none) {
  auto selfDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
  int64_t reduction = 0;
  float beta = 2.0;

  auto outDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(smooth_l1_loss, ascend910B2_smooth_l1_loss_empty_tensor) {
  auto selfDesc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  int64_t reduction = 1;
  float beta = 2.0;

  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.004, 0.004);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

TEST_F(smooth_l1_loss, ascend910B2_smooth_l1_loss_empty_tensor_none) {
  auto selfDesc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  int64_t reduction = 0;
  float beta = 2.0;

  auto outDesc = TensorDesc({3, 0}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.004, 0.004);

  auto ut = OP_API_UT(aclnnSmoothL1Loss, INPUT(selfDesc, targetDesc, reduction, beta),
                      OUTPUT(outDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  // ut.TestPrecision();
}