/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/op_api/aclnn_multilabel_margin_loss.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace std;

class l2_multilabel_margin_loss_test : public testing::Test
{
 protected:
  static void SetUpTestCase() {
    cout << "multilabel_margin_loss_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "multilabel_margin_loss_test TearDown" << endl;
  }
};

TEST_F(l2_multilabel_margin_loss_test, case_001)
{
  auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({3, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 5);
  int64_t reduction = 0;
  auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multilabel_margin_loss_test, case_002)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multilabel_margin_loss_test, case_003)
{
  auto selfDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 6);
  int64_t reduction = 2;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multilabel_margin_loss_test, case_004)
{
  auto selfDesc = TensorDesc({0, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({0, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
  int64_t reduction = 1;

  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({0, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multilabel_margin_loss_test, case_005)
{
  auto selfDesc = TensorDesc({0, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({0, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
  int64_t reduction = 0;

  auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({0, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multilabel_margin_loss_test, case_006)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multilabel_margin_loss_test, case_007)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 0;
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
}

TEST_F(l2_multilabel_margin_loss_test, case_008)
{
  auto selfDesc = TensorDesc({10, 1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
}

TEST_F(l2_multilabel_margin_loss_test, case_009)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT((aclTensor*)nullptr, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, case_010)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, (aclTensor*)nullptr, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, case_011)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT((aclTensor*)nullptr, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, case_012)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 2;
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, (aclTensor*)nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, case_013)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 0;
  auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
}

TEST_F(l2_multilabel_margin_loss_test, Ascend910B2_case_014)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_multilabel_margin_loss_test, Ascend910B2_case_015)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT((aclTensor*)nullptr, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, Ascend910B2_case_016)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, (aclTensor*)nullptr, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, Ascend910B2_case_017)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 1;
  auto outDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT((aclTensor*)nullptr, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, Ascend910B2_case_018)
{
  auto selfDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({10, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 2;
  auto outDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto isTargetDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, (aclTensor*)nullptr));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_multilabel_margin_loss_test, case_019)
{
  auto selfDesc = TensorDesc({40, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto targetDesc = TensorDesc({40, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 6);
  int64_t reduction = 0;
  auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto isTargetDesc = TensorDesc({40, 10}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultilabelMarginLoss, INPUT(selfDesc, targetDesc, reduction),
                      OUTPUT(outDesc, isTargetDesc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // SAMPLE: precision simulate
}
