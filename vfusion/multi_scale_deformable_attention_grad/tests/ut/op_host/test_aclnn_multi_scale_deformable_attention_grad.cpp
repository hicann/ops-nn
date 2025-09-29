/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/aclnn_multi_scale_deformable_attention_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

class l2_multi_scale_deformable_attention_grad_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_multi_scale_deformable_attention_grad_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_multi_scale_deformable_attention_grad_test TearDown" << std::endl;
  }
};

// 正常场景_float32
TEST_F(l2_multi_scale_deformable_attention_grad_test, multi_scale_deformable_attention_grad_float32) {
  auto valueDesc = TensorDesc({2, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({3, 2, 4, 4, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({3, 2, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto gradOutputDesc = TensorDesc({3, 2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto result1Desc = TensorDesc({2, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto result2Desc = TensorDesc({3, 2, 4, 4, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto result3Desc = TensorDesc({3, 2, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttentionGrad,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc, gradOutputDesc),
                      OUTPUT(result1Desc, result2Desc, result3Desc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_bfloat16
TEST_F(l2_multi_scale_deformable_attention_grad_test, multi_scale_deformable_attention_grad_bfloat16) {
  auto valueDesc = TensorDesc({2, 2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({3, 2, 4, 4, 5, 2}, ACL_BF16, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({3, 2, 4, 4, 5}, ACL_BF16, ACL_FORMAT_ND);
  auto gradOutputDesc = TensorDesc({3, 2, 4, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto result1Desc = TensorDesc({3, 2, 4, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto result2Desc = TensorDesc({3, 2, 4, 4, 5, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto result3Desc = TensorDesc({3, 2, 4, 4, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttentionGrad,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc, gradOutputDesc),
                      OUTPUT(result1Desc, result2Desc, result3Desc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype不支持_1
TEST_F(l2_multi_scale_deformable_attention_grad_test, multi_scale_deformable_attention_grad_float64) {
  if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
    auto valueDesc = TensorDesc({2, 2, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto locationDesc = TensorDesc({3, 2, 4, 4, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto attnWeightDesc = TensorDesc({3, 2, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradOutputDesc = TensorDesc({3, 2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto result1Desc = TensorDesc({3, 2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto result2Desc = TensorDesc({3, 2, 4, 4, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto result3Desc = TensorDesc({3, 2, 4, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMultiScaleDeformableAttentionGrad,
                        INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc, gradOutputDesc),
                        OUTPUT(result1Desc, result2Desc, result3Desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}