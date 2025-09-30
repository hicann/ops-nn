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
#include "aclnn_multi_scale_deformable_attn_function.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

class l2_multi_scale_deformable_attn_function_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_multi_scale_deformable_attn_function_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_multi_scale_deformable_attn_function_test TearDown" << std::endl;
  }
};

// 正常场景_float32
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_float32) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_FLOAT, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_float16
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_float16) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 5, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_bfloat16
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_bfloat16) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_BF16, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 5, 2}, ACL_BF16, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_BF16, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dtype不支持_1
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_float64) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 5, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype不支持_2
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_int64) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_FLOAT, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拦截非正常情况 numQueries < 32
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_numQueries) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 16, 3, 4, 5, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拦截非正常情况 numHeads > 16
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_numHeads) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 32, 4, 5, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拦截非正常情况 numLevels > 16
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_numLevels) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 32, 5, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拦截非正常情况 numPoints > 16
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_numPoints) {
  auto valueDesc = TensorDesc({2, 2, 3, 32}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 32, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拦截非正常情况 embedDims % 8 != 0
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_embedDims_div) {
  auto valueDesc = TensorDesc({2, 2, 3, 30}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 5, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 拦截非正常情况 embedDims > 256
TEST_F(l2_multi_scale_deformable_attn_function_test, multi_scale_deformable_attn_function_embedDims) {
  auto valueDesc = TensorDesc({2, 2, 3, 512}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({2, 32, 3, 4, 5, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({2, 32, 3, 4, 5}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({2, 32, 3 * 32}, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 310p 正常场景 float32
TEST_F(l2_multi_scale_deformable_attn_function_test, ascend310P_multi_scale_deformable_attn_function_310p_float32) {
  auto valueDesc = TensorDesc({1, 4, 4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
  auto shapeDesc = TensorDesc({1, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto levelStartIndexDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto locationDesc = TensorDesc({1, 128, 4, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto attnWeightDesc = TensorDesc({1, 128, 4, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outputDesc = TensorDesc({1, 128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMultiScaleDeformableAttnFunction,
                      INPUT(valueDesc, shapeDesc, levelStartIndexDesc, locationDesc, attnWeightDesc),
                      OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}