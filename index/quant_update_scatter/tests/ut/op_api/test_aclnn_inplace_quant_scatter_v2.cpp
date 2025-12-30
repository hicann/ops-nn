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

#include "../../../op_api/aclnn_quant_scatter_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_inplace_quant_scatter_test_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_inplace_quant_scatter_test_v2 SetUp" << endl;
  }
  static void TearDownTestCase() {
    cout << "l2_inplace_quant_scatter_test_v2 TearDown" << endl;
  }
};

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_dtype_combine_err1) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // precision simulate
  
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_dtype_combine_err2) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = nullptr;
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // precision simulate
  
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_self_dtype_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT64, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_indices_dtype_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT8, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_updates_dtype_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_quantScales_dtype_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_quantZeroPoints_dtype_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_INT64, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_round_mode_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "round";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_hi8_round_mode_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({24, 4096, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_quant_scatter_test_v2, ascend910_95_test_abnormal_shape_invalid) {
  // 使用**Desc描述host api输入输出
  auto selfRefDesc = TensorDesc({2, 12, 4096, 128}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({24}, ACL_INT32, ACL_FORMAT_ND);
  auto updatesDesc = TensorDesc({24, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantScalesDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  auto quantZeroPointsDesc = TensorDesc({1, 1, 128}, ACL_BF16, ACL_FORMAT_ND);
  int64_t axis = -2;
  int64_t quantAxis = -1;
  int64_t reduction = 1;
  const char* roundMode = "rint";
  auto ut = OP_API_UT(
      aclnnInplaceQuantScatterV2,
      INPUT(selfRefDesc, indicesDesc, updatesDesc, quantScalesDesc, quantZeroPointsDesc, axis, quantAxis, reduction, roundMode),
      OUTPUT());

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}