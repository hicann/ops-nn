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
#include "../../../op_host/op_api/aclnn_scatter_update.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2InplaceScatterUpdateTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "l2InplaceScatterUpdateTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2InplaceScatterUpdateTest TearDown" << std::endl;
  }
};

// 异常场景：self为空指针，返回ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_self_null) {
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT8, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT((aclTensor *)nullptr, indicesDesc,
                      updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：update为空指针，返回ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_update_null) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT8, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc,
                      (aclTensor *)nullptr, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：indices为空指针，返回ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_indices_null) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_INT8, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_INT8, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, (aclTensor *)nullptr,
                      updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：self的数据类型不在支持范围内，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_self_complex) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self和update的数据类型不一致，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_self_update_type_nonconsistent) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：self和update的shape不一致，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_self_update_shape_nonconsistent) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({2, 3, 1, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：Tensor为5维、返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_dim_five) {
  auto selfDesc = TensorDesc({4, 3, 9, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({2, 3, 9, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  
}

// 异常场景：Tensor为空、返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_tensor_null) {
  auto selfDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
  int64_t axis = 0;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：indices维度不为1或2，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_indices_dimsize) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({1, 1, 4}, ACL_INT32, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：indices维度0, update batch轴不等于1，返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_err_batch_dim) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景：self为4维float16类型、axis为-2，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_type_float16) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 正常场景：self为4维float32类型、axis为-2，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_type_float32) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 正常场景：self为4维int32类型、axis为-2，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_type_int32) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_INT32, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_INT32, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  
}

// 正常场景：self为4维int8类型、axis为-2，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_type_int8) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_INT8, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_INT8, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 正常场景：self为4维bfloat16类型、axis为-2，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, ascend910B2_l2_inplace_scatter_update_test_type_bfloat16) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_BF16, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_BF16, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  
}

// 正常场景：self为4维ACL_FLOAT类型、axis为-1，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_axis_neg_one) {
  auto selfDesc = TensorDesc({4, 8, 64, 64}, ACL_FLOAT, ACL_FORMAT_ND);
  auto updateDesc = TensorDesc({4, 8, 64, 32}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -1;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 正常场景：self为4维ACL_FLOAT类型，format为ACL_FORMAT_NCHW、axis为-2，tensor值未更新，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_bfloat16_nchw) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常场景：self为4维ACL_FLOAT类型，format为ACL_FORMAT_NHWC、axis为-2，tensor值未更新，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_bfloat16_nhwc) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT, ACL_FORMAT_NHWC);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_NHWC).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常场景：self为4维ACL_FLOAT类型，format为ACL_FORMAT_NC1HWC0、axis为-2，tensor值未更新，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_bfloat16_nc1hwc0) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_NC1HWC0).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常场景：self为4维ACL_FLOAT类型，format为ACL_FORMAT_HWCN、axis为-2，tensor值未更新，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_bfloat16_hwcn) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_HWCN).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 正常场景：self为4维ACL_FLOAT类型，format为ACL_FORMAT_HWCN, non-continuous、axis为-2，tensor值未更新，返回ACLNN_SUCCESS，精度校验通过
TEST_F(l2InplaceScatterUpdateTest, l2_inplace_scatter_update_test_bfloat16_hwcn_noncontinuous) {
  auto selfDesc = TensorDesc({4, 8, 10, 64}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5, 2});
  auto updateDesc = TensorDesc({4, 8, 5, 64}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5, 2});
  auto indicesDesc = TensorDesc({4,}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
  int64_t axis = -2;

  auto ut = OP_API_UT(aclnnInplaceScatterUpdate, INPUT(selfDesc, indicesDesc, updateDesc, axis), OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}
