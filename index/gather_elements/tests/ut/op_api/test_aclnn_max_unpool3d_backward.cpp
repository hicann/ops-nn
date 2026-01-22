/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_max_unpool3d_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_max_unpool3d_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "l2_max_unpool3d_backward_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_max_unpool3d_backward_test TearDown" << std::endl;
  }
};

TEST_F(l2_max_unpool3d_backward_test, case_gradOutput_nullptr) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(nullptr, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_unpool3d_backward_test, case_self_nullptr) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, nullptr, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_unpool3d_backward_test, case_indices_nullptr) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, nullptr, output_szie, stride, padding), OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_unpool3d_backward_test, case_out_nullptr) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(nullptr));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_unpool3d_backward_test, case_bfloat16) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_BF16, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_gradOutput_shape2) {
  auto gradOutput_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_self_shape2) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_self_shape5) {
  auto gradOutput_desc = TensorDesc({1, 1, 2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_indices_shape3) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_output_size3) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_0_tensor) {
  auto gradOutput_desc = TensorDesc({0, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({0, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({0, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({0, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_neg1_tensor) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, -1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_neg1_output_size) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, -4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_float32) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_float16) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_int8) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_INT8, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT8, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_uint8) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_UINT8, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_UINT8, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_int16) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_INT16, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT16, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_INT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_int32) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_int64) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_INT64, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool3d_backward_test, case_bool) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_complex64) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_complex128) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX128, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_stride_size4) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool3d_backward_test, case_padding_size4) {
  auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
  auto output_szie = IntArrayDesc(vector<int64_t>{1, 4, 4});
  auto stride = IntArrayDesc(vector<int64_t>{1, 2, 3});
  auto padding = IntArrayDesc(vector<int64_t>{1, 1, 2, 3});
  auto out_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnMaxUnpool3dBackward, INPUT(gradOutput_desc, self_desc, indices_desc, output_szie, stride, padding), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}