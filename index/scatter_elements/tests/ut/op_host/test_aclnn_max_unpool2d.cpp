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
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_max_unpool2d.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_max_unpool2d_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "l2_max_unpool2d_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_max_unpool2d_test TearDown" << std::endl;
  }
};

TEST_F(l2_max_unpool2d_test, case_self_nullptr) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(nullptr, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_unpool2d_test, case_indices_nullptr) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, nullptr, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_unpool2d_test, case_out_nullptr) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_max_unpool2d_test, case_bfloat16) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_BF16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_self_shape2) {
  auto self_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_self_shape5) {
  auto self_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_indices_shape3) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_output_size3) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_0_tensor) {
  auto self_desc = TensorDesc({0, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({0, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_max_unpool2d_test, case_neg1_tensor) {
  auto self_desc = TensorDesc({1, -1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_neg1_output_size) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, -4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_bool) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_BOOL, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_BOOL, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_complex64) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_COMPLEX64, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_max_unpool2d_test, case_complex128) {
  auto self_desc = TensorDesc({1, 1, 2, 2}, ACL_COMPLEX128, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({1, 1, 2, 2}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{4, 4});
  auto out_desc = TensorDesc({1, 1, 4, 4}, ACL_COMPLEX128, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnMaxUnpool2d, INPUT(self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
