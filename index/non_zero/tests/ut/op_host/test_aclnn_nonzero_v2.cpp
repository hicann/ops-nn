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

#include "../../../op_host/op_api/aclnn_nonzero_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_nonzero_v2_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "nonzero_v2_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "nonzero_v2_test TearDown" << std::endl; }
};


// 正常场景_FLOAT
TEST_F(l2_nonzero_v2_test, case_001_FLOAT) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_FLOAT_16
TEST_F(l2_nonzero_v2_test, case_002_FLOAT16) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_BF16
TEST_F(l2_nonzero_v2_test, case_003_FP16) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_INT8
TEST_F(l2_nonzero_v2_test, case_007_INT8) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 10);
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_UINT8
TEST_F(l2_nonzero_v2_test, case_008_UINT8) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(1, 10);
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景_BOOL
TEST_F(l2_nonzero_v2_test, case_009_BOOL) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 10);
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 不支持场景_COMPLEX128
TEST_F(l2_nonzero_v2_test, case_010_COMPLEX128) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({4, 120}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不支持场景_COMPLEX64
TEST_F(l2_nonzero_v2_test, case_011_COMPLEX64) {
  auto self_tensor_desc = TensorDesc({2, 3, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({4, 120}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_UNDEFINED
TEST_F(l2_nonzero_v2_test, case_016_UNDEFINED) {
  auto selfDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(selfDesc), OUTPUT(outDesc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2_nonzero_v2_test, case_017_EMPTY) {
  auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 0}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空指针
TEST_F(l2_nonzero_v2_test, case_018_NULLPTR) {
  auto self_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto out_tensor_desc = TensorDesc({2, 50}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(nullptr), OUTPUT(out_tensor_desc));
  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut_2 = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(nullptr));
  // SAMPLE: only test GetWorkspaceSize
  aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// dim超出限制
TEST_F(l2_nonzero_v2_test, case_019_MAX_DIM) {
  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({9, 200200200}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nonzero_v2_test, case_0191_MAX_DIM) {
  auto self_tensor_desc = TensorDesc({2147483648,}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto out_tensor_desc = TensorDesc({2147483648,}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续
TEST_F(l2_nonzero_v2_test, case_020_uncontiguous) {
  auto self_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(1, 20);
  auto out_tensor_desc = TensorDesc({2, 8}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // only test GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nonzero_v2_test, Ascend910_95_case_001_FLOAT) {
  auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
  auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnNonzeroV2, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}
