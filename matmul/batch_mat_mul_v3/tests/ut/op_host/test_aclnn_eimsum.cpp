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
#include "../../../op_host/op_api/aclnn_einsum.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"

class l2_einsum_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_einsum_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_einsum_test TearDown" << std::endl;
  }
};

// 检查入参是否为nullptr
TEST_F(l2_einsum_test, case_nullptr_1) {
  auto input1 = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "bhid,bhijd->bhij";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(nullptr, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_einsum_test, case_nullptr_2) {
  auto input1 = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "bhid,bhijd->bhij";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(nullptr));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持类型
TEST_F(l2_einsum_test, case_invalid_dType_1) {
  auto input1 = TensorDesc({1, 2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_BOOL, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_invalid_dType_2) {
  auto input1 = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_invalid_dType_3) {
  auto input1 = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_INT16, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_invalid_format) {
  auto input1 = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_INT16, ACL_FORMAT_ND);
  auto ut0 = OP_API_UT(aclnnEinsum,
		       INPUT(tensorListDesc, equation),
		       OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// tensor 数量不对
TEST_F(l2_einsum_test, case_invalid_tensor_num) {
  auto input1 = TensorDesc({2, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// tensor dtype不一致
TEST_F(l2_einsum_test, case_invalid_tensor_dtype) {
  auto input1 = TensorDesc({2, 2, 3, 4}, ACL_INT16, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_INT32, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Dim不对
TEST_F(l2_einsum_test, case_invalid_dim_1) {
  auto input1 = TensorDesc({2, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_invalid_dim_2) {
  auto input1 = TensorDesc({1, 1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_invalid_dim_3) {
  auto input1 = TensorDesc({1, 2, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_invalid_dim_4) {
  auto input1 = TensorDesc({1, 2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_invalid_dim_5) {
  auto input1 = TensorDesc({2, 2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 5, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 5}, ACL_INT32, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tesor传入
TEST_F(l2_einsum_test, case_empty_tensor_1) {
  auto input1 = TensorDesc({1, 0, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 0, 3, 5, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 0, 3, 5}, ACL_INT32, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_empty_tensor_2) {
  auto input1 = TensorDesc({1, 2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto input2 = TensorDesc({1, 2, 3, 0, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "abcd,abced->abce";
  auto out = TensorDesc({1, 2, 3, 0}, ACL_INT32, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_einsum_test, case_norm_2) {
  auto input1 = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  auto input2 = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
  auto tensorListDesc = TensorListDesc({input1, input2});
  auto equation = "a,b->ab";
  auto out = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

  auto ut0 = OP_API_UT(aclnnEinsum,
                      INPUT(tensorListDesc, equation),
                      OUTPUT(out));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut0.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
