/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "opdev/op_log.h"

#include "../../../op_host/op_api/aclnn_binary_cross_entropy.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"


class l2_binary_cross_entropy_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "binary_cross_entropy_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "binary_cross_entropy_test TearDown" << std::endl; }
};

//空指针1
TEST_F(l2_binary_cross_entropy_test, case_nullptr_1) {
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(nullptr, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

//空指针2
TEST_F(l2_binary_cross_entropy_test, case_nullptr_2) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, nullptr, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

//空指针3
TEST_F(l2_binary_cross_entropy_test, case_nullptr_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

//异常数据类型1
TEST_F(l2_binary_cross_entropy_test, case_double_1) {
  auto self = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常数据类型2
TEST_F(l2_binary_cross_entropy_test, case_double_2) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常数据类型3
TEST_F(l2_binary_cross_entropy_test, case_double_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常数据类型4
TEST_F(l2_binary_cross_entropy_test, case_double_4) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//异常输入数据1
TEST_F(l2_binary_cross_entropy_test, case_reduction_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = 3;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//输入shape不一致
TEST_F(l2_binary_cross_entropy_test, case_shape_abnormal) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int64_t reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//空tensor float Reduction::None
TEST_F(l2_binary_cross_entropy_test, case_fp_empty_tensor_none) {
  auto self = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//空tensor float16 Reduction::None
TEST_F(l2_binary_cross_entropy_test, case_fp16_empty_tensor_none) {
  auto self = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::None;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  //ut.TestPrecision();
}

//输入输出dtype不一致
TEST_F(l2_binary_cross_entropy_test, case_dtype_inconsistent) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto target = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto weight = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
  auto out = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
  int reduction = Reduction::Mean;

  auto ut = OP_API_UT(aclnnBinaryCrossEntropy,
                      INPUT(self, target, weight, reduction),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}