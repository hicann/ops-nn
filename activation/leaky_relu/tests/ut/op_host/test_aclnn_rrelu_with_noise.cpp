/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_rrelu_with_noise.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_rrelu_with_noise_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "rrelu_with_noise_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "rrelu_with_noise_test TearDown" << endl; }
};

//nullptr1
TEST_F(l2_rrelu_with_noise_test, case_nullptr_1) {
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(nullptr, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

//nullptr2
TEST_F(l2_rrelu_with_noise_test, case_nullptr_2) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, nullptr, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

//nullptr3
TEST_F(l2_rrelu_with_noise_test, case_nullptr_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, nullptr, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//nullptr4
TEST_F(l2_rrelu_with_noise_test, case_nullptr_4) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, nullptr, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//nullptr4
TEST_F(l2_rrelu_with_noise_test, case_nullptr_5) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto upper = ScalarDesc(0.5f);
  auto lower = ScalarDesc(0.1f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, nullptr, training, seed, offset),
                      OUTPUT(nullptr));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//异常数据类型1
TEST_F(l2_rrelu_with_noise_test, case_int_1) {
  auto self = TensorDesc({5, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//异常数据类型2
TEST_F(l2_rrelu_with_noise_test, case_int_2) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//异常数据类型3
TEST_F(l2_rrelu_with_noise_test, case_int_3) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//fp16 training = false
TEST_F(l2_rrelu_with_noise_test, case_fp16_not_training) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  ut.TestPrecision();
}

//fp32 training = false
TEST_F(l2_rrelu_with_noise_test, case_fp_not_training) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  ut.TestPrecision();
}

//self != noise 2
TEST_F(l2_rrelu_with_noise_test, case_self_not_equal_noise_2) {
  auto self = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//self != noise 3
TEST_F(l2_rrelu_with_noise_test, case_self_not_equal_noise_3) {
  auto self = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  ut.TestPrecision();
}

//self != out
TEST_F(l2_rrelu_with_noise_test, case_self_not_equal_out) {
  auto self = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//输入输出dtype不一致
TEST_F(l2_rrelu_with_noise_test, case_dtype_inconsistent) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//输入输出format不一致
TEST_F(l2_rrelu_with_noise_test, case_format_inconsistent) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//输入输出format不一致
TEST_F(l2_rrelu_with_noise_test, case_noise_format_insame) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

}

//空tensor1
TEST_F(l2_rrelu_with_noise_test, case_empty_tensor_training) {
  auto self = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//空tensor2
TEST_F(l2_rrelu_with_noise_test, case_empty_tensor_not_training) {
  auto self = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

//非连续2
TEST_F(l2_rrelu_with_noise_test, case_not_contiguous_2) {
  auto self = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {10, 5}).ValueRange(0, 2);
  auto noise = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {10, 5}).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

  ut.TestPrecision();
}