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

#include "level2/aclnn_rrelu_with_noise.h"

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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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
  EXPECT_EQ(aclRet, ACL_SUCCESS);

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
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

//fp32 training = true
TEST_F(l2_rrelu_with_noise_test, case_fp16_training) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

//fp32 training = true
TEST_F(l2_rrelu_with_noise_test, case_fp_training) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
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
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 8维以上 1
TEST_F(l2_rrelu_with_noise_test, case_8_dim_1) {
  auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 8维以上 2
TEST_F(l2_rrelu_with_noise_test, case_8_dim_2) {
  auto self = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

//self != noise 1
TEST_F(l2_rrelu_with_noise_test, case_self_not_equal_noise_1) {
  auto self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto noise = TensorDesc({5, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
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
  EXPECT_EQ(aclRet, ACL_SUCCESS);

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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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
  EXPECT_EQ(aclRet, ACL_SUCCESS);

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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check format
// TEST_F(l2_rrelu_with_noise_test, case_format)
// {
//     vector<aclFormat> ValidList = {
//     ACL_FORMAT_UNDEFINED,
//     ACL_FORMAT_NCHW,
//     ACL_FORMAT_NHWC,
//     ACL_FORMAT_ND,
//     ACL_FORMAT_NC1HWC0,
//     ACL_FORMAT_FRACTAL_Z,
//     ACL_FORMAT_NC1HWC0_C04,
//     ACL_FORMAT_HWCN,
//     ACL_FORMAT_NDHWC,
//     ACL_FORMAT_FRACTAL_NZ,
//     ACL_FORMAT_NCDHW,
//     ACL_FORMAT_NDC1HWC0,
//     ACL_FRACTAL_Z_3D};

//     int length = ValidList.size();
//     for (int i = 0; i < length; i++) {
//         auto self = TensorDesc({5, 5}, ACL_FLOAT, ValidList[i]).ValueRange(0, 2);
//         auto noise = TensorDesc({5, 5}, ACL_FLOAT, ValidList[i]).ValueRange(0, 2);
//         auto lower = ScalarDesc(0.1f);
//         auto upper = ScalarDesc(0.5f);
//         bool training = false;
//         int64_t seed = 0;
//         int64_t offset = 0;
//         auto out = TensorDesc({5, 5}, ACL_FLOAT, ValidList[i]).Precision(0.005, 0.005);

//         auto ut = OP_API_UT(aclnnRReluWithNoise,
//                             INPUT(self, noise, lower, upper, training, seed, offset),
//                             OUTPUT(out));

//         uint64_t workspace_size = 0;
//         aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//         EXPECT_EQ(aclRet, ACL_SUCCESS);

//         // ut.TestPrecision();
//     }

//     for (int i = 0; i < length; i++) {
//         auto self = TensorDesc({5, 5}, ACL_FLOAT, ValidList[i]).ValueRange(0, 2);
//         auto noise = TensorDesc({5, 5}, ACL_FLOAT, ValidList[i]).ValueRange(0, 2);
//         auto lower = ScalarDesc(0.1f);
//         auto upper = ScalarDesc(0.5f);
//         bool training = true;
//         int64_t seed = 0;
//         int64_t offset = 0;
//         auto out = TensorDesc({5, 5}, ACL_FLOAT, ValidList[i]).Precision(0.005, 0.005);

//         auto ut = OP_API_UT(aclnnRReluWithNoise,
//                             INPUT(self, noise, lower, upper, training, seed, offset),
//                             OUTPUT(out));

//         uint64_t workspace_size = 0;
//         aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//         EXPECT_EQ(aclRet, ACL_SUCCESS);

//         // ut.TestPrecision();
//     }
// }

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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
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

//非连续1
TEST_F(l2_rrelu_with_noise_test, case_not_contiguous_1) {
  auto self = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {10, 5}).ValueRange(0, 2);
  auto noise = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {10, 5}).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
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
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_rrelu_with_noise_test, ascend910B2_case_dtype_inconsistent) {
  auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto noise = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto lower = ScalarDesc(0.1f);
  auto upper = ScalarDesc(0.5f);
  bool training = true;
  int64_t seed = 0;
  int64_t offset = 0;
  auto out = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.005, 0.005);

  auto ut = OP_API_UT(aclnnRReluWithNoise,
                      INPUT(self, noise, lower, upper, training, seed, offset),
                      OUTPUT(out));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}