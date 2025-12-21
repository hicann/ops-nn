/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv_tbc.cpp
 * \brief
 */

#include <array>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_convolution.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

#define KEEP_DTYPE 0
#define ALLOW_FP32_DOWN_PRECISION 1
#define USE_HF32 3
static int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
static int8_t cubeMathTypeHF32 = USE_HF32;
class conv_tbc_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "conv_tbc_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "conv_tbc_test TearDown" << endl; }
};

TEST_F(conv_tbc_test, test_convTbc_dtype_error) {
  vector<aclDataType> ValidList = {ACL_DT_UNDEFINED, ACL_INT8,   ACL_INT32,     ACL_UINT8,      ACL_INT16,
                                   ACL_UINT16,       ACL_UINT32, ACL_INT64,     ACL_UINT64,     ACL_DOUBLE,
                                   ACL_BOOL,         ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128, ACL_BF16};
  int length = ValidList.size();

  vector<int64_t> inp_dims = {16, 16, 8};
  vector<int64_t> weight_dims = {3, 8, 16};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {16, 16, 16};
  int64_t pad = 1;
  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i],   ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i],   ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, ValidList[i],   ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(conv_tbc_test, test_convTbc_bias_dim_error) {
  vector<int64_t> inp_dims = {16, 16, 8};
  vector<int64_t> weight_dims = {3, 8, 16};
  vector<int64_t> bias_dims = {1, 1, 16};
  vector<int64_t> output_dims = {16, 16, 16};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_1) {
  vector<int64_t> inp_dims = {0, 0, 0};
  vector<int64_t> weight_dims = {0, 0, 0};
  vector<int64_t> bias_dims = {0};
  vector<int64_t> output_dims = {3, 0, 0};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_2) {
  vector<int64_t> inp_dims = {16, 16, 0};
  vector<int64_t> weight_dims = {3, 0, 16};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {16, 16, 16};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS); // input zero output not zero
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_3) {
  vector<int64_t> inp_dims = {1, 16, 8};
  vector<int64_t> weight_dims = {4, 8, 16};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {0, 16, 16};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_4) {
  vector<int64_t> inp_dims = {32, 2, 16};
  vector<int64_t> weight_dims = {3, 16, 64};
  vector<int64_t> bias_dims = {0};
  vector<int64_t> output_dims = {30, 2, 64};
  int64_t pad = 0;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_5) {
  vector<int64_t> inp_dims = {0, 2, 16};
  vector<int64_t> weight_dims = {3, 16, 64};
  vector<int64_t> bias_dims = {64};
  vector<int64_t> output_dims = {-2, 2, 64};
  int64_t pad = 0;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_6) {
  vector<int64_t> inp_dims = {0, 2, 16};
  vector<int64_t> weight_dims = {1, 16, 64};
  vector<int64_t> bias_dims = {64};
  vector<int64_t> output_dims = {0, 2, 64};
  int64_t pad = 0;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_7) {
  vector<int64_t> inp_dims = {32, 2, 0};
  vector<int64_t> weight_dims = {3, 16, 64};
  vector<int64_t> bias_dims = {64};
  vector<int64_t> output_dims = {30, 2, 64};
  int64_t pad = 0;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_empty_tensor_10) {
  vector<int64_t> inp_dims = {32, 2, 16};
  vector<int64_t> weight_dims = {0, 16, 64};
  vector<int64_t> bias_dims = {64};
  vector<int64_t> output_dims = {30, 2, 64};
  int64_t pad = 0;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_dim_error) {
  vector<int64_t> inp_dims = {16, 16, 8, 4};
  vector<int64_t> weight_dims = {3, 8, 16, 4};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {16, 16, 16, 4};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_dim_error_channel_1) {
  vector<int64_t> inp_dims = {16, 16, 8};
  vector<int64_t> weight_dims = {3, 10, 16};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {16, 16, 16};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(conv_tbc_test, test_convTbc_dim_error_channel_2) {
  vector<int64_t> inp_dims = {16, 16, 8};
  vector<int64_t> weight_dims = {3, 8, 16};
  vector<int64_t> bias_dims = {8};
  vector<int64_t> output_dims = {16, 16, 16};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_tbc_test, test_convTbc_nullptr_error) {
  vector<int64_t> inp_dims = {16, 16, 8};
  vector<int64_t> weight_dims = {3, 8, 16};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {16, 16, 16};
  int64_t pad = 1;
  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16,   ACL_FORMAT_ND).ValueRange(0, 2);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16,   ACL_FORMAT_ND).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvTbc,                           // host api第二段接口名称
                      INPUT((aclTensor *)nullptr, weight_desc, bias_desc, pad),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
  // ut.TestPrecision();
}

TEST_F(conv_tbc_test, test_convTbc_cubeMathTypeHF32) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  vector<int64_t> inp_dims = {16, 16, 8};
  vector<int64_t> weight_dims = {3, 8, 16};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {16, 16, 16};
  int64_t pad = 1;
  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT,   ACL_FORMAT_ND).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT,   ACL_FORMAT_ND).ValueRange(0, 2);
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT,   ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvTbc,                                  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, pad),  // host api输入
                        OUTPUT(output_desc), cubeMathTypeHF32);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}
