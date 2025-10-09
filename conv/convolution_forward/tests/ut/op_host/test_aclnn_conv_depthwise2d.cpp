/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv_depthwise2d.cpp
 * \brief
 */

#include <float.h>

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
class conv_depthwise2d_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "conv_depthwise2d_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "conv_depthwise2d_test TearDown" << endl; }
};

TEST_F(conv_depthwise2d_test, test_input_FP16_weight_FP16_output_FP32) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_depthwise2d_test, test_input_FP32_weight_FP32_output_FP16) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_depthwise2d_test, test_input_FP32_weight_FP16_output_FP16) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_depthwise2d_test, test_input_FP32_weight_FP16_output_FP32) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_depthwise2d_test, test_NCHW) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                              dilation_desc),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(conv_depthwise2d_test, test_NHWC) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  vector<int64_t> inp_dims = {2, 32, 16, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 32, 16, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NHWC).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                              dilation_desc),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
  }
}

TEST_F(conv_depthwise2d_test, test_NCL) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                              dilation_desc),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
  }
}

TEST_F(conv_depthwise2d_test, test_kernelSize_err) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  vector<int64_t> inp_dims = {2, 32, 16, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {5, 5};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 32, 16, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                              dilation_desc),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
  }
}

TEST_F(conv_depthwise2d_test, test_infershape_error) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 33, 9};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input dim should be equal to weight
TEST_F(conv_depthwise2d_test, test_input_weight_dim_error_1) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// attr dim should be equal to input dim - 2
TEST_F(conv_depthwise2d_test, test_attr_dim_error) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input weight dim number should be 4
TEST_F(conv_depthwise2d_test, test_input_weight_dim_error_2) {
  vector<int64_t> inp_dims = {1, 2, 16, 32, 16};
  vector<int64_t> weight_dims = {1, 16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Nullptr
TEST_F(conv_depthwise2d_test, test_nullptr_error) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 16};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(
      aclnnConvDepthwise2d,  // host api第二段接口名称
      INPUT(nullptr, nullptr, kernelSize_desc, bias_desc, nullptr, nullptr, nullptr),  // host api输入
      OUTPUT(nullptr), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR); // 平台切换后UT异常，临时下架
}

// Bias Nullptr
TEST_F(conv_depthwise2d_test, test_bias_nullptr) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 16};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, nullptr, strides_desc, padding_desc, dilation_desc),
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // // ut.TestPrecision();
}

// shape > 0  input/weight
// TEST_F(conv_depthwise2d_test, test_shape_error_1) {
//   vector<int64_t> inp_dims = {-2, 16, 32, 16};
//   vector<int64_t> weight_dims = {16, 1, -3, -3};
//   vector<int64_t> kernelSize_dims = {3, 3};
//   vector<int64_t> bias_dims = {16};
//   vector<int64_t> strides_dims = {1, 1};
//   vector<int64_t> padding_dims = {1, 1};
//   vector<int64_t> dilation_dims = {1, 1};
//   vector<int64_t> output_dims = {2, 16, 32, 16};

//   auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
//   auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
//   auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
//   auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
//   auto strides_desc = IntArrayDesc(strides_dims);
//   auto padding_desc = IntArrayDesc(padding_dims);
//   auto dilation_desc = IntArrayDesc(dilation_dims);
//   auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
//                       INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
//                             dilation_desc),  // host api输入
//                       OUTPUT(output_desc), cubeMathType);

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
//   EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// shape > 0 stride/padding/dilation
TEST_F(conv_depthwise2d_test, test_shape_error_2) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {-1, 0};
  vector<int64_t> padding_dims = {-1, 1};
  vector<int64_t> dilation_dims = {-1, 0};
  vector<int64_t> output_dims = {2, 16, 32, 16};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// conv mode image size after padding should be greater than or equal to filter size.
TEST_F(conv_depthwise2d_test, test_shape_error_3) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 48, 48};
  vector<int64_t> kernelSize_dims = {48, 48};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bias shape
TEST_F(conv_depthwise2d_test, test_bias_shape) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {1, 16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bias shape
TEST_F(conv_depthwise2d_test, test_bias_equal_outChannel) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {8};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// weight 1st dim shoud be should be divisible by input channel
TEST_F(conv_depthwise2d_test, test_weight_1stDim) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {8, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 16};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x channel should be equal to filter channel * groups
TEST_F(conv_depthwise2d_test, test_channel_group) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 2, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_depthwise2d_test, test_empty_tensor_Batch) {
  vector<int64_t> inp_dims = {0, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {0, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(conv_depthwise2d_test, test_empty_tensor_Channel) {
  vector<int64_t> inp_dims = {2, 0, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 0, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// no contiguous
TEST_F(conv_depthwise2d_test, test_FP16_no_contiguous) {
  vector<int64_t> inp_dims = {2, 2, 16, 32};
  vector<int64_t> weight_dims = {2, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {2};
  vector<int64_t> output_dims = {2, 2, 16, 32};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW, {1, 1, 1, 16}, 0, {2, 2, 32, 16}).ValueRange(0, 2);
  auto weight_desc =
      TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW, {1, 1, 1, 3}, 0, {2, 1, 3, 3}).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// weight spacial dim should be larger than 0
TEST_F(conv_depthwise2d_test, test_weight_spatial_dim) {
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 0, 3};
  vector<int64_t> kernelSize_dims = {0, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                            dilation_desc),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(conv_depthwise2d_test, ascend910B2_test_1971) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> kernelSize_dims = {3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto kernelSize_desc = IntArrayDesc(kernelSize_dims);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvDepthwise2d,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, kernelSize_desc, bias_desc, strides_desc, padding_desc,
                              dilation_desc),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}
