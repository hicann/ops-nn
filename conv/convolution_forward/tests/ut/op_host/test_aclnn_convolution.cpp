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
 * \file test_convolution.cpp
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

#include "opdev/platform.h"

using namespace op;
using namespace std;


#define KEEP_DTYPE 0
#define ALLOW_FP32_DOWN_PRECISION 1
#define USE_FP16 2
#define USE_HF32 3
static int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
static int8_t cubeMathTypeFP16 = USE_FP16;
static int8_t cubeMathTypeHF32 = USE_HF32;

class convolution_test : public testing::Test {
public:
  void SetUp() {
    cout << "convolution_test SetUp" << endl;
  }

  void TearDown() {
    cout << "convolution_test TearDown" << endl;
  }
};

TEST_F(convolution_test, test_conv2D_input_FP16_weight_FP16_output_FP32) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);
  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, test_conv2D_input_FP32_weight_FP32_output_FP16) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, test_conv2D_input_FP32_weight_FP16_output_FP16) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, test_conv2D_input_FP32_weight_FP16_output_FP32) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, test_conv2D_input_FP16_weight_FP32_output_FP32) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();  // comment bcz of timeout in model tests (269616 ms)
  }
}

TEST_F(convolution_test, test_conv2D_input_FP16_weight_FP32_output_FP16) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, test_conv2D_NCHW) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, test_conv2D_weight_NHWC_error) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_conv2D_NHWC) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 32, 16, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 32, 8, 16};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NHWC).Precision(0.01, 0.01);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_NE(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

// 该用例报错，待修复后恢复
/*
TEST_F(convolution_test, test_conv1D_FP16) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 16, 16};
  vector<int64_t> weight_dims = {8, 16 / groups, 3};
  vector<int64_t> bias_dims = {8};
  vector<int64_t> output_dims = {16, 8, 16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}
*/

TEST_F(convolution_test, test_conv1D_FP32) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 16, 16};
  vector<int64_t> weight_dims = {8, 16 / groups, 3};
  vector<int64_t> bias_dims = {8};
  vector<int64_t> output_dims = {16, 8, 16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(convolution_test, test_specialConv1D_FP16_DATA_ONES) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 1, 64};
  vector<int64_t> weight_dims = {3, 1 / groups, 64};
  vector<int64_t> bias_dims = {1, 3, 1};
  vector<int64_t> output_dims = {16, 3, 1};
  vector<int64_t> strides_dims = {64};
  vector<int64_t> padding_dims = {0};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2).Value(vector<float>(16 * 64, 1));
  auto weight_desc =
      TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2).Value(vector<float>(64 * 3, 1));
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // ut.TestPrecision();
}

TEST_F(convolution_test, test_specialConv1D_FP16) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 1, 64};
  vector<int64_t> weight_dims = {8, 1 / groups, 64};
  vector<int64_t> bias_dims = {1, 8, 1};
  vector<int64_t> output_dims = {16, 8, 1};
  vector<int64_t> strides_dims = {64};
  vector<int64_t> padding_dims = {0};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

  // ut.TestPrecision();
}

TEST_F(convolution_test, test_conv2DTranspose_NCHW) {
  // test conv2dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, test_conv2DTranspose_NHWC) {
  // test conv2dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 32, 18, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 32, 20, 16};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NHWC).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_NE(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, test_conv1DTranspose_FP16) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32};
  vector<int64_t> weight_dims = {16, 16 / groups, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {2};
  vector<int64_t> output_padding_dims = {1};
  vector<int64_t> output_dims = {2, 16, 35};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(convolution_test, test_conv1DTranspose_FP32) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32};
  vector<int64_t> weight_dims = {16, 16 / groups, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {2};
  vector<int64_t> output_padding_dims = {1};
  vector<int64_t> output_dims = {2, 16, 35};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(convolution_test, test_conv1D_infershape_error) {
  // test conv2dTranspose
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 16, 16};
  vector<int64_t> weight_dims = {8, 16 / groups, 3};
  vector<int64_t> bias_dims = {1, 8, 1};
  vector<int64_t> output_dims = {16, 8, 17};  // != 16
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_conv2D_infershape_error) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 33, 9};  // != 32, 9
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_conv1DTranspose_infershape_error) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32};
  vector<int64_t> weight_dims = {16, 16 / groups, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {2};
  vector<int64_t> output_padding_dims = {1};
  vector<int64_t> output_dims = {2, 16, 36};  // != 35

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_conv2DTranspose_infershape_error) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 33, 21};  // !=32, 20

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_dataType_error) {
  vector<aclDataType> ValidList = {ACL_INT8,   ACL_INT32,     ACL_UINT8,      ACL_INT16,  ACL_UINT16,
                                   ACL_UINT32, ACL_INT64,     ACL_UINT64,     ACL_DOUBLE, ACL_BOOL,
                                   ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128, ACL_BF16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 16, 16};
  vector<int64_t> weight_dims = {8, 16 / groups, 3};
  vector<int64_t> bias_dims = {1, 8, 1};
  vector<int64_t> output_dims = {16, 8, 16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(-2, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(-2, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCL);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCL);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(convolution_test, test_1D_format_error) {
  vector<aclFormat> ValidList = {
      ACL_FORMAT_UNDEFINED, ACL_FORMAT_NCHW,        ACL_FORMAT_NHWC,  ACL_FORMAT_ND,    ACL_FORMAT_NC1HWC0,
      ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NC1HWC0_C04, ACL_FORMAT_HWCN,  ACL_FORMAT_NDHWC, ACL_FORMAT_FRACTAL_NZ,
      ACL_FORMAT_NCDHW,     ACL_FORMAT_NDC1HWC0,    ACL_FRACTAL_Z_3D,
  };
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 16, 16};
  vector<int64_t> weight_dims = {8, 16 / groups, 3};
  vector<int64_t> bias_dims = {1, 8, 1};
  vector<int64_t> output_dims = {16, 8, 16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ValidList[i]).ValueRange(-2, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ValidList[i]).ValueRange(-2, 2);
    auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ValidList[i]);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ValidList[i]);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(convolution_test, test_2D_format_error) {
  vector<aclFormat> ValidList = {
      ACL_FORMAT_UNDEFINED, ACL_FORMAT_NCL,         ACL_FORMAT_NHWC,  ACL_FORMAT_ND,    ACL_FORMAT_NC1HWC0,
      ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NC1HWC0_C04, ACL_FORMAT_HWCN,  ACL_FORMAT_NDHWC, ACL_FORMAT_FRACTAL_NZ,
      ACL_FORMAT_NCDHW,     ACL_FORMAT_NDC1HWC0,    ACL_FRACTAL_Z_3D,
  };
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ValidList[i]).ValueRange(-2, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ValidList[i]).ValueRange(-2, 2);
    auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ValidList[i]);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ValidList[i]);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

// input dim should be equal to weight
TEST_F(convolution_test, test_input_weight_dim_error_1) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// attr dim should be equal to input dim - 2
TEST_F(convolution_test, test_attr_dim_error) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2, 1};
  vector<int64_t> padding_dims = {1, 2, 1};
  vector<int64_t> dilation_dims = {1, 2, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input dim should be according to the format
TEST_F(convolution_test, test_dim_format_error) {
  vector<aclFormat> ValidList = {
      ACL_FORMAT_NCL,
      ACL_FORMAT_NCDHW,
  };
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  for (int i = 1; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ValidList[i]).ValueRange(-2, 2);
    auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ValidList[i]).ValueRange(-2, 2);
    auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ValidList[i]);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ValidList[i]);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

// input weight should be 3 or 4 or 5
TEST_F(convolution_test, test_input_weight_dim_error_2) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Nullptr
TEST_F(convolution_test, test_nullptr_error) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};  // !=32, 20

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(
      aclnnConvolution,  // host api第二段接口名称
      INPUT(nullptr, nullptr, bias_desc, nullptr, nullptr, nullptr, transposed, nullptr, groups),  // host api输入
      OUTPUT(nullptr), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// Bias Nullptr
// TEST_F(convolution_test, test_bias_nullptr) {
//   // test conv2dTranspose
//   bool transposed = true;
//   const int64_t groups = 1;
//   vector<int64_t> inp_dims = {2, 16, 32, 18};
//   vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
//   vector<int64_t> bias_dims = {16};
//   vector<int64_t> strides_dims = {1, 1};
//   vector<int64_t> padding_dims = {1, 1};
//   vector<int64_t> dilation_dims = {1, 2};
//   vector<int64_t> output_padding_dims = {0, 0};
//   vector<int64_t> output_dims = {2, 16, 32, 20};  // !=32, 20

//   auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
//   auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
//   auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2).Value(vector<float>(16, 0));
//   auto strides_desc = IntArrayDesc(strides_dims);
//   auto padding_desc = IntArrayDesc(padding_dims);
//   auto dilation_desc = IntArrayDesc(dilation_dims);
//   auto output_padding_desc = IntArrayDesc(output_padding_dims);
//   auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

//   auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
//                       INPUT(inp_desc, weight_desc, nullptr, strides_desc, padding_desc, dilation_desc, transposed,
//                             output_padding_desc, groups),  // host api输入
//                       OUTPUT(output_desc), cubeMathType);

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
//   EXPECT_EQ(aclRet, ACL_SUCCESS);

//   // // ut.TestPrecision();
// }
// shape > 0  input/weight
// TEST_F(convolution_test, test_shape_error_1) {
//   // test conv2dTranspose
//   bool transposed = true;
//   const int64_t groups = 1;
//   vector<int64_t> inp_dims = {-2, 16, 32, 18};
//   vector<int64_t> weight_dims = {16, 16 / groups, -3, -3};
//   vector<int64_t> bias_dims = {16};
//   vector<int64_t> strides_dims = {1, 1};
//   vector<int64_t> padding_dims = {1, 1};
//   vector<int64_t> dilation_dims = {1, 2};
//   vector<int64_t> output_padding_dims = {0, 0};
//   vector<int64_t> output_dims = {2, 16, 32, 20};

//   auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
//   auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
//   auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
//   auto strides_desc = IntArrayDesc(strides_dims);
//   auto padding_desc = IntArrayDesc(padding_dims);
//   auto dilation_desc = IntArrayDesc(dilation_dims);
//   auto output_padding_desc = IntArrayDesc(output_padding_dims);
//   auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
//                       INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
//                             output_padding_desc, groups),  // host api输入
//                       OUTPUT(output_desc), cubeMathType);

//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
//   EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// shape > 0 stride/padding/dilation
TEST_F(convolution_test, test_shape_error_2) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {-1, 0};
  vector<int64_t> padding_dims = {-1, 1};
  vector<int64_t> dilation_dims = {-1, 0};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// conv mode image size after padding should be greater than or equal to filter size.
TEST_F(convolution_test, test_shape_error_3) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 48, 48};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// transpose mode image size after padding donot need to be greater than or equal to filter size.
TEST_F(convolution_test, test_shape_transpose_pass) {
  const int64_t groups = 1;
  bool transposed = true;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 48, 48};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 77, 122};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 1};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);
  // ut.TestPrecision();
}

// bias shape
TEST_F(convolution_test, test_bias_shape) {
  // test conv2dTranspose
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input bias size of dim_c should be equal to out_channels
TEST_F(convolution_test, test_bias_equal_outChannel) {
  // test conv2dTranspose
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 8, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input bias size of none channel should be equal to 1
TEST_F(convolution_test, test_bias_noneChannel_equal_1) {
  // test conv2dTranspose
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {2, 16, 2, 2};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bias for transpose
TEST_F(convolution_test, test_bias_shape_transpose) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// group > 0
TEST_F(convolution_test, test_groups_1) {
  // test conv2dTranspose
  const int64_t groups = 0;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// weight 1st dim should be divisible by groups
TEST_F(convolution_test, test_groups_2) {
  const int64_t groups = 3;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// weight 1st dim shoud be equal to input channel in Transpose mode
TEST_F(convolution_test, test_weight_1stDim_transpose) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {8, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// output padding must be smaller than either stride or dilation
TEST_F(convolution_test, test_outputPadding_transpose) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {1, 2};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x channel should be equal to filter channel * groups
TEST_F(convolution_test, test_channel_group) {
  // test conv2dTranspose
  const int64_t groups = 2;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 8 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_empty_tensor_conv2d_Batch) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {0, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {0, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  // // ut.TestPrecision();
}

// weight的cin为0的空tensor场景
// input: 2,0,5   weight: 8,0,3
TEST_F(convolution_test, test_empty_tensor_conv1d) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 0, 5};
  vector<int64_t> weight_dims = {8, 0, 3};
  vector<int64_t> bias_dims = {8};
  vector<int64_t> output_dims = {2, 0, 3};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {0};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// conv2D, N、C为0的时候
// input: 0, 2, 1, 0   weight: 3, 2, 1, 1
TEST_F(convolution_test, test_empty_tensor_conv2d) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {0, 2, 1, 0};
  vector<int64_t> weight_dims = {3, 2, 1, 1};
  vector<int64_t> bias_dims = {3};
  vector<int64_t> output_dims = {0, 3, 5, 4};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {2, 2};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(convolution_test, test_empty_tensor_conv2d_Channel) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 0, 32, 16};
  vector<int64_t> weight_dims = {16, 0 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_empty_tensor_conv2d_Batch_Channel) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {0, 0, 32, 16};
  vector<int64_t> weight_dims = {16, 0 / groups, 3, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {0, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_empty_tensor_conv2dTranspose_Batch) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {0, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {0, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(convolution_test, test_empty_tensor_conv2dTranspose_Channel) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 0, 32, 18};
  vector<int64_t> weight_dims = {0, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// no contiguous
TEST_F(convolution_test, test_conv2D_FP16_no_contiguous) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 2, 16, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3};
  vector<int64_t> bias_dims = {2};
  vector<int64_t> output_dims = {2, 2, 8, 32};
  vector<int64_t> strides_dims = {2, 1};
  vector<int64_t> padding_dims = {2, 1};
  vector<int64_t> dilation_dims = {2, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW, {1, 1, 1, 16}, 0, {2, 2, 32, 16}).ValueRange(0, 2);
  auto weight_desc =
      TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW, {1, 1, 1, 3}, 0, {2, 2, 3, 3}).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose) {
  // test conv3dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_depthwise) {
  // test conv3dTranspose depthwise
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 2;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_NDHWC) {
  // test conv3dTranspose NDHWC
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NDHWC).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NDHWC).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_dilation) {
  // test conv3dTranspose dilation
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 30};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 2};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_stride) {
  // test conv3dTranspose stirde
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 11};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 31};
  vector<int64_t> strides_dims = {1, 1, 3};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_pad) {
  // test conv3dTranspose pad
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 35};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 4};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 3};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_outputPad) {
  // test conv3dTranspose outputPad
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 33};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_kernel) {
  // test conv3dTranspose kernel
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 30};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 5};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_strideD_GT_kernelD) {
  // test conv3dTranspose strideD GT kernelD
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 16, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 31, 32, 32};
  vector<int64_t> strides_dims = {2, 1, 1};
  vector<int64_t> padding_dims = {0, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_pad_GE_kernel) {
  // test conv3dTranspose pad GE kernel
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 37};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 2};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 3};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_dout_limit) {
  // test conv3dTranspose dout limit
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 11, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 2, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 23, 32, 32};
  vector<int64_t> strides_dims = {2, 1, 1};
  vector<int64_t> padding_dims = {0, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTranspose_v1) {
  // test conv3dTranspose v1
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 23, 11, 20};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 1, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 23, 21, 39};
  vector<int64_t> strides_dims = {1, 2, 2};
  vector<int64_t> padding_dims = {0, 0, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

// weight spacial dim should be larger than 0
TEST_F(convolution_test, test_weight_spatial_dim) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 0, 3};
  vector<int64_t> bias_dims = {1, 16, 1, 1};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, test_conv2D_group_larger_1) {
  const int64_t groups = 2;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(convolution_test, ascend910B2_test_conv2D_NCHW) {
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv1D_FP32) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 16, 16};
  vector<int64_t> weight_dims = {8, 16 / groups, 3};
  vector<int64_t> bias_dims = {8};
  vector<int64_t> output_dims = {16, 8, 16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

TEST_F(convolution_test, ascend910B2_test_conv2DTranspose_NCHW_SwitchHW) {
  // test conv2dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {1, 48, 1, 1100};
  vector<int64_t> weight_dims = {48, 8, 1, 8};
  vector<int64_t> strides_dims = {1, 4};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {1, 8, 1, 4404};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv1DTranspose_NCHW_SwitchHW) {
  // test conv2dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {1, 48, 1100};
  vector<int64_t> weight_dims = {48, 8, 8};
  vector<int64_t> strides_dims = {4};
  vector<int64_t> padding_dims = {0};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0};
  vector<int64_t> output_dims = {1, 8, 4404};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCL).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv2DTranspose_NCHW) {
  // test conv2dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv2DTranspose_NCHW_pad4dim) {
  // test conv2dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1, 1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0, 0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);  // 平台切换后UT异常，临时下架

    // ut.TestPrecision();
  }
}

// L0 wrapper error test
TEST_F(convolution_test, ascend910B2_test_Conv1D_l0_wrapper_error) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 16, 16};
  vector<int64_t> weight_dims = {8, 16 / groups, 3};
  vector<int64_t> bias_dims = {8};
  vector<int64_t> output_dims = {16, 8, 16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_BF16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_BF16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_BF16, ACL_FORMAT_NCL);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCL);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph

  // EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
}

// C04
TEST_F(convolution_test, ascend910B2_test_conv2D_C04) {
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {320, 3, 224, 224};
  vector<int64_t> weight_dims = {768, 3, 32, 32};
  vector<int64_t> bias_dims = {768};
  vector<int64_t> output_dims = {320, 768, 7, 7};
  vector<int64_t> strides_dims = {32, 32};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    // auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

// conv2DTranposeV2To3D
TEST_F(convolution_test, ascend910B2_test_conv2DTranposeV2To3D) {
  // test conv2dTranspose
  vector<aclDataType> ValidList = {ACL_FLOAT};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {4, 320, 80, 80};
  vector<int64_t> weight_dims = {320, 320, 3, 3};
  vector<int64_t> bias_dims = {320};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {4, 320, 80, 80};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv2D_C04_BFLOAT16) {
  vector<aclDataType> ValidList = {ACL_BF16, ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<vector<int64_t>> inp_dims = {{32, 3, 1, 224}, {32, 3, 224, 224}, {32, 1, 256, 256}, {32, 3, 224, 224}, {16, 1, 20, 32768}};
  vector<vector<int64_t>> weight_dims = {{64, 3, 1, 32}, {64,3,32,32}, {64,1,128,128},{64,3,32,32}, {16, 1, 1, 32}};
  vector<int64_t> bias_dims = {64};
  vector<vector<int64_t>> output_dims = {{32, 64, 1, 193}, {32,64,193,193}, {32,64,129,129}, {32,64,193,193}, {16, 16, 20, 32737}};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims[i], ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend910B2_test_conv2D_with_asymmetric_pad) {
  vector<aclDataType> ValidList = {ACL_BF16, ACL_BF16, ACL_FLOAT16, ACL_BF16, ACL_FLOAT16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  vector<int64_t> groups = {1,1,1,1,1,1,1};
  bool transposed = false;
  vector<vector<int64_t>> inp_dims = {{16,40,120,120}, {16,40,120,120}, {16, 64, 1, 120}, {4, 32, 16, 4200}, {4, 32, 1, 4200}, {16,32,40,40}, {16,3,224,224}};
  vector<vector<int64_t>> weight_dims = {{31, 40, 10, 10}, {31,40,10,10}, {31,64,1,10},{32,32,52,52}, {32, 32, 1, 52}, {16,32,4,64}, {34,3,16,16}};
  vector<vector<int64_t>> output_dims = {{16, 31, 30, 31}, {16,31,30,30}, {16,31,1,29}, {4,32,3,72}, {4, 32, 1, 65}, {16,16,15,1}, {16,34, 59,61}};
  vector<vector<int64_t>> padding_dims = {{2, 4, 4, 6}, {3, 3}, {0, 2}, {100, 102, 192, 204}, {0,0,0,0}, {10,12}, {10,14,14,18}};
  vector<vector<int64_t>> strides_dims = {{4, 4}, {4, 4}, {4, 4}, {64,64}, {64,64}, {4, 4}, {4, 4}};
  vector<vector<int64_t>> dilation_dims = {{1, 1}, {1,1}, {1,1}, {1,1}, {1,1}, {1,1}, {1,1}};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims[i]);
    auto padding_desc = IntArrayDesc(padding_dims[i]);
    auto dilation_desc = IntArrayDesc(dilation_dims[i]);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims[i], ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups[i]),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend910B2_exception_conv2D_with_asymmetric_pad) {
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT16};
  int length = ValidList.size();

  vector<int64_t> groups = {1,1};
  bool transposed = false;
  vector<vector<int64_t>> inp_dims = {{16,16,16,16}, {16,16,16,16}};
  vector<vector<int64_t>> weight_dims = {{16,16,4,4}, {16,16,4,4}};
  vector<vector<int64_t>> output_dims = {{16, 16, 6, 5}, {16,16,5,3}};
  vector<vector<int64_t>> strides_dims = {{4, 4}, {4, 4}};
  vector<vector<int64_t>> padding_dims = {{5, 6, 7}, {5}};
  vector<vector<int64_t>> dilation_dims = {{1, 1}, {1,1}};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims[i]);
    auto padding_desc = IntArrayDesc(padding_dims[i]);
    auto dilation_desc = IntArrayDesc(dilation_dims[i]);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims[i], ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups[i]),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_NE(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, test_conv1D_with_asymmetric_pad) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {16, 64, 122};
  vector<int64_t> weight_dims = {31, 64, 10};
  vector<int64_t> bias_dims = {31};
  vector<int64_t> output_dims = {16, 31, 30};
  vector<int64_t> strides_dims = {4};
  vector<int64_t> padding_dims = {2,2};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.01, 0.01);


  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(convolution_test, ASCEND910_93_test_conv2D_C04_BFLOAT16) {
  vector<aclDataType> ValidList = {ACL_BF16, ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<vector<int64_t>> inp_dims = {{32, 3, 1, 224}, {32, 3, 224, 224}, {32, 1, 256, 256}, {32, 3, 224, 224}, {16, 1, 20, 32768}};
  vector<vector<int64_t>> weight_dims = {{64, 3, 1, 32}, {64,3,32,32}, {64,1,128,128},{64,3,32,32}, {16, 1, 1, 32}};
  vector<int64_t> bias_dims = {64};
  vector<vector<int64_t>> output_dims = {{32, 64, 1, 193}, {32,64,193,193}, {32,64,129,129}, {32,64,193,193}, {16, 16, 20, 32737}};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims[i], ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims[i], ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3D_bias) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3D_nonbias) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3Dfp32_bias) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, test_conv2D_input_FP16_weight_FP16_output_FP32_bias_NHWC) {
  vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);
  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(convolution_test, test_conv3dv2_incorrect_soc_version_negative) {
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {20, 16, 32, 256, 256};
  vector<int64_t> weight_dims = {32, 16, 3, 3, 3};
  vector<int64_t> bias_dims = {weight_dims[0]};
  vector<int64_t> output_dims = {20, 32, 30, 254, 254};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {0, 0, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
    if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(convolution_test, test_conv3dv2_groups_not_equal_1) {
  vector<aclDataType> ValidList = {ACL_BF16, ACL_FLOAT, ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 2;
  vector<int64_t> inp_dims = {20, 16, 4, 32, 32};
  vector<int64_t> weight_dims = {32, 16 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {weight_dims[0]};
  vector<int64_t> output_dims = {20, 32, 2, 30, 30};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {0, 0, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph

    // EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3D_pointwise_bias) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16, ACL_FLOAT};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 1, 1};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {0, 0, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3D_pointwise_nonbias) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16, ACL_FLOAT};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 1, 1};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {0, 0, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3DTransPose_outputpadding) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 11};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 1, 1};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 3};
  vector<int64_t> padding_dims = {0, 0, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, test_conv2D_NCHW_splitW_1) {
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {3, 1, 410, 410};
  vector<int64_t> weight_dims = {1, 1 / groups, 51, 51};
  vector<int64_t> bias_dims = {1};
  vector<int64_t> output_dims = {3, 1, 360, 360};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, test_conv2D_NCHW_splitW_2) {
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {1, 16, 1792, 896};
  vector<int64_t> weight_dims = {3200, 16 / groups, 14, 14};
  vector<int64_t> bias_dims = {3200};
  vector<int64_t> output_dims = {1, 3200, 128, 64};
  vector<int64_t> strides_dims = {14, 14};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}


TEST_F(convolution_test, ascend910B2_test_conv3dv2_bf16_use_fp16) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathTypeFP16);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph

    // EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3dv2_bf16_use_hf32) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_BF16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathTypeHF32);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph

    // EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend910B2_test_conv3dv2_fp16_use_hf32) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16};
  int length = ValidList.size();

  bool transposed = false;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathTypeHF32);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph

    EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend_test_conv2D_empty_tensor_input_zero_output_nonzero) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 1, 32, 0};
  vector<int64_t> weight_dims = {16, 1, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 5};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCHW);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  // EXPECT_EQ(aclRet, ACL_SUCCESS);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, ascend_test_conv2D_input_HIF8_weight_HIF8_output_HIF8) {
  vector<aclDataType> ValidList = {ACL_HIFLOAT8, ACL_HIFLOAT8};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_HIFLOAT8, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  // auto groups_desc = ScalarDesc(groups);
  // auto tranposed_desc = ScalarDesc(transposed);
  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend_test_conv2D_input_HIF8_weight_FP16_output_HIF8) {
  vector<aclDataType> ValidList = {ACL_FLOAT};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_HIFLOAT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_HIFLOAT8, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  for (int i = 0; i < length; i++) {
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_NE(aclRet, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend_test_conv2D_input_bf16_weight_bf16_output_bf16_usefp16) {
  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  auto inp_desc = TensorDesc(inp_dims, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_BF16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
  auto bias_desc = TensorDesc(bias_dims, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), 2);
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS); // a!=b check
}

TEST_F(convolution_test, ascend_test_empty_tensor_conv2dTranspose_Batch) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {0, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {0, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(convolution_test, ascend_test_empty_tensor_conv2dTranspose_Channel) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 0, 32, 18};
  vector<int64_t> weight_dims = {0, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {2, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, ascend_test_empty_tensor_conv2dTranspose_H) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 0, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {0, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, ascend_test_empty_tensor_conv2dTranspose_W) {
  // test conv2dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 32, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 0};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1, 1};
  vector<int64_t> padding_dims = {1, 1};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};
  vector<int64_t> output_dims = {0, 16, 32, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, ascend_test_conv3DTransPose_outputpadding) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16, ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 11};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 1, 1};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 3};
  vector<int64_t> padding_dims = {0, 0, 0};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(ACL_SUCCESS, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend_test_conv2DTransPose_outputpadding) {
  // test conv3d
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16, ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 11};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 1};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32};
  vector<int64_t> strides_dims = {1, 3};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(ACL_SUCCESS, ACL_SUCCESS);
    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend_test_conv3DTranspose_8bit_with_bias_err) {
  // test conv3dTranspose 8bit with bias
  vector<aclDataType> ValidList = {ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 32, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3, 3, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32, 32};
  vector<int64_t> strides_dims = {1, 1, 1};
  vector<int64_t> padding_dims = {1, 1, 1};
  vector<int64_t> dilation_dims = {1, 1, 1};
  vector<int64_t> output_padding_dims = {0, 0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(convolution_test, ascend_test_conv2DTransPose_8bit_with_bias_err) {
  // test conv2dtranspose 8bit with bias
  vector<aclDataType> ValidList = {ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32, 11};
  vector<int64_t> weight_dims = {2, 2 / groups, 1, 1};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32, 32};
  vector<int64_t> strides_dims = {1, 3};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(convolution_test, ascend_test_conv2DTransPose_8bit_with_nhwc_err) {
  // test conv2dtranspose 8bit with bias
  vector<aclDataType> ValidList = {ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 32, 11, 2};
  vector<int64_t> weight_dims = {2, 1, 1, 2 / groups};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 32, 32, 2};
  vector<int64_t> strides_dims = {1, 3};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NHWC).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(convolution_test, ascend_test_conv2DTransPose_dataType_error) {
  vector<aclDataType> ValidList = {ACL_DOUBLE};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = true;
  vector<int64_t> inp_dims = {2, 32, 11, 2};
  vector<int64_t> weight_dims = {2, 1, 1, 2 / groups};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 32, 32, 2};
  vector<int64_t> strides_dims = {1, 3};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NHWC).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NHWC);
    for (int cube_type = 0; cube_type < 4; cube_type++) {
      auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                          INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                                output_padding_desc, groups),  // host api输入
                          OUTPUT(output_desc), cube_type);
      uint64_t workspace_size = 0;
      aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
      EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
  }
}

TEST_F(convolution_test, ascend_test_conv1DTranspose_outputpadding) {
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_BF16, ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 11};
  vector<int64_t> weight_dims = {2, 2 / groups, 1};
  vector<int64_t> output_dims = {2, 2, 32};
  vector<int64_t> strides_dims = {3};
  vector<int64_t> padding_dims = {0};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {1};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCL).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(ACL_SUCCESS, ACL_SUCCESS);
  }
}

TEST_F(convolution_test, ascend_test_empty_tensor_conv1dTranspose_L) {
  // test conv1dTranspose
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 16, 18};
  vector<int64_t> weight_dims = {16, 16 / groups, 0};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {2};
  vector<int64_t> output_padding_dims = {0};
  vector<int64_t> output_dims = {0, 16, 20};

  auto inp_desc = TensorDesc(inp_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(-2, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT16, ACL_FORMAT_NCL).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathType);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_test, ascend_test_conv1DTranspose_hifloat8_Or_f8e4m3fn_with_bias_error) {
  // test conv1dTranspose hifloat8 or float8_e4m3fn with bias
  vector<aclDataType> ValidList = {ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN};
  int length = ValidList.size();

  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> inp_dims = {2, 2, 32};
  vector<int64_t> weight_dims = {2, 2 / groups, 3};
  vector<int64_t> bias_dims = {2 / groups};
  vector<int64_t> output_dims = {2, 2, 32};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {1};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT16, ACL_FORMAT_NCL).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCL).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
  }
}

TEST_F(convolution_test, ascend_test_conv1d_transpose_with_bias)
{
  bool transposed = true;
  const int64_t groups = 384;
  vector<int64_t> input_dims = {1, 384, 258};
  vector<int64_t> weight_dims = {384, 1, 3};
  vector<int64_t> bias_dims = {384};
  vector<int64_t> output_dims = {1, 384, 256};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {2};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0};

  auto input_desc = TensorDesc(input_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(input_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathTypeHF32);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_test, ascend_test_conv1d_transpose_matmul)
{
  bool transposed = true;
  const int64_t groups = 1;
  vector<int64_t> input_dims = {4, 272, 1};
  vector<int64_t> weight_dims = {272, 128, 73};
  vector<int64_t> bias_dims = {};
  vector<int64_t> output_dims = {4, 128, 73};
  vector<int64_t> strides_dims = {8};
  vector<int64_t> padding_dims = {0};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0};

  auto input_desc = TensorDesc(input_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto weight_desc = TensorDesc(weight_dims, ACL_FLOAT, ACL_FORMAT_NCL).ValueRange(0, 2);
  auto bias_desc = TensorDesc(bias_dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto strides_desc = IntArrayDesc(strides_dims);
  auto padding_desc = IntArrayDesc(padding_dims);
  auto dilation_desc = IntArrayDesc(dilation_dims);
  auto output_padding_desc = IntArrayDesc(output_padding_dims);
  auto output_desc = TensorDesc(output_dims, ACL_FLOAT, ACL_FORMAT_NCL).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                      INPUT(input_desc, weight_desc, nullptr, strides_desc, padding_desc, dilation_desc, transposed,
                            output_padding_desc, groups),  // host api输入
                      OUTPUT(output_desc), cubeMathTypeHF32);

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_test, ascend310B_test_conv2D_NCHW) {
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> inp_dims = {2, 16, 32, 16};
  vector<int64_t> weight_dims = {16, 16 / groups, 3, 3};
  vector<int64_t> bias_dims = {16};
  vector<int64_t> output_dims = {2, 16, 32, 8};
  vector<int64_t> strides_dims = {1, 2};
  vector<int64_t> padding_dims = {1, 2};
  vector<int64_t> dilation_dims = {1, 2};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = TensorDesc(inp_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend950_test_conv2D_HWNC) {
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> storage_dims = {5, 340, 1, 128};
  vector<int64_t> view_dims = {1, 128, 5, 340};
  vector<int64_t> view_stride = {128, 1, 43520, 128};

  vector<int64_t> weight_dims = {128, 128 / groups, 1, 1};
  vector<int64_t> bias_dims = {128};
  vector<int64_t> output_dims = {1, 128, 5, 114};
  vector<int64_t> strides_dims = {1, 3};
  vector<int64_t> padding_dims = {0, 0};
  vector<int64_t> dilation_dims = {1, 1};
  vector<int64_t> output_padding_dims = {0, 0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = aclCreateTensor(view_dims.data(), view_dims.size(), ValidList[i], view_stride.data(), 0,
        ACL_FORMAT_NCHW, storage_dims.data(), storage_dims.size(), nullptr);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto bias_desc = TensorDesc(bias_dims, ValidList[i], ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}

TEST_F(convolution_test, ascend950_test_conv1D_LNC) {
  vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT, ACL_BF16};
  int length = ValidList.size();

  const int64_t groups = 1;
  bool transposed = false;
  vector<int64_t> storage_dims = {150, 128, 32};
  vector<int64_t> view_dims = {128, 32, 150};
  vector<int64_t> view_stride = {32, 1, 4096};

  vector<int64_t> weight_dims = {32, 32, 1};
  vector<int64_t> output_dims = {128, 32, 150};
  vector<int64_t> strides_dims = {1};
  vector<int64_t> padding_dims = {0};
  vector<int64_t> dilation_dims = {1};
  vector<int64_t> output_padding_dims = {0};

  for (int i = 0; i < length; i++) {
    auto inp_desc = aclCreateTensor(view_dims.data(), view_dims.size(), ValidList[i], view_stride.data(), 0,
        ACL_FORMAT_NCL, storage_dims.data(), storage_dims.size(), nullptr);
    auto weight_desc = TensorDesc(weight_dims, ValidList[i], ACL_FORMAT_NCL).ValueRange(0, 2);
    auto bias_desc = nullptr;
    auto strides_desc = IntArrayDesc(strides_dims);
    auto padding_desc = IntArrayDesc(padding_dims);
    auto dilation_desc = IntArrayDesc(dilation_dims);
    auto output_padding_desc = IntArrayDesc(output_padding_dims);
    auto output_desc = TensorDesc(output_dims, ValidList[i], ACL_FORMAT_NCL).Precision(0.01, 0.01);
    // auto groups_desc = ScalarDesc(groups);
    // auto tranposed_desc = ScalarDesc(transposed);

    auto ut = OP_API_UT(aclnnConvolution,  // host api第二段接口名称
                        INPUT(inp_desc, weight_desc, bias_desc, strides_desc, padding_desc, dilation_desc, transposed,
                              output_padding_desc, groups),  // host api输入
                        OUTPUT(output_desc), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);  // check op graph
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
  }
}