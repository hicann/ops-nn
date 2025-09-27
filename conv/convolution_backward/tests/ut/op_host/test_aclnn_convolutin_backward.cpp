/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "aclnn_convolution_backward.h"
#include "op_api/op_api_def.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
namespace{
class convolution_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "convolution_backward_test SetUp" << std::endl; }

  static void TearDownTestCase() { std::cout << "convolution_backward_test TearDown" << std::endl; }

  static void test_run(vector<int64_t> inputDims, aclDataType inputDtype, aclFormat inputFormat,
                       vector<int64_t> weightDims, aclDataType weightDtype, aclFormat weightFormat,
                       vector<int64_t> gradDims, aclDataType gradDtype, aclFormat gradFormat,
                       vector<int64_t> gradInputDims, aclDataType gradInputDtype, aclFormat gradInputFormat,
                       vector<int64_t> gradWeightDims, aclDataType gradWeightDtype, aclFormat gradWeightFormat,
                       vector<int64_t> gradBiasDims, aclDataType gradBiasDtype, aclFormat gradBiasFormat) {
    auto input_tensor_desc = TensorDesc(inputDims, inputDtype, inputFormat);
    auto weight_tensor_desc = TensorDesc(weightDims, weightDtype, weightFormat);
    auto grad_output_tensor_desc = TensorDesc(gradDims, gradDtype, gradFormat);
    auto gradInput = TensorDesc(gradInputDims, gradInputDtype, gradInputFormat);
    auto gradWeight = TensorDesc(gradWeightDims, gradWeightDtype, gradWeightFormat);
    auto gradBias = TensorDesc(gradBiasDims, gradBiasDtype, gradBiasFormat);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{gradDims[1]});
    auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  }
};

// test in ascend910

// test Conv2DBackward
TEST_F(convolution_backward_test, test_Conv2DBackward_Fp16_keep_dtype) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, test_Conv2DBackward_Fp16_keep_dtype_pad4dim) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 0, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, test_Conv2DBackward_Fp32_allow_fp32_down_precision) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 1;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, test_Conv2DBackward_Fp32_use_fp16) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 2;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// test Conv1DBackward
TEST_F(convolution_backward_test, test_Conv1DBackward_Fp16) {
  auto input_tensor_desc = TensorDesc({16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto weight_tensor_desc = TensorDesc({8, 16, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto grad_output_tensor_desc = TensorDesc({16, 8, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto gradWeight = TensorDesc({8, 16, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto gradBias = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// test empty
TEST_F(convolution_backward_test, test_ConvBackward_empty_error) {
  auto input_tensor_desc = TensorDesc({16, 16, 0, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({16, 16, 32, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 2});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 2});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 2});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test in ascend910B
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_Fp32_keep_dtype) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_Fp32_use_hf32) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 3;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// test dilation
TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_pre_dilation_FP16) {
  auto input_tensor_desc = TensorDesc({512, 512, 77, 77}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 21, 21}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{4, 4});
  auto padding_desc = IntArrayDesc(vector<int64_t>{4, 4});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 77, 77}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_pre_dilation_FP32) {
  auto input_tensor_desc = TensorDesc({512, 512, 77, 77}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 21, 21}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{4, 4});
  auto padding_desc = IntArrayDesc(vector<int64_t>{4, 4});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 77, 77}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_post_dilation_FP16) {
  auto input_tensor_desc = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_pre_dilation_hout1_wout1_FP16) {
  auto input_tensor_desc = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{6, 6});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_post_dilation_hout1_wout1_FP16) {
  auto input_tensor_desc = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{6, 6});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_post_dilation_FP32) {
  auto input_tensor_desc = TensorDesc({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 覆盖gradOutput、input、weight dtype不一致的情况
TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_dtype_inconsistent) {
  // FP32 + FP16 + FP16
  test_run({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW,
           {512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW,
           {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT16, ACL_FORMAT_ND);
  // FP16 + FP32 + FP16
  test_run({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW,
           {512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW,
           {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT16, ACL_FORMAT_ND);
  // FP16 + FP16 + FP32
  test_run({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW,
           {512, 2048, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW,
           {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT, ACL_FORMAT_ND);

  // FP32 + FP32 + FP16
  test_run({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW,
           {512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW,
           {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT16, ACL_FORMAT_ND);
  // FP16 + FP32 + FP32
  test_run({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW,
           {512, 2048, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW,
           {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT, ACL_FORMAT_ND);
  // FP32 + FP16 + FP32
  test_run({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW,
           {512, 2048, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW,
           {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT, ACL_FORMAT_ND);
}

// test Conv2DBackward BF16
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_Bf16) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_BF16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_BF16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 7, 7}, ACL_BF16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_BF16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_BF16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({512, 512, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{512});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({512, 512, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({512, 512, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({512, 512, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_attr_error) {
  auto input_tensor_desc = TensorDesc({1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{-1, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_FP16) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({512, 512, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{512});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({512, 512, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackwardInput_FP16_Conv2Matmul_1x1Kernel) {
  auto input_tensor_desc = TensorDesc({8, 64, 2, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({512, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({8, 512, 2, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{512});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
  auto gradInput = TensorDesc({8, 64, 2, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({512, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({512}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackwardInput_FP16_Conv2Matmul_Stride_Eq_Kernel) {
  auto input_tensor_desc = TensorDesc({1, 128, 18, 225, 225}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({256, 128, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 256, 6, 75, 75}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{256});
  auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3, 3});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
  auto gradInput = TensorDesc({1, 128, 18, 225, 225}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({256, 128, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_depthwise) {
  // test conv3dbackward depthwise
  auto input_tensor_desc = TensorDesc({128, 128, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({128, 1, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({128, 128, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{128});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 128;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({128, 128, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({128, 1, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({128}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_NDHWC) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NDHWC);
  auto weight_tensor_desc = TensorDesc({512, 512, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NDHWC);
  auto grad_output_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NDHWC);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{512});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NDHWC);
  auto gradWeight = TensorDesc({512, 512, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NDHWC);
  auto gradBias = TensorDesc({512}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_dilation) {
  auto input_tensor_desc = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 6, 6}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 2, 2});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_stride) {
  auto input_tensor_desc = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 4}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 8, 3}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 3});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 4}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_pad) {
  auto input_tensor_desc = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 4, 4}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 13}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 3});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 4, 4}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_kernel) {
  auto input_tensor_desc = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 5, 5}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 6, 6}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 5, 5}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_strideD_GT_kernelD) {
  auto input_tensor_desc = TensorDesc({1, 16, 11, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 6, 5, 5}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{2, 2, 2});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 11, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 2, 2}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_din_limit) {
  auto input_tensor_desc = TensorDesc({1, 16, 15, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 2, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 7, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{2, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 15, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 2, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_v1) {
  auto input_tensor_desc = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 1, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_v2) {
  auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_dw_bf16_out_fp32) {
  // test conv3dbackward depthwise
  auto input_tensor_desc = TensorDesc({128, 128, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({128, 1, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({128, 128, 7, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{128});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 128;
  auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
  auto gradInput = TensorDesc({128, 128, 7, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({128, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_dw_use_fp16_out_fp32) {
  // test conv3dbackward depthwise
  auto input_tensor_desc = TensorDesc({128, 128, 7, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({128, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({128, 128, 7, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{128});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 128;
  auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
  auto gradInput = TensorDesc({128, 128, 7, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({128, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);

  // USE_FP16
  int8_t cubeMathType = 2;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackpropInputV2_WhiteCase) {
  auto input_tensor_desc = TensorDesc({4, 320, 80, 80}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({320, 320, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({4, 320, 80, 80}, ACL_FLOAT, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{320});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
  auto gradInput = TensorDesc({4, 320, 80, 80}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({320, 320, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({320}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackpropFilterV3_WhiteCase) {
  auto input_tensor_desc = TensorDesc({1, 640, 104, 152}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({640, 640, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({1, 640, 104, 152}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{640});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
  auto gradInput = TensorDesc({1, 640, 104, 152}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({640, 640, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({640}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_USE_FP16) {
  auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 2; // USE_FP16

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend910_95_test_Conv3DBackward_Bf16_USE_FP16) {
  auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 2; // USE_FP16

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend910_95_test_Conv3DBackward_Bf16_USE_HF32) {
  auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 3; // USE_HF32

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_USE_HF32) {
  auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 3; // USE_HF32

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend910_95_test_Conv3DBackward_Fp16_USE_HF32) {
  auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 3; // USE_HF32

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Fp16_USE_HF32) {
  auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 3; // USE_HF32

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv2DBackward_Fp16_keep_dtype_pad4dim) {
  auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({512, 2048, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 0, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv1DBackward_Fp16_keep_dtype) {
  auto input_tensor_desc = TensorDesc({16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto weight_tensor_desc = TensorDesc({8, 16, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto grad_output_tensor_desc = TensorDesc({16, 8, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, false, true});
  auto gradInput = TensorDesc({16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto gradWeight = TensorDesc({8, 16, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
  auto gradBias = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv1DBackward_Fp32_keep_dtype) {
  auto input_tensor_desc = TensorDesc({16, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto weight_tensor_desc = TensorDesc({8, 16, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto grad_output_tensor_desc = TensorDesc({16, 8, 16}, ACL_FLOAT, ACL_FORMAT_NCL);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, false, true});
  auto gradInput = TensorDesc({16, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto gradWeight = TensorDesc({8, 16, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto gradBias = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv1DBackward_HF32) {
  auto input_tensor_desc = TensorDesc({16, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto weight_tensor_desc = TensorDesc({8, 16, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto grad_output_tensor_desc = TensorDesc({16, 8, 16}, ACL_FLOAT, ACL_FORMAT_NCL);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, false, true});
  auto gradInput = TensorDesc({16, 16, 16}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto gradWeight = TensorDesc({8, 16, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
  auto gradBias = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

  int8_t cubeMathType = 3;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv1DBackward_Bf16_keep_dtype) {
  auto input_tensor_desc = TensorDesc({4, 51, 21}, ACL_BF16, ACL_FORMAT_NCL);
  auto weight_tensor_desc = TensorDesc({8, 51, 2}, ACL_BF16, ACL_FORMAT_NCL);
  auto grad_output_tensor_desc = TensorDesc({4, 8, 20}, ACL_BF16, ACL_FORMAT_NCL);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{3});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, false, true});
  auto gradInput = TensorDesc({4, 51, 21}, ACL_BF16, ACL_FORMAT_NCL);
  auto gradWeight = TensorDesc({8, 51, 2}, ACL_BF16, ACL_FORMAT_NCL);
  auto gradBias = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv2DBackward_Fp16_keep_dtype_pad4dim_nhwc) {
  auto input_tensor_desc = TensorDesc({512, 7, 7, 512}, ACL_FLOAT16, ACL_FORMAT_NHWC);
  auto weight_tensor_desc = TensorDesc({2048, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_NHWC);
  auto grad_output_tensor_desc = TensorDesc({512, 6, 6, 2048}, ACL_FLOAT16, ACL_FORMAT_NHWC);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 0, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({512, 7, 7, 512}, ACL_FLOAT16, ACL_FORMAT_NHWC);
  auto gradWeight = TensorDesc({2048, 1, 1, 512}, ACL_FLOAT16, ACL_FORMAT_NHWC);
  auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv3DBackward_Fp16_USE_HF32_NDHWC) {
  auto input_tensor_desc = TensorDesc({1, 2, 10, 10, 16}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
  auto weight_tensor_desc = TensorDesc({16, 1, 3, 3, 16}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
  auto grad_output_tensor_desc = TensorDesc({1, 2, 10, 10, 16}, ACL_FLOAT16, ACL_FORMAT_NDHWC);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({1, 2, 10, 10, 16}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
  auto gradWeight = TensorDesc({16, 1, 3, 3, 16}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
  auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 3; // USE_HF32

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// test empty
TEST_F(convolution_backward_test, ascend910_95_test_ConvBackward_empty) {
  auto input_tensor_desc = TensorDesc({0, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({0, 16, 32, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 2});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 2});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 2});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({0, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// test empty
TEST_F(convolution_backward_test, ascend910_95_test_ConvBackward_transposed_empty) {
  auto input_tensor_desc = TensorDesc({0, 16, 32, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({0, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 2});
  auto padding_desc = IntArrayDesc(vector<int64_t>{1, 2});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 2});
  bool transposed = true;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({0, 16, 32, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv2DBackwardInput_FP16_Conv2Matmul_Stride_Eq_Kernel) {
  auto input_tensor_desc = TensorDesc({1, 128, 225, 225}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto weight_tensor_desc = TensorDesc({256, 128, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto grad_output_tensor_desc = TensorDesc({1, 256, 75, 75}, ACL_FLOAT16, ACL_FORMAT_NCHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{256});
  auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
  auto gradInput = TensorDesc({1, 128, 225, 225}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradWeight = TensorDesc({256, 128, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto gradBias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend910_95_test_Conv3DBackward_DwStrideEqualKernelTransToKernel1x1x1) {
  auto input_tensor_desc = TensorDesc({1, 3, 16, 224, 224}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({768, 3, 2, 16, 16}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({1, 768, 8, 14, 14}, ACL_BF16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{768});
  auto stride_desc = IntArrayDesc(vector<int64_t>{2, 16, 16});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = false;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
  auto gradInput = TensorDesc({1, 3, 16, 224, 224}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({768, 3, 2, 16, 16}, ACL_BF16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({768}, ACL_BF16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;
  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_ConvBackward_transpose_empty_error) {
  auto input_tensor_desc = TensorDesc({0, 1, 0, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto weight_tensor_desc = TensorDesc({1, 1, 0, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto grad_output_tensor_desc = TensorDesc({100, 100, 0, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

  auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
  auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
  bool transposed = true;
  auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
  int groups = 1;
  auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
  auto gradInput = TensorDesc({0, 1, 0, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradWeight = TensorDesc({1, 1, 0, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
  auto gradBias = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);

  int8_t cubeMathType = 0;

  auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                      padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                OUTPUT(gradInput, gradWeight, gradBias));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv1DBackpropFilter_fp16_matmul)
{
    auto input_tensor_desc = TensorDesc({4, 128,73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto weight_tensor_desc = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto grad_output_tensor_desc = TensorDesc({4, 272, 1}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{272});
    auto stride_desc = IntArrayDesc(vector<int64_t>{8});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({4, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradWeight = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradBias = TensorDesc({272}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(
            grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc, padding_desc,
            dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv1DBackpropInput_fp16_matmul)
{
    auto input_tensor_desc = TensorDesc({4, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto weight_tensor_desc = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto grad_output_tensor_desc = TensorDesc({4, 272, 1}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{272});
    auto stride_desc = IntArrayDesc(vector<int64_t>{8});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({4, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradWeight = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradBias = TensorDesc({272}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(
            grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc, padding_desc,
            dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910_95_test_Conv1DBackpropFilter_fp32_group)
{
    auto input_tensor_desc = TensorDesc({1, 384, 256}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto weight_tensor_desc = TensorDesc({384, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto grad_output_tensor_desc = TensorDesc({1, 384, 258}, ACL_FLOAT, ACL_FORMAT_NCL);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{384});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{2});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
    int groups = 384;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({1, 384, 256}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto gradWeight = TensorDesc({384, 1, 3}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto gradBias = TensorDesc({384}, ACL_FLOAT, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(
            grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc, padding_desc,
            dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend910_95_test_ConvBack2D_all_valid_type) {
    vector<aclDataType> ValidList = {
      ACL_FLOAT,
      ACL_FLOAT16,
      ACL_BF16
      };
    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
      auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ValidList[i], ACL_FORMAT_NCHW);
      auto weight_tensor_desc = TensorDesc({512, 512, 1, 1}, ValidList[i], ACL_FORMAT_NCHW);
      auto grad_output_tensor_desc = TensorDesc({512, 512, 7, 7}, ValidList[i], ACL_FORMAT_NCHW);

      auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{512});
      auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
      auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
      auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
      bool transposed = false;
      auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
      int groups = 1;
      auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
      auto gradInput = TensorDesc({512, 512, 7, 7}, ValidList[i], ACL_FORMAT_NCHW);
      auto gradWeight = TensorDesc({512, 512, 1, 1}, ValidList[i], ACL_FORMAT_NCHW);
      auto gradBias = TensorDesc({512}, ValidList[i], ACL_FORMAT_ND);

      int8_t cubeMathType = 0;

      auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                  INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                        padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                  OUTPUT(gradInput, gradWeight, gradBias));

      // SAMPLE: only test GetWorkspaceSize
      uint64_t workspace_size = 0;
      aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

TEST_F(convolution_backward_test, ascend910_95_test_ConvBack3D_all_valid_type) {
    vector<aclDataType> ValidList = {
      ACL_FLOAT,
      ACL_FLOAT16,
      ACL_BF16
      };
    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
      auto input_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ValidList[i], ACL_FORMAT_NCDHW);
      auto weight_tensor_desc = TensorDesc({512, 512, 1, 1, 1}, ValidList[i], ACL_FORMAT_NCDHW);
      auto grad_output_tensor_desc = TensorDesc({512, 512, 7, 7, 7}, ValidList[i], ACL_FORMAT_NCDHW);

      auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{512});
      auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
      auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
      auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
      bool transposed = false;
      auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
      int groups = 1;
      auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
      auto gradInput = TensorDesc({512, 512, 7, 7, 7}, ValidList[i], ACL_FORMAT_NCDHW);
      auto gradWeight = TensorDesc({512, 512, 1, 1, 1}, ValidList[i], ACL_FORMAT_NCDHW);
      auto gradBias = TensorDesc({512}, ValidList[i], ACL_FORMAT_ND);

      int8_t cubeMathType = 0;

      auto ut =
      OP_API_UT(aclnnConvolutionBackward,
                  INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                        padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
                  OUTPUT(gradInput, gradWeight, gradBias));

      // SAMPLE: only test GetWorkspaceSize
      uint64_t workspace_size = 0;
      aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
      EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}
}