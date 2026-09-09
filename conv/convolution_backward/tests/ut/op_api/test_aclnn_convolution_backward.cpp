/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_convolution_backward.h"
#include "op_api/op_api_def_nn.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
namespace {
class convolution_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "convolution_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "convolution_backward_test TearDown" << std::endl; }

    static void test_run(vector<int64_t> inputDims, aclDataType inputDtype, aclFormat inputFormat,
                         vector<int64_t> weightDims, aclDataType weightDtype, aclFormat weightFormat,
                         vector<int64_t> gradDims, aclDataType gradDtype, aclFormat gradFormat,
                         vector<int64_t> gradInputDims, aclDataType gradInputDtype, aclFormat gradInputFormat,
                         vector<int64_t> gradWeightDims, aclDataType gradWeightDtype, aclFormat gradWeightFormat,
                         vector<int64_t> gradBiasDims, aclDataType gradBiasDtype, aclFormat gradBiasFormat)
    {
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
TEST_F(convolution_backward_test, test_Conv2DBackward_Fp16_keep_dtype)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// TEST_F(convolution_backward_test, test_Conv2DBackward_Fp16_keep_dtype_pad4dim) {
//   auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({512, 2048, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 0, 1});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
//   auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(convolution_backward_test, test_Conv2DBackward_Fp32_allow_fp32_down_precision)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, test_Conv2DBackward_Fp32_use_fp16)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// test Conv1DBackward
TEST_F(convolution_backward_test, test_Conv1DBackward_Fp16)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, test_Conv1DBackward_invalid_format_error)
{
    auto input_tensor_desc = TensorDesc({16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// test empty
TEST_F(convolution_backward_test, test_ConvBackward_empty_error)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 防御性负向用例:输入为空指针应被参数校验拦截
TEST_F(convolution_backward_test, test_ConvBackward_nullptr_error)
{
    auto input_tensor_desc = TensorDesc({16, 16, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({16, 16, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({16, 16, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(nullptr, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc, padding_desc, dilation_desc,
              transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, nullptr, weight_tensor_desc, bias_sizes_desc, stride_desc, padding_desc,
              dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));
    workspace_size = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, nullptr, bias_sizes_desc, stride_desc, padding_desc,
              dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));
    workspace_size = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test in ascend910B
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_Fp32_keep_dtype)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_Fp32_use_hf32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// // test dilation
// TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_pre_dilation_FP16) {
//   auto input_tensor_desc = TensorDesc({512, 512, 77, 77}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({512, 2048, 21, 21}, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{4, 4});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{4, 4});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
//   auto gradInput = TensorDesc({512, 512, 77, 77}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_pre_dilation_FP32) {
//   auto input_tensor_desc = TensorDesc({512, 512, 77, 77}, ACL_FLOAT, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({512, 2048, 21, 21}, ACL_FLOAT, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{4, 4});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{4, 4});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
//   auto gradInput = TensorDesc({512, 512, 77, 77}, ACL_FLOAT, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({2048, 512, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({2048}, ACL_FLOAT, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_post_dilation_FP16) {
//   auto input_tensor_desc = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
//   auto gradInput = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_pre_dilation_hout1_wout1_FP16) {
//   auto input_tensor_desc = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({2048, 512, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({512, 2048, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{6, 6});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
//   auto gradInput = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({2048, 512, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_post_dilation_hout1_wout1_FP16) {
//   auto input_tensor_desc = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({512, 2048, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{2048});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{6, 6});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
//   auto gradInput = TensorDesc({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_post_dilation_FP32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// // 覆盖gradOutput、input、weight dtype不一致的情况
// TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_dtype_inconsistent) {
//   // FP32 + FP16 + FP16
//   test_run({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW,
//            {512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW,
//            {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT16, ACL_FORMAT_ND);
//   // FP16 + FP32 + FP16
//   test_run({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW,
//            {512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW,
//            {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT16, ACL_FORMAT_ND);
//   // FP16 + FP16 + FP32
//   test_run({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW,
//            {512, 2048, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW,
//            {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT, ACL_FORMAT_ND);

//   // FP32 + FP32 + FP16
//   test_run({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW,
//            {512, 2048, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW,
//            {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT16, ACL_FORMAT_ND);
//   // FP16 + FP32 + FP32
//   test_run({512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW,
//            {512, 2048, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW,
//            {2048, 512, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT, ACL_FORMAT_ND);
//   // FP32 + FP16 + FP32
//   test_run({512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW, {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW,
//            {512, 2048, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW, {512, 512, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW,
//            {2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW, {2048}, ACL_FLOAT, ACL_FORMAT_ND);
// }

// test Conv2DBackward BF16
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_Bf16)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_attr_error)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_FP16)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackwardInput_FP16_Conv2Matmul_1x1Kernel)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackwardInput_FP16_Conv2Matmul_Stride_Eq_Kernel)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_depthwise)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_NDHWC)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_dilation)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_stride)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_pad)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_kernel)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_strideD_GT_kernelD)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_din_limit)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_v1)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_v2)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_dw_bf16_out_fp32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_dw_use_fp16_out_fp32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackpropInputV2_WhiteCase)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackpropFilterV3_WhiteCase)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_USE_FP16)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_Bf16_USE_FP16)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_Bf16_USE_HF32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Bf16_USE_HF32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_Fp16_USE_HF32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Fp16_USE_HF32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// TEST_F(convolution_backward_test, ascend950_test_Conv2DBackward_Fp16_keep_dtype_pad4dim) {
//   auto input_tensor_desc = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({512, 2048, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 0, 1});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
//   auto gradInput = TensorDesc({512, 512, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({2048, 512, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({2048}, ACL_FLOAT16, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(convolution_backward_test, ascend950_test_Conv1DBackward_Fp16_keep_dtype)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend950_test_Conv1DBackward_Fp32_keep_dtype)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend950_test_Conv1DBackward_HF32)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend950_test_Conv1DBackward_Bf16_keep_dtype)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend950_test_Conv2DBackward_Fp16_keep_dtype_pad4dim_nhwc)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_Fp16_USE_HF32_NDHWC)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
// test empty
TEST_F(convolution_backward_test, ascend950_test_ConvBackward_empty)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// test empty
TEST_F(convolution_backward_test, ascend950_test_ConvBackward_transposed_empty)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// TEST_F(convolution_backward_test, ascend950_test_Conv2DBackwardInput_FP16_Conv2Matmul_Stride_Eq_Kernel) {
//   auto input_tensor_desc = TensorDesc({1, 128, 225, 225}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto weight_tensor_desc = TensorDesc({256, 128, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto grad_output_tensor_desc = TensorDesc({1, 256, 75, 75}, ACL_FLOAT16, ACL_FORMAT_NCHW);

//   auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{256});
//   auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
//   auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
//   bool transposed = false;
//   auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
//   int groups = 1;
//   auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
//   auto gradInput = TensorDesc({1, 128, 225, 225}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradWeight = TensorDesc({256, 128, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
//   auto gradBias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);

//   int8_t cubeMathType = 0;

//   auto ut =
//       OP_API_UT(aclnnConvolutionBackward,
//                 INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//                       padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask,
//                       cubeMathType),
//                 OUTPUT(gradInput, gradWeight, gradBias));

//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_DwStrideEqualKernelTransToKernel1x1x1)
{
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
    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend950_test_Conv3DBackpropInputV2_OpenSoraPlan1_0_bfloat16_ID4520_0003_Trans2Mm)
{
    auto input_tensor_desc = TensorDesc({1, 128, 4, 64, 64}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({256, 128, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 256, 4, 64, 64}, ACL_BF16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{256});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 128, 4, 64, 64}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({256, 128, 1, 1, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({256}, ACL_BF16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend950_test_ConvBackward_transpose_empty_error)
{
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

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// TEST_F(convolution_backward_test, ascend950_test_Conv1DBackpropFilter_fp16_matmul)
// {
//     auto input_tensor_desc = TensorDesc({4, 128,73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto weight_tensor_desc = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto grad_output_tensor_desc = TensorDesc({4, 272, 1}, ACL_FLOAT16, ACL_FORMAT_NCL);

//     auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{272});
//     auto stride_desc = IntArrayDesc(vector<int64_t>{8});
//     auto padding_desc = IntArrayDesc(vector<int64_t>{0});
//     auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
//     bool transposed = false;
//     auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
//     int groups = 1;
//     auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
//     auto gradInput = TensorDesc({4, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto gradWeight = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto gradBias = TensorDesc({272}, ACL_FLOAT16, ACL_FORMAT_ND);

//     int8_t cubeMathType = 0;
//     auto ut = OP_API_UT(
//         aclnnConvolutionBackward,
//         INPUT(
//             grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//             padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
//         OUTPUT(gradInput, gradWeight, gradBias));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(convolution_backward_test, ascend950_test_Conv1DBackpropInput_fp16_matmul)
// {
//     auto input_tensor_desc = TensorDesc({4, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto weight_tensor_desc = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto grad_output_tensor_desc = TensorDesc({4, 272, 1}, ACL_FLOAT16, ACL_FORMAT_NCL);

//     auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{272});
//     auto stride_desc = IntArrayDesc(vector<int64_t>{8});
//     auto padding_desc = IntArrayDesc(vector<int64_t>{0});
//     auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
//     bool transposed = false;
//     auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
//     int groups = 1;
//     auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
//     auto gradInput = TensorDesc({4, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto gradWeight = TensorDesc({272, 128, 73}, ACL_FLOAT16, ACL_FORMAT_NCL);
//     auto gradBias = TensorDesc({272}, ACL_FLOAT16, ACL_FORMAT_ND);

//     int8_t cubeMathType = 0;
//     auto ut = OP_API_UT(
//         aclnnConvolutionBackward,
//         INPUT(
//             grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
//             padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
//         OUTPUT(gradInput, gradWeight, gradBias));

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(convolution_backward_test, ascend950_test_Conv1DBackpropFilter_fp32_group)
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
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
TEST_F(convolution_backward_test, ascend950_test_ConvBack2D_all_valid_type)
{
    vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16, ACL_BF16};
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

        auto ut = OP_API_UT(
            aclnnConvolutionBackward,
            INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                  padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
            OUTPUT(gradInput, gradWeight, gradBias));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

TEST_F(convolution_backward_test, ascend950_test_ConvBack3D_all_valid_type)
{
    vector<aclDataType> ValidList = {ACL_FLOAT, ACL_FLOAT16, ACL_BF16};
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

        auto ut = OP_API_UT(
            aclnnConvolutionBackward,
            INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
                  padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
            OUTPUT(gradInput, gradWeight, gradBias));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_DH_Swap)
{
    auto input_tensor_desc = TensorDesc({1, 512, 4099, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({512, 1, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 512, 4096, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{512});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 512;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 512, 4099, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({512, 1, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(convolution_backward_test, ascend910B2_test_groups_zero_error)
{
    auto input_tensor_desc = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 0;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_weightCout_not_divisible_by_groups_error)
{
    auto input_tensor_desc = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({17, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 17, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{17});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 2;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({17, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({17}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_input_dim_less_than_3_error)
{
    auto input_tensor_desc = TensorDesc({0, 0}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({0, 0}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_gradOutput_input_dim_not_equal_error)
{
    auto input_tensor_desc = TensorDesc({1, 16, 10}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto weight_tensor_desc = TensorDesc({16, 16, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 10}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradWeight = TensorDesc({16, 16, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_empty_tensor_biasSizes_nullptr_error)
{
    auto input_tensor_desc = TensorDesc({0, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({0, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({0, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnConvolutionBackwardGetWorkspaceSize(
        grad_output_tensor_desc.ToAclTypeRawPtr(), input_tensor_desc.ToAclTypeRawPtr(),
        weight_tensor_desc.ToAclTypeRawPtr(), nullptr, stride_desc.ToAclTypeRawPtr(), padding_desc.ToAclTypeRawPtr(),
        dilation_desc.ToAclTypeRawPtr(), transposed, output_padding_desc.ToAclTypeRawPtr(), groups,
        output_mask.ToAclTypeRawPtr(), cubeMathType, gradInput.ToAclTypeRawPtr(), gradWeight.ToAclTypeRawPtr(),
        gradBias.ToAclTypeRawPtr(), &workspace_size, &executor);

    if (executor != nullptr) {
        delete executor;
    }

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_gradOutput_empty_tensor_error)
{
    auto input_tensor_desc = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({0, 16, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_weight_empty_tensor_error)
{
    auto input_tensor_desc = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({0, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({0, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

    int8_t cubeMathType = 0;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(convolution_backward_test, ascend910B2_test_Conv1DTransposeBackward_FP16)
{
    auto input_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto weight_tensor_desc = TensorDesc({16, 8, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto grad_output_tensor_desc = TensorDesc({4, 8, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{8});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({4, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradWeight = TensorDesc({16, 8, 3}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradBias = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DTransposeBackward_FP32)
{
    auto input_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DTransposeBackward_FP16)
{
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DTransposeBackward_FP32)
{
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DTransposeBackward_BF16)
{
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_PreDilation_FP16_2dim_padding)
{
    auto input_tensor_desc = TensorDesc({1, 16, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({32, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 32, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{32});
    auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({32, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_PreDilation_FP32_4dim_padding)
{
    auto input_tensor_desc = TensorDesc({1, 16, 14, 14}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({32, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 32, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{32});
    auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 14, 14}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({32, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_PreDilation_FP32_gradOutputH1)
{
    auto input_tensor_desc = TensorDesc({1, 16, 4, 14}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({32, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 32, 1, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{32});
    auto stride_desc = IntArrayDesc(vector<int64_t>{10, 3});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 4, 14}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({32, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_PreDilation_FP16_gradOutputW1)
{
    auto input_tensor_desc = TensorDesc({1, 16, 14, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({32, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 32, 5, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{32});
    auto stride_desc = IntArrayDesc(vector<int64_t>{3, 10});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 14, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({32, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackpropFilterBy1x1Dw)
{
    auto input_tensor_desc = TensorDesc({1024, 3, 224, 224}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({1024, 3, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1024, 1024, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1024});
    auto stride_desc = IntArrayDesc(vector<int64_t>{16, 16});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1024, 3, 224, 224}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({1024, 3, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({1024}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend950_test_Conv3DBackpropFilterBy1x1Dw)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
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
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1, 3, 16, 224, 224}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({768, 3, 2, 16, 16}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({768}, ACL_BF16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend950_test_Conv1DBackwardByMatmul)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto input_tensor_desc = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto weight_tensor_desc = TensorDesc({6, 4, 8}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto grad_output_tensor_desc = TensorDesc({2, 6, 1}, ACL_FLOAT16, ACL_FORMAT_NCL);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{6});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({2, 4, 8}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradWeight = TensorDesc({6, 4, 8}, ACL_FLOAT16, ACL_FORMAT_NCL);
    auto gradBias = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackwardDwByMmMode_FmEqKernel)
{
    auto input_tensor_desc = TensorDesc({1, 4, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({6, 4, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 6, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{6});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({1, 4, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({6, 4, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend950_test_Conv3DBackwardW1B1FmNDxByMm)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto input_tensor_desc = TensorDesc({1, 4, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({6, 4, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 6, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{6});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 4, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({6, 4, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({6}, ACL_FLOAT16, ACL_FORMAT_ND);

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

TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_Fp32_USE_HF32)
{
    auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    int8_t cubeMathType = 3; // USE_HF32
    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
// special case skips pre-dilation and calls PerformConv2DBackpropInput directly
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_pre_dilation_special_shape)
{
    auto input_tensor_desc = TensorDesc({8, 3, 896, 896}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({1280, 3, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({8, 1280, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1280});
    auto stride_desc = IntArrayDesc(vector<int64_t>{14, 14});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({8, 3, 896, 896}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({1280, 3, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({1280}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L1124-L1146: PostDilation 4 padding ,
// gradOutput H/W 1 stride > input+pad dilation (1x1 kernel + stride 3 > 2)
TEST_F(convolution_backward_test, ascend910B2_test_ConvBackward_post_dilation_pad4dim_hout1)
{
    auto input_tensor_desc = TensorDesc({1, 4, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({4, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{4});
    auto stride_desc = IntArrayDesc(vector<int64_t>{3, 3});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 4, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({4, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L1368-L1371: Check2DTransTo1x1DwFlag hits the C04 whitelist
// padding 0 false, dw Conv2DBackpropFilter
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_1x1Dw_C04_pad_not_zero)
{
    auto input_tensor_desc = TensorDesc({1024, 3, 224, 224}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({1024, 3, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1024, 1024, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1024});
    auto stride_desc = IntArrayDesc(vector<int64_t>{16, 16});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1024, 3, 224, 224}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({1024, 3, 16, 16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({1024}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L1376-L1381: Check2DTransTo1x1DwFlag hits the C04 whitelist
// stride != kernel false (stride 15 != kernel 17, 14x14)
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DBackward_1x1Dw_C04_stride_ne_kernel)
{
    auto input_tensor_desc = TensorDesc({1024, 3, 224, 224}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({1024, 3, 17, 17}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1024, 1024, 14, 14}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{1024});
    auto stride_desc = IntArrayDesc(vector<int64_t>{15, 15});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1024, 3, 224, 224}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({1024, 3, 17, 17}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({1024}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L367-L374: IsConv3DVecFallbackCase dtype
// (input FP16, weight/gradOutput BF16) vec
TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_vec_dtype_mismatch)
{
    auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_BF16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L3065-L3075: 3D dx(outputMask[1]=false)
// CalcPromoteType (Regbase)
TEST_F(convolution_backward_test, ascend910B2_test_Conv3DTransposeBackward_dx_only)
{
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L1674-L1676: 2D dx weight dtype FP32
// Conv2d5HdFp32(useHf32=false)
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DTransposeBackward_Fp32_dx)
{
    auto input_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L1671-L1673: 2D dx FP32 + cubeMathType=USE_HF32
// then falls to Conv2d5HdFp32 (useHf32=true)
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DTransposeBackward_Fp32_USE_HF32_dx)
{
    auto input_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    int8_t cubeMathType = 3; // USE_HF32

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// cover aclnn_convolution_backward.cpp L1677-L1679: 2D dx input dtype BF16
// Conv2d5HdBf16
TEST_F(convolution_backward_test, ascend910B2_test_Conv2DTransposeBackward_Bf16_dx)
{
    auto input_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_BF16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 7, 7}, ACL_BF16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 7, 7}, ACL_BF16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({16, 16, 3, 3}, ACL_BF16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L2830-L2845: 910B2 3D dx CheckWeightPreTransposeEnable
// , weight NCDHW->NDHWC
TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_weight_pre_transpose)
{
    auto input_tensor_desc = TensorDesc({1, 16, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 4, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 2, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1, 16, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L279-L282/L336-L347: CalcConv3DDxL1Estimate cW=16
// kBlockSize % cW == 0 hValueMax (IsExceedL1For3DDx IsConv3DVecFallbackCase )
TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackward_vec_l1_estimate_w16)
{
    auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 16}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 16}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 2, 10, 16}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L2163-L2164/L2182-L2184: GenDxInOutByConvBp2MmMode
// FEATURE_MAP_EQ_KERNEL gradShapeVec outShapeVec else (kernel input DHW
// gradOutput DHW 1, dx)
TEST_F(convolution_backward_test, ascend910B2_test_Conv3DBackwardDxByMm_FmEqKernel_dx_only)
{
    auto input_tensor_desc = TensorDesc({1, 16, 2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L2420-L2423/L2703-L2708: 950 3D dw IsW1B1FmNDxTransToMm
// kernel 1 IsTransTo1x1Dw stride != kernel
TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_dw_kernel3_pad0)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L2409-L2412/L2509-L2512/L2698-L2701: 950 3D 1x1x1 kernel +
// dilation=2 IsW1B1FmNDxTransToMm / Is1x1DwTransToMm / IsTransTo1x1Dw dilation != 1
TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_w1b1_dilation2)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 2, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L2500-L2503: 950 3D 1x1x1 kernel + stride=2
// Is1x1DwTransToMm stride != 1
TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_1x1dw_stride2)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 2, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 2, 2});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1, 16, 2, 10, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L2364-L2366/L2467-L2470: 1x1x1 kernel shape L2Cache
// (128MB) IsGreaterL2Cache / IsGreaterL2CacheForDw1x1 , matmul cube
TEST_F(convolution_backward_test, ascend950_test_Conv3DBackward_w1b1_l2_exceed)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 256, 64, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({256, 256, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 256, 64, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{256});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1, 256, 64, 64, 64}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({256, 256, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L3013-L3015/L3020/L773-L784/L3105-L3108/L3115-L3118:
// 950 3D FP16 Conv3dv2NCDHWFp16, dx OutputPostProcessTransposed,
// dw Fp162Fp32 OutputPostProcessWithoutTransdata
TEST_F(convolution_backward_test, ascend950_test_Conv3DTransposeBackward_Fp16)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L3004-L3006: 950 3D FP32 Conv3dv2NCDHWFp32
// (useHf32=false)
TEST_F(convolution_backward_test, ascend950_test_Conv3DTransposeBackward_Fp32)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L3007-L3009: 950 3D BF16 Conv3dv2NCDHWBf16
TEST_F(convolution_backward_test, ascend950_test_Conv3DTransposeBackward_Bf16)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L2099-L2101/L2391-L2393/L2680-L2682/L3010-L3012:
// 950 3D HIFLOAT8: 8bit matmul(NO_MM), dx Conv3dv2NCDHWHif8,
// gradWeight FP32 (dw 8bit ), IsW1B1FmNDxTransToMm/IsTransTo1x1Dw 8bit
TEST_F(convolution_backward_test, ascend950_test_Conv3DTransposeBackward_Hif8)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_HIFLOAT8, ACL_FORMAT_ND);

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

// cover aclnn_convolution_backward.cpp L3066-L3067/L387-L403/L697-L703/L721-L724/L3153-L3157:
// 950 3D HIFLOAT8 dx: CalcPromoteTypeTransposed/GetUpperFloatDataTypeTransposed,
// AttrPreProcess InputPreProcess 8bit + USE_FP16 Cast,
// CheckCubeMathTypeFor3D USE_FP16 + 8bit
TEST_F(convolution_backward_test, ascend950_test_Conv3DTransposeBackward_Hif8_dx_only_USE_FP16)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_HIFLOAT8, ACL_FORMAT_ND);

    int8_t cubeMathType = 2; // USE_FP16

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// cover aclnn_convolution_backward.cpp L3146-L3151: 950 3D HIFLOAT8 + cubeMathType=
// ALLOW_FP32_DOWN_PRECISION
TEST_F(convolution_backward_test, ascend950_test_Conv3DTransposeBackward_Hif8_dx_only_ALLOW_FP32)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_HIFLOAT8, ACL_FORMAT_ND);

    int8_t cubeMathType = 1; // ALLOW_FP32_DOWN_PRECISION

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// cover aclnn_convolution_backward.cpp L3159-L3163: 950 3D HIFLOAT8 + cubeMathType=
// USE_HF32
TEST_F(convolution_backward_test, ascend950_test_Conv3DTransposeBackward_Hif8_dx_only_USE_HF32)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 1, 7, 7}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 3}, ACL_HIFLOAT8, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({16}, ACL_HIFLOAT8, ACL_FORMAT_ND);

    int8_t cubeMathType = 3; // USE_HF32

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

} // namespace
