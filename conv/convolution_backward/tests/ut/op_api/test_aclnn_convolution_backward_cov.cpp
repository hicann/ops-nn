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
class convolution_backward_cov_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "convolution_backward_cov_test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "convolution_backward_cov_test TearDown" << std::endl; }
};

// cover convolutionbackward.cpp L288-L297: DAV_2201 CheckV2Functionality L1 size limit
// (fmapH * Wo * strideW > 4096 Conv3ddx v2 V1)
TEST_F(convolution_backward_cov_test, cov_v2_l1_limit_dav2201)
{
    auto input_tensor_desc = TensorDesc({1, 16, 1, 64, 1200}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 16, 1, 62, 1200}, ACL_BF16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({1, 16, 1, 64, 1200}, ACL_BF16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 16, 1, 3, 1}, ACL_BF16, ACL_FORMAT_NCDHW);
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

// cover convolutionbackward.cpp L760-L846/L873-L905: 950 FP32 3D dw CheckPreNHTransposeEnable ,
// N2H transpose ; dx CheckN2HEnable/N2HOptimize/N2HChangeOutput L1156-L1157
TEST_F(convolution_backward_cov_test, cov_nh_transpose_dw_full_pass)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L805-L813: 950 FP32 3D dw strideD != 1 NH transpose disable
TEST_F(convolution_backward_cov_test, cov_nh_transpose_dw_stride_d)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{2, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L815-L825: 950 FP32 3D dw dilationW > 255 NH transpose disable
TEST_F(convolution_backward_cov_test, cov_nh_transpose_dw_dilation_w)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 256});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L826-L832: 950 FP32 3D dw pad 0 NH transpose disable
TEST_F(convolution_backward_cov_test, cov_nh_transpose_dw_pad)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 3, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L835-L844: 950 FP32 3D dw inputN < 1500 NH transpose disable
TEST_F(convolution_backward_cov_test, cov_nh_transpose_dw_small_batch)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({100, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({100, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({100, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L873-L905: 950 FP32 3D transposed dw Conv3DBackpropFilterFp322Fp32
// CheckPreNHTransposeEnable transpose
TEST_F(convolution_backward_cov_test, cov_nh_transpose_dw_transposed_conv)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = true;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, false});
    auto gradInput = TensorDesc({2048, 64, 1, 1, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1080-L1082: 950 3D dx batch < 1500 N2H disable
TEST_F(convolution_backward_cov_test, cov_n2h_dx_batch_limit)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({100, 64, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({100, 64, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({100, 64, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1084-L1087: 950 3D dx inputD != 1 N2H disable (1D )
TEST_F(convolution_backward_cov_test, cov_n2h_dx_dh_not_one)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 2, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 2, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 64, 2, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1088-L1090 + L1156-L1158: 950 3D dx cin > 128 N2H disable;
// FP32 + 1x1x1 kernel CheckWeightPreTransposeEnable dk*hk*wk <= 1
TEST_F(convolution_backward_cov_test, cov_n2h_dx_cin_over_limit)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 200, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({40, 200, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 40, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{40});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 200, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({40, 200, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({40}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1057-L1059 + L1147-L1153: 950 3D dx wi >= 60 wi/ cin
// CheckN2HAttrCriteria ; FP16 dk <= 1 weight
TEST_F(convolution_backward_cov_test, cov_n2h_dx_criteria_wi_ge_60)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 40, 1, 1, 62}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({40, 40, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 40, 1, 1, 62}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{40});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 40, 1, 1, 62}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({40, 40, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({40}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1060-L1062: 950 3D dx 40 <= wi < 60 wi/ cin
// hits the disabled branch of CheckN2HAttrCriteria
TEST_F(convolution_backward_cov_test, cov_n2h_dx_criteria_wi_40_to_60)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 10, 1, 1, 50}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({10, 10, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 10, 1, 1, 50}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{10});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 10, 1, 1, 50}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({10, 10, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1023-L1031: 950 3D dx strideW > 63 CheckN2HAttrAvailable
TEST_F(convolution_backward_cov_test, cov_n2h_dx_stride_w_over)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 1, 1, 128}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 64});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 64, 1, 1, 128}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1036-L1041: 950 3D dx pad 0 CheckN2HAttrAvailable
TEST_F(convolution_backward_cov_test, cov_n2h_dx_pad_nonzero)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 3, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 5, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 64, 3, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1046-L1051: 950 3D dx dilation != 1 CheckN2HAttrAvailable
TEST_F(convolution_backward_cov_test, cov_n2h_dx_dilation_nonone)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 64, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 2});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 64, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 64, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1183-L1194: 950 3D dx N2H weight N >= 64 && C >= 128,
// N2HOptimize weight
TEST_F(convolution_backward_cov_test, cov_n2h_dx_weight_transpose)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 128, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({64, 128, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 64, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{64});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 128, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({64, 128, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1098-L1100 + L1116-L1119: 950 3D dx groups > 1 // CheckN2HEnable
// CheckWeightPreTransposeEnable groups
TEST_F(convolution_backward_cov_test, cov_weight_pre_transpose_groups)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 32, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 16, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 2;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 32, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
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

// cover convolutionbackward.cpp L1159-L1161: 950 FP32 3D dx kernel 1x3x3 cin=200 // CheckWeightPreTransposeEnable cin
// (cin != 16 && cin < 256)
TEST_F(convolution_backward_cov_test, cov_weight_pre_transpose_cin)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 200, 2, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({40, 200, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 40, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{40});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 200, 2, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({40, 200, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradBias = TensorDesc({40}, ACL_FLOAT, ACL_FORMAT_ND);

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

// cover convolutionbackward.cpp L1162: 950 FP32 3D dx kernel 1x3x3 cin == 16 // CheckWeightPreTransposeEnable true,
// weight NCDHW->NDHWC
TEST_F(convolution_backward_cov_test, cov_weight_pre_transpose_enable)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({2048, 16, 2, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({2048, 16, 2, 6, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, false});
    auto gradInput = TensorDesc({2048, 16, 2, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);
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

// cover convolution_backward_checker.cpp L50-L60: cubeMathType 4)
TEST_F(convolution_backward_cov_test, cov_checker_cubemathtype_invalid)
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

    int8_t cubeMathType = 4;

    auto ut = OP_API_UT(
        aclnnConvolutionBackward,
        INPUT(grad_output_tensor_desc, input_tensor_desc, weight_tensor_desc, bias_sizes_desc, stride_desc,
              padding_desc, dilation_desc, transposed, output_padding_desc, groups, output_mask, cubeMathType),
        OUTPUT(gradInput, gradWeight, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// cover convolution_backward_checker.cpp L699-L709: input weight cin
TEST_F(convolution_backward_cov_test, cov_checker_channel_not_divisible)
{
    auto input_tensor_desc = TensorDesc({2, 6, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({8, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({2, 8, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{8});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({2, 6, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({8, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW);
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

// cover convolution_backward_checker.cpp L767-L770: 910B() 3D D kernel // CheckResolutionGEKernelShape
TEST_F(convolution_backward_cov_test, cov_checker_3d_resolution_2201)
{
    auto input_tensor_desc = TensorDesc({1, 4, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({8, 4, 3, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 8, 1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{8});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 4, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({8, 4, 3, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
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

// cover convolution_backward_checker.cpp L146-L199 + L225-L237: 950 3D D kernel //
// GetExpectValueDHW_95/CheckResolutionGEKernelShape_95/GetInputShapeSize/GetWeightShapeSize
TEST_F(convolution_backward_cov_test, cov_checker_3d_resolution_950)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({1, 4, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({8, 4, 3, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({1, 8, 1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{8});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({1, 4, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({8, 4, 3, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
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

// cover convolution_backward_checker.cpp L900-L903 + L926-L934(L929/L932) + L1012-L1022 + L978-L993:
// 950 tensor 2D + 4 padding/outputPadding, CheckParamsDim 4 padding tensor
TEST_F(convolution_backward_cov_test, cov_checker_empty_2d_pad4_950)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({0, 16, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({16, 16, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({0, 16, 7, 7}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 1, 0, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0, 0});
    int groups = 1;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({0, 16, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
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
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// cover convolution_backward_checker.cpp L958-L966(L961) + L1014-L1015: 950 tensor 3D
// inputC/weightCin groups CheckParamsGroup
TEST_F(convolution_backward_cov_test, cov_checker_empty_3d_groups_950)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto input_tensor_desc = TensorDesc({0, 8, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto weight_tensor_desc = TensorDesc({16, 8, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto grad_output_tensor_desc = TensorDesc({0, 16, 2, 6, 6}, ACL_FLOAT16, ACL_FORMAT_NCDHW);

    auto bias_sizes_desc = IntArrayDesc(vector<int64_t>{16});
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1});
    bool transposed = false;
    auto output_padding_desc = IntArrayDesc(vector<int64_t>{0, 0, 0});
    int groups = 2;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});
    auto gradInput = TensorDesc({0, 8, 2, 8, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto gradWeight = TensorDesc({16, 8, 1, 3, 3}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
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
} // namespace
