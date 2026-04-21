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
#include "../../../op_api/aclnn_convolution_backward.h"
#include "op_api/op_api_def.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
namespace {
int64_t CalculateOutputSize(int64_t inSize, int64_t padding0, int64_t padding1,
                            int64_t kernelSize, int64_t dilation, int64_t stride) {
    return (inSize + padding0 + padding1 - ((kernelSize - 1) * dilation + 1)) / stride + 1;
}

int64_t CalculateOffsetOutChannels(int64_t outH, int64_t outW, int64_t K_H, int64_t K_W) {
    return outH * K_H;
}

int64_t CalculateOffsetOutWidth(int64_t outW, int64_t K_W) {
    return outW * K_W;
}

int64_t CalculateOffsetChannels(bool modulated, int64_t deformableGroups, int64_t K_H, int64_t K_W) {
    int64_t multiplier = modulated ? 3 : 2;
    return multiplier * deformableGroups * K_H * K_W;
}

class deformable_conv2d_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "deformable_conv2d_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "deformable_conv2d_backward_test TearDown" << std::endl;
    }
};

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_FP16)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 64, inH = 128, inW = 128;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_FP32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 2, inC = 32, inH = 64, inW = 64;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_BF16)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 64, inH = 64, inW = 64;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_BF16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_BF16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_BF16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_BF16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_BF16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_BF16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_BF16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_BF16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_BF16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_groups)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 64, inH = 32, inW = 32;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 4, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_deformable_groups)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 32, inW = 32;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 2;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_stride2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 64, inW = 64;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 2, 2});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_dilation2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 64, inW = 64;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 2, 2});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_kernel5x5)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 32, inW = 32;
    int64_t outC = 64, K_H = 5, K_W = 5;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{2, 2, 2, 2});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_kernel7x7)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 32, inW = 32;
    int64_t outC = 64, K_H = 7, K_W = 7;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{3, 3, 3, 3});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_stride4)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 64, inW = 64;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 4, 4});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_dilation3)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 64, inW = 64;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{3, 3, 3, 3});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 3, 3});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_invalid_offsetOut_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 64, inH = 128, inW = 128;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, 128, 128}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_invalid_offset_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 64, inH = 128, inW = 128;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, 18, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_invalid_gradBias_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 64, inH = 128, inW = 128;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(deformable_conv2d_backward_test, test_DeformableConv2dBackward_invalid_weight_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t N = 1, inC = 32, inH = 64, inW = 64;
    int64_t outC = 64, K_H = 3, K_W = 3;
    int64_t groups = 1, deformableGroups = 1;
    bool modulated = true;
    auto stride_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});
    auto dilation_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1});

    int64_t outH = CalculateOutputSize(inH, padding_desc.Get()[0], padding_desc.Get()[1], K_H, dilation_desc.Get()[2], stride_desc.Get()[2]);
    int64_t outW = CalculateOutputSize(inW, padding_desc.Get()[2], padding_desc.Get()[3], K_W, dilation_desc.Get()[3], stride_desc.Get()[3]);
    int64_t offsetOutC = CalculateOffsetOutChannels(outH, outW, K_H, K_W);
    int64_t offsetOutW = CalculateOffsetOutWidth(outW, K_W);
    int64_t offsetC = CalculateOffsetChannels(modulated, deformableGroups, K_H, K_W);

    auto input_tensor_desc = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto grad_output_tensor_desc = TensorDesc({N, outC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_out_tensor_desc = TensorDesc({N, inC, offsetOutC, offsetOutW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({outC, 64, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto offset_tensor_desc = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto gradInput = TensorDesc({N, inC, inH, inW}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeight = TensorDesc({outC, inC / groups, K_H, K_W}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradBias = TensorDesc({outC}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradOffset = TensorDesc({N, offsetC, outH, outW}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto kernel_size_desc = IntArrayDesc(vector<int64_t>{K_H, K_W});

    auto ut = OP_API_UT(
        aclnnDeformableConv2dBackward,
        INPUT(
            input_tensor_desc, grad_output_tensor_desc, offset_out_tensor_desc, weight_tensor_desc, offset_tensor_desc,
            kernel_size_desc, stride_desc, padding_desc, dilation_desc, groups, deformableGroups, modulated),
        OUTPUT(gradInput, gradWeight, gradOffset, gradBias));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
} // namespace