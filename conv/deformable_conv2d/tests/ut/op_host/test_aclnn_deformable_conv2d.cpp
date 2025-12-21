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
#include "../../../op_host/op_api/aclnn_deformable_conv2d.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class l2DeformableConv2dTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2DeformableConv2dTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2DeformableConv2dTest TearDown" << std::endl;
    }
};

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_float32)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_float16)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_bfloat16)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_nullptr_deformable_out)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_nullptr_x)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            nullptr, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_invalid_dtype)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_invalid_shape)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({33}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_invalid_format)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, 1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = 1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2DeformableConv2dTest, ascend910B2_deformable_conv2d_invalid_attr)
{
    auto xDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto weightDesc = TensorDesc({2, 2, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto offsetDesc = TensorDesc({2, 3, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto biasOptionalDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);

    vector<int64_t> kernelSize = {1, 1};
    auto kernelSizeArray = IntArrayDesc(kernelSize);
    vector<int64_t> strideSize = {1, -1, 1, 1};
    auto strideSizeArray = IntArrayDesc(strideSize);
    vector<int64_t> paddingSize = {0, 0, 0, 0};
    auto paddingSizeArray = IntArrayDesc(paddingSize);
    vector<int64_t> dilationSize = {-1, 1, 1, 1};
    auto dilationSizeArray = IntArrayDesc(dilationSize);

    int64_t groups = 1;
    int64_t deformableGroups = -1;
    bool modulated = true;

    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto deformOutOptionalDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnDeformableConv2d,
        INPUT(
            xDesc, weightDesc, offsetDesc, biasOptionalDesc, kernelSizeArray, strideSizeArray, paddingSizeArray,
            dilationSizeArray, groups, deformableGroups, modulated),
        OUTPUT(outDesc, deformOutOptionalDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}