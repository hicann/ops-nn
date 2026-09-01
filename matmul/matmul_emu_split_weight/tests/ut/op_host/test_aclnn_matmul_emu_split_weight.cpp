/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cfloat>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_matmul_emu_split_weight.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_matmul_emu_split_weight_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_matmul_emu_split_weight_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_matmul_emu_split_weight_test TearDown" << endl; }
};

// x 为 nullptr
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_x_nullptr_fail)
{
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight,
                        INPUT((aclTensor*)nullptr, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// w_high 为 nullptr
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_wHigh_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight,
                        INPUT(x_desc, (aclTensor*)nullptr, wLow_desc, y_desc, wLowScale, yDtype), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// w_low 为 nullptr
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_wLow_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight,
                        INPUT(x_desc, wHigh_desc, (aclTensor*)nullptr, y_desc, wLowScale, yDtype), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y 为 nullptr
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_y_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight,
                        INPUT(x_desc, wHigh_desc, wLow_desc, (aclTensor*)nullptr, wLowScale, yDtype), OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 数据类型不支持 (FP32)
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_x_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// w_high 数据类型与 x 不一致
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_wHigh_dtype_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y 数据类型不支持 (BF16)
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_y_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_BF16, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// yDtype 不支持 (非0)
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_yDtype_invalid_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 1;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale 不合法 (非1/256)
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_scale_invalid_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.01f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale 为 NaN
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_scale_nan_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = NAN;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 维度不是2D
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_x_dim_fail)
{
    TensorDesc x_desc = TensorDesc({2, 128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// K 维度不匹配
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_k_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({255, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({255, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// w_low shape 与 w_high 不一致
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_wLow_shape_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 127}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y shape 与 [M, N] 不匹配
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_y_shape_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 127}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// NZ 格式不支持
TEST_F(l2_matmul_emu_split_weight_test, ascend910B1_matmul_emu_split_weight_nz_format_fail)
{
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {8, 1, 16, 16});
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_emu_split_weight_test, ascend910B_matmul_emu_split_weight_bf16_fp32_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_emu_split_weight_test, ascend910B_matmul_emu_split_weight_small_shape_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x_desc = TensorDesc({16, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_emu_split_weight_test, ascend910_93_matmul_emu_split_weight_bf16_fp32_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND910_93);
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// // Ascend950 正常场景: UT环境缺少平台信息，tiling返回错误
// TEST_F(l2_matmul_emu_split_weight_test, ascend950_matmul_emu_split_weight_tiling_fail)
// {
//     SocVersionManager versionManager(SocVersion::ASCEND950);
//     TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

//     float wLowScale = 0.00390625f;
//     int8_t yDtype = 0;

//     auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
//                         OUTPUT());
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_NE(aclRet, ACLNN_SUCCESS);
// }

// Ascend950 非连续tensor: UT环境缺少平台信息，tiling返回错误
TEST_F(l2_matmul_emu_split_weight_test, ascend950_matmul_emu_split_weight_non_contiguous_tiling_fail)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND, {256 * 2, 1}, 0, {128, 256});
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_matmul_emu_split_weight_test, ascend950_matmul_emu_split_weight_bf16_fp32_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    TensorDesc x_desc = TensorDesc({128, 256}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wHigh_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc wLow_desc = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({128, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    float wLowScale = 0.00390625f;
    int8_t yDtype = 0;

    auto ut = OP_API_UT(aclnnMatmulEmuSplitWeight, INPUT(x_desc, wHigh_desc, wLow_desc, y_desc, wLowScale, yDtype),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
