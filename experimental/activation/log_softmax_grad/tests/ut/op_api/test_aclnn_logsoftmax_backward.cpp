/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "opdev/op_log.h"
#include "../../../op_host/op_api/aclnn_logsoftmax_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class LogSoftmaxBackwardTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LogSoftmaxBackwardTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LogSoftmaxBackwardTest TearDown" << std::endl; }
};

// 测试合法数据类型float32
TEST_F(LogSoftmaxBackwardTest, case_001_float32)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试合法数据类型float16
TEST_F(LogSoftmaxBackwardTest, case_002_float16)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试合法数据类型bfloat16
TEST_F(LogSoftmaxBackwardTest, case_003_bfloat16)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 测试非法数据类型double
TEST_F(LogSoftmaxBackwardTest, case_004_float64)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试非法数据类型int32
TEST_F(LogSoftmaxBackwardTest, case_005_invalid_type_int)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试空Tensor
TEST_F(LogSoftmaxBackwardTest, case_006_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outputDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试非连续tensor
TEST_F(LogSoftmaxBackwardTest, case_007_not_contiguous)
{
    auto gradOutputDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto outputDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试空指针gradOutput
TEST_F(LogSoftmaxBackwardTest, case_008_nullptr_gradOutput)
{
    auto tensorDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT((aclTensor*)nullptr, tensorDesc, dim), OUTPUT(tensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试空指针output
TEST_F(LogSoftmaxBackwardTest, case_009_nullptr_output)
{
    auto tensorDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(tensorDesc, (aclTensor*)nullptr, dim), OUTPUT(tensorDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试空指针gradInput
TEST_F(LogSoftmaxBackwardTest, case_010_nullptr_gradInput)
{
    auto tensorDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(tensorDesc, tensorDesc, dim), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试维度大于8
TEST_F(LogSoftmaxBackwardTest, case_011_max_dim)
{
    auto gradOutputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试不同dtype
TEST_F(LogSoftmaxBackwardTest, case_012_dtype_mismatch_gradOutput_output)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试不同shape
TEST_F(LogSoftmaxBackwardTest, case_013_shape_mismatch)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试不同dim位置 (dim=0)
TEST_F(LogSoftmaxBackwardTest, case_014_dim_0)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = 0;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试不同dim位置 (dim=1)
TEST_F(LogSoftmaxBackwardTest, case_015_dim_1)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = 1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试不同dim位置 (dim=2)
TEST_F(LogSoftmaxBackwardTest, case_016_dim_2)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = 2;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试1D tensor
TEST_F(LogSoftmaxBackwardTest, case_017_1d_tensor)
{
    auto gradOutputDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试2D tensor
TEST_F(LogSoftmaxBackwardTest, case_018_2d_tensor)
{
    auto gradOutputDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试5D tensor
TEST_F(LogSoftmaxBackwardTest, case_019_5d_tensor)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试8D tensor (最大支持维度)
TEST_F(LogSoftmaxBackwardTest, case_020_8d_tensor)
{
    auto gradOutputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试非连续tensor float16
TEST_F(LogSoftmaxBackwardTest, case_021_not_contiguous_float16)
{
    auto gradOutputDesc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto outputDesc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试不同dtype (output为float, gradOutput为float16)
TEST_F(LogSoftmaxBackwardTest, case_022_dtype_mismatch_reverse)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试gradOutput空tensor
TEST_F(LogSoftmaxBackwardTest, case_023_gradOutput_empty)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outputDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试output空tensor
TEST_F(LogSoftmaxBackwardTest, case_024_output_empty)
{
    auto gradOutputDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试complex64类型
TEST_F(LogSoftmaxBackwardTest, case_025_complex64)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试complex128类型
TEST_F(LogSoftmaxBackwardTest, case_026_complex128)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto outputDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试不同shape导致shape不兼容
TEST_F(LogSoftmaxBackwardTest, case_027_incompatible_shape)
{
    auto gradOutputDesc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试正数dim
TEST_F(LogSoftmaxBackwardTest, case_028_dim_positive)
{
    auto gradOutputDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outputDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 0);
    int64_t dim = 2;

    auto gradInputDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试float16非连续tensor
TEST_F(LogSoftmaxBackwardTest, case_029_not_contiguous_float32)
{
    auto gradOutputDesc = TensorDesc({6, 8}, ACL_FLOAT, ACL_FORMAT_ND, {1, 6}, 0, {8, 6}).ValueRange(-2, 2);
    auto outputDesc = TensorDesc({6, 8}, ACL_FLOAT, ACL_FORMAT_ND, {1, 6}, 0, {8, 6}).ValueRange(-10, 0);
    int64_t dim = 0;

    auto gradInputDesc = TensorDesc({6, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

// 测试bfloat16非连续tensor
TEST_F(LogSoftmaxBackwardTest, case_030_not_contiguous_bfloat16)
{
    auto gradOutputDesc = TensorDesc({5, 4}, ACL_BF16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto outputDesc = TensorDesc({5, 4}, ACL_BF16, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 0);
    int64_t dim = -1;

    auto gradInputDesc = TensorDesc({5, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.005, 0.005);

    auto ut = OP_API_UT(aclnnLogSoftmaxBackward, INPUT(gradOutputDesc, outputDesc, dim), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}
