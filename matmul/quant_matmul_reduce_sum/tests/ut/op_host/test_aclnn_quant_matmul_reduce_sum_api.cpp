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
 * \file test_aclnn_quant_matmul_reduce_sum_api.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_quant_matmul_reduce_sum.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_quant_matmul_reduce_sum_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_quant_matmul_reduce_sum_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_quant_matmul_reduce_sum_test TearDown" << endl;
    }
};

/*
 常规 opapi_UT 算例
*/
static void QuantMatmulTestCaseNormal(int64_t b, int64_t m, int64_t k, int64_t n)
{
    // 使用**Desc描述host api输入输出
    auto x1Desc = TensorDesc({b, m, k}, ACL_INT8, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({b, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto dimsDesc = IntArrayDesc(vector<int64_t>{0});
    auto x2ScaleDesc = TensorDesc({n}, ACL_BF16, ACL_FORMAT_ND);
    auto x1ScaleDesc = TensorDesc({b, m}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({m, n}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantMatmulReduceSumWeightNz,
        INPUT(
            x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
            nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
            false, false, -1, dimsDesc, false),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    bool pass = (ACLNN_ERR_INNER_NULLPTR == aclRet || ACLNN_SUCCESS == aclRet);
    EXPECT_EQ(pass, true); // 执行成功
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_normal_input_shape_1)
{
    QuantMatmulTestCaseNormal(8, 2048, 1024, 7168);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_normal_input_shape_2)
{
    QuantMatmulTestCaseNormal(16, 1024, 1024, 7168);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_normal_input_rand_shape_0)
{
    QuantMatmulTestCaseNormal(2, 4, 5, 6);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_normal_input_rand_shape_1)
{
    QuantMatmulTestCaseNormal(32, 2432, 4480, 3968);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_normal_input_rand_shape_2)
{
    QuantMatmulTestCaseNormal(16, 1024, 4224, 1024);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_normal_input_rand_shape_3)
{
    QuantMatmulTestCaseNormal(8, 3584, 1152, 1920);
}

/*
 常规 opapi_UT 异常算例1： 空指针
*/
static void QuantMatmulTestCaseNullptr(int inputIndex, int outputIndex)
{
    int64_t b = 2;
    int64_t m = 3;
    int64_t k = 4;
    int64_t n = 5;
    auto x1Desc = TensorDesc({b, m, k}, ACL_INT8, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({b, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto dimsDesc = IntArrayDesc(vector<int64_t>{0});
    auto x2ScaleDesc = TensorDesc({n}, ACL_BF16, ACL_FORMAT_ND);
    auto x1ScaleDesc = TensorDesc({b, m}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({m, n}, ACL_BF16, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;

    if (inputIndex == 0) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulReduceSumWeightNz,
            INPUT(
                nullptr, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
                false, false, -1, dimsDesc, false),
            OUTPUT(outDesc));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    } else if (inputIndex == 1) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulReduceSumWeightNz,
            INPUT(
                x1Desc, nullptr, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
                false, false, -1, dimsDesc, false),
            OUTPUT(outDesc));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    } else if (inputIndex == 2) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulReduceSumWeightNz,
            INPUT(
                x1Desc, x2Desc, nullptr, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
                false, false, -1, dimsDesc, false),
            OUTPUT(outDesc));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    } else if (inputIndex == 3) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulReduceSumWeightNz,
            INPUT(
                x1Desc, x2Desc, x1ScaleDesc, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
                false, false, -1, dimsDesc, false),
            OUTPUT(outDesc));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    } else if (inputIndex == 12) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulReduceSumWeightNz,
            INPUT(
                x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
                false, false, -1, nullptr, false),
            OUTPUT(outDesc));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (outputIndex == 0) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulReduceSumWeightNz,
            INPUT(
                x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
                false, false, -1, dimsDesc, false),
            OUTPUT(nullptr));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x1_empty)
{
    QuantMatmulTestCaseNullptr(0, -1);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_empty)
{
    QuantMatmulTestCaseNullptr(1, -1);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x1_scale_empty)
{
    QuantMatmulTestCaseNullptr(2, -1);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_scale_empty)
{
    QuantMatmulTestCaseNullptr(3, -1);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_out_empty)
{
    QuantMatmulTestCaseNullptr(-1, 0);
}

/*
 常规 opapi_UT 异常算例2： 数据类型不支持
*/
static void QuantMatmulTestCaseInvalidDatatype(int inputIndex, int outputIndex, aclDataType aclDtype)
{
    int64_t b = 2;
    int64_t m = 3;
    int64_t k = 4;
    int64_t n = 5;
    auto x1Desc = TensorDesc({b, m, k}, ACL_INT8, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({b, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto dimsDesc = IntArrayDesc(vector<int64_t>{0});
    auto x2ScaleDesc = TensorDesc({n}, ACL_BF16, ACL_FORMAT_ND);
    auto x1ScaleDesc = TensorDesc({b, m}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({m, n}, ACL_BF16, ACL_FORMAT_ND);

    if (inputIndex == 0) {
        x1Desc = TensorDesc({b, m, k}, aclDtype, ACL_FORMAT_ND);
    } else if (inputIndex == 1) {
        x2Desc = TensorDesc({b, k, n}, aclDtype, ACL_FORMAT_FRACTAL_NZ);
    } else if (inputIndex == 2) {
        x1ScaleDesc = TensorDesc({b, m}, aclDtype, ACL_FORMAT_ND);
    } else if (inputIndex == 3) {
        x2ScaleDesc = TensorDesc({n}, aclDtype, ACL_FORMAT_ND);
    }

    if (outputIndex == 0) {
        outDesc = TensorDesc({m, n}, aclDtype, ACL_FORMAT_ND);
    }

    auto ut = OP_API_UT(
        aclnnQuantMatmulReduceSumWeightNz,
        INPUT(
            x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
            nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
            false, false, -1, dimsDesc, false),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x1_int32)
{
    QuantMatmulTestCaseInvalidDatatype(0, -1, ACL_INT32);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_int32)
{
    QuantMatmulTestCaseInvalidDatatype(1, -1, ACL_INT32);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x1_scale_int32)
{
    QuantMatmulTestCaseInvalidDatatype(2, -1, ACL_INT32);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_scale_int32)
{
    QuantMatmulTestCaseInvalidDatatype(3, -1, ACL_INT32);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_out_int32)
{
    QuantMatmulTestCaseInvalidDatatype(-1, 0, ACL_INT32);
}

/*
 常规 opapi_UT 异常算例3： 数据格式不支持
*/
static void QuantMatmulTestCaseInvalidFormat(int inputIndex, int outputIndex, aclFormat invalidFormat)
{
    int64_t b = 2;
    int64_t m = 3;
    int64_t k = 4;
    int64_t n = 5;
    auto x1Desc = TensorDesc({b, m, k}, ACL_INT8, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({b, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto dimsDesc = IntArrayDesc(vector<int64_t>{0});
    auto x2ScaleDesc = TensorDesc({n}, ACL_BF16, ACL_FORMAT_ND);
    auto x1ScaleDesc = TensorDesc({b, m}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({m, n}, ACL_BF16, ACL_FORMAT_ND);

    if (inputIndex == 0) {
        x1Desc = TensorDesc({b, m, k}, ACL_INT8, invalidFormat);
    } else if (inputIndex == 1) {
        x2Desc = TensorDesc({b, k, n}, ACL_INT8, invalidFormat);
    } else if (inputIndex == 2) {
        x1ScaleDesc = TensorDesc({b, m}, ACL_FLOAT, invalidFormat);
    } else if (inputIndex == 3) {
        x2ScaleDesc = TensorDesc({n}, ACL_BF16, invalidFormat);
    }

    if (outputIndex == 0) {
        outDesc = TensorDesc({m, n}, ACL_BF16, invalidFormat);
    }

    auto ut = OP_API_UT(
        aclnnQuantMatmulReduceSumWeightNz,
        INPUT(
            x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
            nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
            false, false, -1, dimsDesc, false),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x1_NZ)
{
    QuantMatmulTestCaseInvalidFormat(0, -1, ACL_FORMAT_FRACTAL_NZ);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_ND)
{
    QuantMatmulTestCaseInvalidFormat(1, -1, ACL_FORMAT_ND);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_scale_NZ)
{
    QuantMatmulTestCaseInvalidFormat(3, -1, ACL_FORMAT_FRACTAL_NZ);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_out_NZ)
{
    QuantMatmulTestCaseInvalidFormat(-1, 0, ACL_FORMAT_FRACTAL_NZ);
}

/*
 常规 opapi_UT 异常算例4： 数据shape不支持
*/
static void QuantMatmulTestCaseInvalidShape(int inputIndex, int outputIndex)
{
    int64_t b = 2;
    int64_t m = 3;
    int64_t k = 4;
    int64_t n = 5;
    auto x1Desc = TensorDesc({b, m, k}, ACL_INT8, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({b, k, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto dimsDesc = IntArrayDesc(vector<int64_t>{0});
    auto x2ScaleDesc = TensorDesc({n}, ACL_BF16, ACL_FORMAT_ND);
    auto x1ScaleDesc = TensorDesc({b, m}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({m, n}, ACL_BF16, ACL_FORMAT_ND);

    if (inputIndex == 0) {
        x1Desc = TensorDesc({b, m + 1, k}, ACL_INT8, ACL_FORMAT_ND);
    } else if (inputIndex == 1) {
        x2Desc = TensorDesc({b, k + 1, n}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    } else if (inputIndex == 2) {
        x1ScaleDesc = TensorDesc({b, m + 1}, ACL_FLOAT, ACL_FORMAT_ND);
    } else if (inputIndex == 3) {
        x2ScaleDesc = TensorDesc({n + 1}, ACL_BF16, ACL_FORMAT_ND);
    }

    if (outputIndex == 0) {
        outDesc = TensorDesc({m, n + 1}, ACL_BF16, ACL_FORMAT_ND);
    }

    auto ut = OP_API_UT(
        aclnnQuantMatmulReduceSumWeightNz,
        INPUT(
            x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
            nullptr, // nullptr: yScale, x1Offset, x2Offset, yOffset, bias
            false, false, -1, dimsDesc, false),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x1_not_match)
{
    QuantMatmulTestCaseInvalidShape(0, -1);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_not_match)
{
    QuantMatmulTestCaseInvalidShape(1, -1);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x1_scale_not_match)
{
    QuantMatmulTestCaseInvalidShape(2, -1);
}

TEST_F(l2_quant_matmul_reduce_sum_test, ascend910B2_test_x2_scale_not_match)
{
    QuantMatmulTestCaseInvalidShape(3, -1);
}
