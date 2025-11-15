/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_gemm.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
using namespace op;

class l2_gemm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "gemm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "gemm_test TearDown" << endl;
    }

    void test_run(
        vector<int64_t> ADims, aclDataType ADtype, aclFormat AFormat, vector<int64_t> ARange, vector<int64_t> BDims,
        aclDataType BDtype, aclFormat BFormat, vector<int64_t> BRange, vector<int64_t> CDims, aclDataType CDtype,
        aclFormat CFormat, vector<int64_t> CRange, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat,
        float alpha, float beta, int64_t transA, int64_t transB, int8_t cubeMathType)
    {
        auto A = TensorDesc(ADims, ADtype, AFormat).ValueRange(ARange[0], ARange[1]);
        auto B = TensorDesc(BDims, BDtype, BFormat).ValueRange(BRange[0], BRange[1]);
        auto C = TensorDesc(CDims, CDtype, CFormat).ValueRange(CRange[0], CRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    }

    void test_run_inval(
        vector<int64_t> ADims, aclDataType ADtype, aclFormat AFormat, vector<int64_t> ARange, vector<int64_t> BDims,
        aclDataType BDtype, aclFormat BFormat, vector<int64_t> BRange, vector<int64_t> CDims, aclDataType CDtype,
        aclFormat CFormat, vector<int64_t> CRange, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat,
        float alpha, float beta, int64_t transA, int64_t transB, int8_t cubeMathType)
    {
        auto A = TensorDesc(ADims, ADtype, AFormat).ValueRange(ARange[0], ARange[1]);
        auto B = TensorDesc(BDims, BDtype, BFormat).ValueRange(BRange[0], BRange[1]);
        auto C = TensorDesc(CDims, CDtype, CFormat).ValueRange(CRange[0], CRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

// 异常流程，不支持的数据类型
TEST_F(l2_gemm_test, case_various_dtype_invalid)
{
    float alpha = 2.0;
    float beta = 2.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    test_run_inval(
        {16, 16}, ACL_INT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_INT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_INT16,
        ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_INT16, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    test_run_inval(
        {16, 16}, ACL_UINT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_UINT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_UINT16,
        ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_UINT16, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
}

// C 空指针
TEST_F(l2_gemm_test, case_empty_tensor_input_1)
{
    auto A = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, nullptr, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// A空指针
TEST_F(l2_gemm_test, case_empty_tensor_input_2)
{
    auto B = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(nullptr, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// B空指针
TEST_F(l2_gemm_test, case_empty_tensor_input_3)
{
    auto A = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, nullptr, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// out空指针
TEST_F(l2_gemm_test, case_empty_tensor_out)
{
    auto A = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(nullptr), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);
}

// A不是2维
TEST_F(l2_gemm_test, case_A_dim_not_2)
{
    auto A = TensorDesc({16, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// A, B不满足相乘条件
TEST_F(l2_gemm_test, case_cannot_matmul)
{
    auto A = TensorDesc({16, 17}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// C 和 A@B无法broadcast
TEST_F(l2_gemm_test, case_cannot_axpy)
{
    auto A = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({17, 17}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// C 和 A@B无法broadcast
TEST_F(l2_gemm_test, case_cannot_axpy2)
{
    auto A = TensorDesc({14, 13}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({15, 14}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({17, 17}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 1;
    int64_t transB = 1;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor1
TEST_F(l2_gemm_test, case_empty_tensor_1)
{
    auto A = TensorDesc({4, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor2
TEST_F(l2_gemm_test, case_empty_tensor_2)
{
    auto A = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({1, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_gemm_test, test_hf32_trans)
{
    auto A = TensorDesc({14, 13}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({15, 14}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({13, 15}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({15, 15}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 1;
    int64_t transB = 1;
    int8_t cubeMathType = 3;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor混合场景
TEST_F(l2_gemm_test, case_empty_tensor_3)
{
    float alpha = 2.0;
    float beta = 2.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    // C: 2 x 2, A: 2 x 0, B: 0 x 2, out: 2 x 2
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);

    // C: 1 / 1 x 1, A: 2 x 0, B: 0 x 2, out: 2 x 2
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1, 1}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);

    // C: 0, A: 2 x 0, B: 0 x 2, out: 2 x 2  拦截报错
    test_run_inval(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);

    // C: 2 x 0, A: 2 x 0, B: 0 x 0, out: 0 x 0
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    // C: 0 / 1 / 1 x 1, A: 2 x 0, B: 0 x 0, out: 0 x 0
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    test_run(
        {2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1, 1}, ACL_FLOAT,
        ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
}