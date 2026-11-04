/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_gemm.h"
#include "op_api/op_api_def_nn.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace std;
using namespace op;

class l2_gemm_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "gemm_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "gemm_test TearDown" << endl; }

    void test_run(vector<int64_t> ADims, aclDataType ADtype, aclFormat AFormat, vector<int64_t> ARange,
                  vector<int64_t> BDims, aclDataType BDtype, aclFormat BFormat, vector<int64_t> BRange,
                  vector<int64_t> CDims, aclDataType CDtype, aclFormat CFormat, vector<int64_t> CRange,
                  vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat, float alpha, float beta,
                  int64_t transA, int64_t transB, int8_t cubeMathType)
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

    void test_run_inval(vector<int64_t> ADims, aclDataType ADtype, aclFormat AFormat, vector<int64_t> ARange,
                        vector<int64_t> BDims, aclDataType BDtype, aclFormat BFormat, vector<int64_t> BRange,
                        vector<int64_t> CDims, aclDataType CDtype, aclFormat CFormat, vector<int64_t> CRange,
                        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat, float alpha, float beta,
                        int64_t transA, int64_t transB, int8_t cubeMathType)
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

// 正常流程，支持的数据类型
TEST_F(l2_gemm_test, case_various_dtype_valid)
{
    float alpha = 2.0;
    float beta = 2.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    test_run({16, 16}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {16, 16},
             ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB,
             cubeMathType);
    test_run({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16},
             ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, alpha, beta, transA, transB,
             cubeMathType);
}

// 正常流程，支持的数据类型
TEST_F(l2_gemm_test, case_various_dtype_valid2)
{
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    test_run({16, 16}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {16, 16},
             ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB,
             cubeMathType);
    test_run({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16},
             ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, alpha, beta, transA, transB,
             cubeMathType);
}

// 异常流程，不支持的数据类型
TEST_F(l2_gemm_test, case_various_dtype_invalid)
{
    float alpha = 2.0;
    float beta = 2.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    test_run_inval({16, 16}, ACL_INT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_INT16, ACL_FORMAT_ND, {0, 2}, {16, 16},
                   ACL_INT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_INT16, ACL_FORMAT_ND, alpha, beta, transA, transB,
                   cubeMathType);
    test_run_inval({16, 16}, ACL_UINT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_UINT16, ACL_FORMAT_ND, {0, 2}, {16, 16},
                   ACL_UINT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_UINT16, ACL_FORMAT_ND, alpha, beta, transA, transB,
                   cubeMathType);
}

// 正常流程，有转置的情况
TEST_F(l2_gemm_test, case_various_transpose_valid)
{
    test_run({4, 3}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {3, 5}, ACL_FLOAT,
             ACL_FORMAT_ND, {0, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 1, 0, 1);
    test_run({3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {3, 5}, ACL_FLOAT,
             ACL_FORMAT_ND, {0, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 0, 1, 1);
    test_run({4, 3}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {0, 2}, {3, 5}, ACL_FLOAT,
             ACL_FORMAT_ND, {0, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 1, 1, 1);
}

// beta == 0 场景
TEST_F(l2_gemm_test, case_beta_0)
{
    auto A = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    float alpha = 2.0;
    float beta = 0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// beta == 1 场景
TEST_F(l2_gemm_test, case_beta_1)
{
    auto A = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    float alpha = 2.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
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

// 非连续
TEST_F(l2_gemm_test, case_not_contiguous)
{
    auto A = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {1, 16}, 0, {32, 16}).ValueRange(0, 2);
    auto B = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 32}, 0, {16, 32}).ValueRange(0, 2);
    auto C = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
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

// 校验output shape要与预期一致
TEST_F(l2_gemm_test, case_output_shape)
{
    float alpha = 2.0;
    float beta = 2.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    // A: 2 x 5, B: 5 x 3, C: 2 x 3, out: 2 x 3
    test_run({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 3}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {2, 3}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);

    // A: 2 x 5, B: 5 x 3, C: 2 x 3, out: 2 x 4
    test_run_inval({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 3},
                   ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 4}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB,
                   cubeMathType);
}

TEST_F(l2_gemm_test, test_hf32_trans)
{
    auto A = TensorDesc({14, 13}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto B = TensorDesc({15, 14}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto C = TensorDesc({13, 15}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out = TensorDesc({13, 15}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 1;
    int64_t transB = 1;
    int8_t cubeMathType = 3;

    auto ut = OP_API_UT(aclnnGemm, INPUT(A, B, C, alpha, beta, transA, transB), OUTPUT(out), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
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
    test_run({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);

    // C: 1 / 1 x 1, A: 2 x 0, B: 0 x 2, out: 2 x 2
    test_run({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    test_run({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1, 1}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);

    // C: 0, A: 2 x 0, B: 0 x 2, out: 2 x 2  拦截报错
    test_run_inval({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 2}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0}, ACL_FLOAT,
                   ACL_FORMAT_ND, {-5, 5}, {2, 2}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);

    // C: 2 x 0, A: 2 x 0, B: 0 x 0, out: 0 x 0
    test_run({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {2, 0}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    // C: 0 / 1 / 1 x 1, A: 2 x 0, B: 0 x 0, out: 0 x 0
    test_run({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    test_run({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
    test_run({2, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, {-5, 5}, {1, 1}, ACL_FLOAT,
             ACL_FORMAT_ND, {-5, 5}, {0, 0}, ACL_FLOAT, ACL_FORMAT_ND, alpha, beta, transA, transB, cubeMathType);
}

// cubeMathType = USE_FP32_ADD 测试用例
TEST_F(l2_gemm_test, case_cubeMathType_USE_FP32_ADD)
{
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = USE_FP32_ADD;
    test_run({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16},
             ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, alpha, beta, transA, transB,
             cubeMathType);
}

// cubeMathType = KEEP_DTYPE 测试用例
TEST_F(l2_gemm_test, case_cubeMathType_KEEP_DTYPE)
{
    float alpha = 1.0;
    float beta = 1.0;
    int64_t transA = 0;
    int64_t transB = 0;
    int8_t cubeMathType = KEEP_DTYPE;
    test_run({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16},
             ACL_FLOAT16, ACL_FORMAT_ND, {0, 2}, {16, 16}, ACL_FLOAT16, ACL_FORMAT_ND, alpha, beta, transA, transB,
             cubeMathType);
}

// 950路由Addmm：覆盖transA/transB转换为tensor view的四种组合。
TEST_F(l2_gemm_test, gemm_dav3510_route_addmm_transpose_view)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    test_run({3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT,
             ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 0, 0, KEEP_DTYPE);
    test_run({4, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT,
             ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 1, 0, KEEP_DTYPE);
    test_run({3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT,
             ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 0, 1, KEEP_DTYPE);
    test_run({4, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT,
             ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 1, 1, KEEP_DTYPE);
}

// 950路由Addmm：alpha为0时选择Alpha0计算图。
TEST_F(l2_gemm_test, gemm_dav3510_route_addmm_alpha_zero)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    test_run({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5},
             ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT16, ACL_FORMAT_ND, 0.0, 2.0, 0, 0, KEEP_DTYPE);
}

// 950路由Addmm：beta为0时选择Beta0计算图。
TEST_F(l2_gemm_test, gemm_dav3510_route_addmm_beta_zero)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    test_run({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5},
             ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT16, ACL_FORMAT_ND, 2.0, 0.0, 0, 0, KEEP_DTYPE);
}

// 950路由Addmm：C为[1, N]且alpha、beta均为1时覆盖bias融合候选图。
TEST_F(l2_gemm_test, gemm_dav3510_route_addmm_bias)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    test_run({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {1, 5},
             ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT16, ACL_FORMAT_ND, 1.0, 1.0, 0, 0, KEEP_DTYPE);
}

// 950路由Addmm：低精度输入、FLOAT32输出覆盖原生高精度累加路径。
TEST_F(l2_gemm_test, gemm_dav3510_route_addmm_fp16_bf16_in_fp32_out)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    test_run({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5},
             ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 0, 0, KEEP_DTYPE);
    test_run({3, 4}, ACL_BF16, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_BF16, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_BF16,
             ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT, ACL_FORMAT_ND, 1.0, 1.0, 0, 0, KEEP_DTYPE);
}

// 950路由Addmm：低精度输出使用USE_FP32_ADD覆盖通用高精度计算图。
TEST_F(l2_gemm_test, gemm_dav3510_route_addmm_use_fp32_add)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    test_run({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {4, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5},
             ACL_FLOAT16, ACL_FORMAT_ND, {-2, 2}, {3, 5}, ACL_FLOAT16, ACL_FORMAT_ND, 2.0, 3.0, 0, 0, USE_FP32_ADD);
}

#include <cstdint>
#include <dlfcn.h>

#include "gtest/gtest.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "level0/padv3.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

// Keep ErrorManager declarations in global scope when including cube_util.cpp below.
#include "base/err_msg.h"

namespace gemm_util_whitebox {

namespace l0op {
__attribute__((visibility("hidden"))) const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype,
                                                            aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Fill(const aclTensor* dims, const aclTensor* value,
                                                            const aclIntArray* outShape, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Mul(const aclTensor* self, const aclTensor* other,
                                                           aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* ReFormat(const aclTensor* x, const op::Format& format,
                                                                aclOpExecutor* executor = nullptr);
__attribute__((visibility("hidden"))) const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape,
                                                               aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape,
                                                               aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* TransData(const aclTensor* x, op::Format dstPrimaryFormat,
                                                                 int64_t groups, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Transpose(const aclTensor* x, const aclIntArray* perm,
                                                                 aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y,
                                                                aclOpExecutor* executor);
__attribute__((visibility("hidden"))) bool MmCheckHitV3Shape(const aclTensor* x1, const aclTensor* x2,
                                                             const aclTensor* bias, const bool transposeX1,
                                                             const bool transposeX2, op::Format mat2_format,
                                                             bool supportSplitK);
} // namespace l0op

#include "../../../../common/op_host/op_api/cube_util.cpp"
#include "../../../../common/op_host/op_api/matmul.cpp"
#include "../../../../common/op_host/op_api/matmul_util.cpp"

namespace l0op {
const aclTensor* Cast(const aclTensor* self, op::DataType, aclOpExecutor*) { return self; }
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor*) { return x; }
const aclTensor* Fill(const aclTensor*, const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* Mul(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
const aclTensor* ReFormat(const aclTensor* x, const op::Format& format, aclOpExecutor*)
{
    return (x != nullptr && x->GetStorageFormat() == format) ? x : nullptr;
}
const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)
{
    op::Shape out = shape;
    int64_t known = 1;
    int64_t inferIdx = -1;
    for (size_t i = 0; i < out.GetDimNum(); i++) {
        if (out.GetDim(i) == -1) {
            inferIdx = static_cast<int64_t>(i);
        } else {
            known *= out.GetDim(i);
        }
    }
    if (inferIdx >= 0) {
        out.SetDim(inferIdx, x->GetViewShape().GetShapeSize() / known);
    }
    return executor->CreateView(x, out, 0);
}
const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape, aclOpExecutor* executor)
{
    using ReshapeFn = const aclTensor* (*)(const aclTensor*, const aclIntArray*, aclOpExecutor*);
    static ReshapeFn realReshape = reinterpret_cast<ReshapeFn>(
        dlsym(RTLD_DEFAULT, "_ZN4l0op7ReshapeEPK9aclTensorPK11aclIntArrayP13aclOpExecutor"));
    if (realReshape == nullptr) {
        return nullptr;
    }
    return realReshape(x, shape, executor);
}
const aclTensor* TransData(const aclTensor*, op::Format, int64_t, aclOpExecutor*) { return nullptr; }
const aclTensor* Transpose(const aclTensor* x, const aclIntArray*, aclOpExecutor*) { return x; }
const aclTensor* ViewCopy(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
bool MmCheckHitV3Shape(const aclTensor*, const aclTensor*, const aclTensor*, bool, bool, op::Format, bool)
{
    return false;
}
} // namespace l0op

} // namespace gemm_util_whitebox

namespace gemm_util_whitebox {
const aclTensor* ExecProcessEmptyTensorWithTrans(const aclTensor* self, const aclTensor* mat2, int64_t transSelf,
                                                 int64_t transMat2, aclOpExecutor* executor)
{
    return ProcessEmptyTensorWithTrans(self, mat2, transSelf, transMat2, executor);
}
} // namespace gemm_util_whitebox

using namespace op;

namespace {

constexpr int8_t kCubeMathKeepDtype = 0;

aclTensor* MakeTensor(aclOpExecutor* executor, const std::initializer_list<int64_t>& dims, op::DataType dtype,
                      op::Format format)
{
    return executor->AllocTensor(op::Shape(dims), dtype, format);
}

} // namespace

class GemmUtilWhiteboxTest : public testing::Test {};

TEST_F(GemmUtilWhiteboxTest, ProcessEmptyTensorWithTrans_EmptyOut)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {4, 0}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {0, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);

    auto* out = gemm_util_whitebox::ExecProcessEmptyTensorWithTrans(self, mat2, 1, 1, executor);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape().GetDimNum(), 2);
    EXPECT_EQ(out->GetViewShape().GetDim(0), 0);
    EXPECT_EQ(out->GetViewShape().GetDim(1), 0);
    EXPECT_TRUE(out->IsEmpty());
}

TEST_F(GemmUtilWhiteboxTest, ProcessEmptyTensorWithTrans_NonEmptyOut)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* self = MakeTensor(executor, {4, 0}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {0, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);

    auto* out = gemm_util_whitebox::ExecProcessEmptyTensorWithTrans(self, mat2, 0, 0, executor);
    EXPECT_EQ(out, nullptr);
}

class GemmV3UtilWhiteboxTest : public testing::Test {};

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_ShapeInvalid)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {4, 8}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {9, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {4, 4}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_FALSE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_EmptyTensor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {4, 8}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {8, 0}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {4, 0}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_FALSE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_NzFormat)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {1024, 512}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    const aclTensor* mat2 = MakeTensor(executor, {512, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {1024, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_FALSE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_UnsupportedArch)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {1024, 512}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {512, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {1024, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_FALSE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_Dav2201DtypeReject)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {1024, 512}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {512, 1024}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {1024, 1024}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_FALSE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_Dav2201ShapeTooSmall)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {256, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {128, 256}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {256, 256}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_FALSE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_Dav2201Supported)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {1024, 512}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {512, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {1024, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_TRUE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_Bf16Supported)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {1024, 512}, op::DataType::DT_BF16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {512, 1024}, op::DataType::DT_BF16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {1024, 1024}, op::DataType::DT_BF16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_TRUE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}

TEST_F(GemmV3UtilWhiteboxTest, CheckGemmV3Support_Arch3510Supported)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    const aclTensor* mat1 = MakeTensor(executor, {1024, 512}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* mat2 = MakeTensor(executor, {512, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    const aclTensor* out = MakeTensor(executor, {1024, 1024}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(mat1, nullptr);
    ASSERT_NE(mat2, nullptr);
    ASSERT_NE(out, nullptr);
    gemm_util_whitebox::Ops::NN::MmOpInfo mmOpInfo{};

    EXPECT_TRUE(
        gemm_util_whitebox::Ops::NN::CheckGemmV3Support(mat1, mat2, nullptr, out, mmOpInfo, kCubeMathKeepDtype));
}
