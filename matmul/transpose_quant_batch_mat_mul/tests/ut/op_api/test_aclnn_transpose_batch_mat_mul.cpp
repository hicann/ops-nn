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
#include <float.h>
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_transpose_quant_batch_mat_mul.h"

#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_transpose_quant_batch_mat_mul_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_transpose_quant_batch_mat_mul_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_transpose_quant_batch_mat_mul_test TearDown" << endl; }
};

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_01)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t M = 32;
    int64_t K = 512;
    int64_t N = 128;
    int64_t Batch = 16;
    // 1 0 2
    TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    // 0 1 2
    TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    vector<int64_t> perm_x1 = {1, 0, 2};
    vector<int64_t> perm_x2 = {0, 1, 2};
    vector<int64_t> perm_y = {1, 0, 2};
    TensorDesc x1Scale_desc = TensorDesc(
        {
            32,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc(
        {
            128,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto perm_x1_desc = IntArrayDesc(perm_x1);
    auto perm_x2_desc = IntArrayDesc(perm_x2);
    auto perm_y_desc = IntArrayDesc(perm_y);
    int32_t groupSize = 0;
    int32_t dtype = 1; // FP16
    int32_t batch_split_factor = 1;
    TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc,
                              perm_x2_desc, perm_y_desc, batch_split_factor),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_02)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t M = 32;
    int64_t K = 512;
    int64_t N = 128;
    int64_t Batch = 16;
    // 1 0 2
    TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    // 0 1 2
    TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    vector<int64_t> perm_x1 = {1, 0, 2};
    vector<int64_t> perm_x2 = {0, 1, 2};
    vector<int64_t> perm_y = {1, 0, 2};
    TensorDesc x1Scale_desc = TensorDesc(
        {
            32,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc(
        {
            128,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto perm_x1_desc = IntArrayDesc(perm_x1);
    auto perm_x2_desc = IntArrayDesc(perm_x2);
    auto perm_y_desc = IntArrayDesc(perm_y);
    int32_t groupSize = 0;
    int32_t dtype = 1; // FP16
    int32_t batch_split_factor = 1;
    TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc,
                              perm_x2_desc, perm_y_desc, batch_split_factor),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_03)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t M = 32;
    int64_t K = 512;
    int64_t N = 128;
    int64_t Batch = 16;
    // 1 0 2
    TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    // 0 1 2
    TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    vector<int64_t> perm_x1 = {1, 0, 2};
    vector<int64_t> perm_x2 = {0, 1, 2};
    vector<int64_t> perm_y = {1, 0, 2};
    TensorDesc x1Scale_desc = TensorDesc(
        {
            32,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc(
        {
            128,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto perm_x1_desc = IntArrayDesc(perm_x1);
    auto perm_x2_desc = IntArrayDesc(perm_x2);
    auto perm_y_desc = IntArrayDesc(perm_y);
    int32_t groupSize = 0;
    int32_t dtype = 27; // BF16
    int32_t batch_split_factor = 1;
    TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc,
                              perm_x2_desc, perm_y_desc, batch_split_factor),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_04)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t M = 32;
    int64_t K = 512;
    int64_t N = 128;
    int64_t Batch = 16;
    // 1 0 2
    TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    // 0 1 2
    TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    vector<int64_t> perm_x1 = {1, 0, 2};
    vector<int64_t> perm_x2 = {0, 1, 2};
    vector<int64_t> perm_y = {1, 0, 2};
    TensorDesc x1Scale_desc = TensorDesc(
        {
            32,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc(
        {
            128,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto perm_x1_desc = IntArrayDesc(perm_x1);
    auto perm_x2_desc = IntArrayDesc(perm_x2);
    auto perm_y_desc = IntArrayDesc(perm_y);
    int32_t groupSize = 0;
    int32_t dtype = 27; // BF16
    int32_t batch_split_factor = 1;
    TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc,
                              perm_x2_desc, perm_y_desc, batch_split_factor),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_05)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t M = 35;
    int64_t K = 192;
    int64_t N = 744;
    int64_t Batch = 32;
    // 1 0 2
    TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    // 0 1 2
    TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    vector<int64_t> perm_x1 = {1, 0, 2};
    vector<int64_t> perm_x2 = {0, 2, 1};
    vector<int64_t> perm_y = {1, 0, 2};
    TensorDesc x1Scale_desc = TensorDesc({35, 32, 3, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({32, 744, 3, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto perm_x1_desc = IntArrayDesc(perm_x1);
    auto perm_x2_desc = IntArrayDesc(perm_x2);
    auto perm_y_desc = IntArrayDesc(perm_y);
    int32_t groupSize = 0;
    int32_t dtype = 27; // BF16
    int32_t batch_split_factor = 1;
    TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc,
                              perm_x2_desc, perm_y_desc, batch_split_factor),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_success_fp8_expect)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({32, 16, 512}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 512, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 1, 0, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_hifp8_x1scale_null)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({32, 16, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 256, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({64}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, nullptr, x2Scale_desc, 1, 0, p1, p2, py, 1), OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_hifp8_bf16_out)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({32, 16, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 256, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({64}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, nullptr, x2Scale_desc, 27, 0, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_mx_groupsize)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t M = 32;
    int64_t K = 512;
    int64_t N = 128;
    int64_t Batch = 16;
    TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({M, Batch, K / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({Batch, K / 64, N, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_BF16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    int64_t groupSize = 32;
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 27, groupSize, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_mx_weight_nz)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t M = 32;
    int64_t K = 512;
    int64_t N = 128;
    int64_t B = 16;
    int64_t K0 = 16;
    int64_t N0 = 16;
    TensorDesc x1_desc = TensorDesc({M, B, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({B, K, N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ, {}, 0,
                                    {B, N / N0, K / K0, K0, N0});
    TensorDesc x1Scale_desc = TensorDesc({M, B, K / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({B, K / 64, N, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({M, B, N}, ACL_BF16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    int64_t groupSize = 32;
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMulWeightNz,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 27, groupSize, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_SUCCESS);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_invalid_empty)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({0}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 512, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 1, 0, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_invalid_perm_x1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({32, 16, 512}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 512, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 1, 0, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_invalid_bias)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({32, 16, 512}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 512, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, bias_desc, x1Scale_desc, x2Scale_desc, 1, 0, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_invalid_batch_split)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({32, 16, 512}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 512, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 1, 0, p1, p2, py, 2),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_invalid_fp8_k)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc x1_desc = TensorDesc({32, 16, 256}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 256, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 1, 0, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_transpose_quant_batch_mat_mul_test, unsupported_soc)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc x1_desc = TensorDesc({32, 16, 512}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 512, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc x1Scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto p1 = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto p2 = IntArrayDesc(std::vector<int64_t>{0, 1, 2});
    auto py = IntArrayDesc(std::vector<int64_t>{1, 0, 2});
    auto ut = OP_API_UT(aclnnTransposeQuantBatchMatMul,
                        INPUT(x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, 1, 0, p1, p2, py, 1),
                        OUTPUT(out_desc));
    uint64_t ws = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&ws), ACLNN_ERR_PARAM_INVALID);
}
