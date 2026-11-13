/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_host/op_api/aclnn_fused_matmul.h"
#include "op_api/op_api_def_nn.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace std;
using namespace op;

class l2_fusedmatmul_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_fusedmatmul_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_fusedmatmul_test TearDown" << endl; }
};

TEST_F(l2_fusedmatmul_test, dav3510_test_middle_shape_fp16)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_scale_add_mixed_scalar_dtype)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 3, 10}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({2, 10, 4}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
    ScalarDesc alpha_desc = ScalarDesc(2.0F, ACL_FLOAT);
    ScalarDesc beta_desc = ScalarDesc(3.0F, ACL_FLOAT16);
    int8_t cubeMathType = op::USE_HF32;
    auto ut = OP_API_UT(
        aclnnFusedMatmulV2,
        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, alpha_desc, beta_desc, "add", cubeMathType),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_scale_add_bf16_keep_dtype_valid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 3, 10}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({2, 10, 4}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);
    ScalarDesc alpha_desc = ScalarDesc(2.0F, ACL_FLOAT);
    ScalarDesc beta_desc = ScalarDesc(3.0F, ACL_FLOAT);
    int8_t cubeMathType = op::KEEP_DTYPE;
    auto ut = OP_API_UT(
        aclnnFusedMatmulV2,
        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, alpha_desc, beta_desc, "add", cubeMathType),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_scale_add_fp16_fp32_add_valid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 3, 10}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({2, 10, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    ScalarDesc alpha_desc = ScalarDesc(2.0F, ACL_FLOAT);
    ScalarDesc beta_desc = ScalarDesc(3.0F, ACL_FLOAT);
    int8_t cubeMathType = op::USE_FP32_ADD;
    auto ut = OP_API_UT(
        aclnnFusedMatmulV2,
        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, alpha_desc, beta_desc, "add", cubeMathType),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_scale_add_fp32_hf32)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 3, 10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({2, 10, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    ScalarDesc alpha_desc = ScalarDesc(2.0F, ACL_FLOAT);
    ScalarDesc beta_desc = ScalarDesc(3.0F, ACL_FLOAT);
    int8_t cubeMathType = op::USE_HF32;
    auto ut = OP_API_UT(
        aclnnFusedMatmulV2,
        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, alpha_desc, beta_desc, "add", cubeMathType),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_scale_add_fp32_keep_dtype_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 3, 10}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({2, 10, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    ScalarDesc alpha_desc = ScalarDesc(2.0F, ACL_FLOAT);
    ScalarDesc beta_desc = ScalarDesc(3.0F, ACL_FLOAT);
    int8_t cubeMathType = op::KEEP_DTYPE;
    auto ut = OP_API_UT(
        aclnnFusedMatmulV2,
        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, alpha_desc, beta_desc, "add", cubeMathType),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_hf32_bias)
{
    TensorDesc x1_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = op::USE_HF32;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_hf32_relu)
{
    TensorDesc x1_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = op::USE_HF32;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_f32_relu)
{
    TensorDesc x1_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = op::KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bias_f16_f32)
{
    TensorDesc x1_desc = TensorDesc({128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bmm_bias_3d_failed_before_split)
{
    TensorDesc x1_desc = TensorDesc({1, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({1, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_mm_k_zero_add_bias_failed_before_split)
{
    TensorDesc x1_desc = TensorDesc({8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bmm_k_zero_mul_bias_failed_before_split)
{
    TensorDesc x1_desc = TensorDesc({3, 8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({3, 0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({3, 8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3, 8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, x3_desc, "mul", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bmm_k_zero_add_without_bias_success)
{
    TensorDesc x1_desc = TensorDesc({3, 8, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({3, 0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({3, 8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({3, 8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}
TEST_F(l2_fusedmatmul_test, dav3510_test_middle_shape_fp16_failed_1)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 7;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_not_support_bias)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_middle_shape_fp16_failed_3)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "notsupport", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_add_mul_x1_x2_broadcast_failed)
{
    TensorDesc x1_desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    for (const char* fusedOpType : {"add", "mul"}) {
        auto ut = OP_API_UT(aclnnFusedMatmul,
                            INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, fusedOpType, cubeMathType),
                            OUTPUT(out_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_16cast32_x1_x2_broadcast_failed)
{
    TensorDesc x1_desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_y_broadcast_failed)
{
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_add_x3_batch_broadcast_success)
{
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({1, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_mul_x3_matrix_broadcast_success)
{
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "mul", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_add_x3_partial_batch_failed)
{
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({2, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_add_x3_m_broadcast_failed)
{
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_batch_add_x3_n_broadcast_failed)
{
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 16, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_batch_shape_failed)
{
    TensorDesc x1_desc = TensorDesc({1, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "gelu_erf", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_erf_fp16_success)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "gelu_erf", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_tanh_fp16_success)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "gelu_tanh", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_erf_bias_fp16_success)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "gelu_erf", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_tanh_bias_bf16_fp32_success)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "gelu_tanh", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_erf_fp16_bias_bf16_2d_failed)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "gelu_erf", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_erf_fp16_bias_fp32_2d_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "gelu_erf", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_tanh_bf16_bias_bf16_2d_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "gelu_tanh", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_tanh_bf16_bias_fp32_2d_v2_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmulV2,
                        INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, (aclScalar*)nullptr,
                              (aclScalar*)nullptr, "gelu_tanh", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_gelu_fp32_failed)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = op::USE_HF32;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "gelu_erf", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_fused_op_type_nullptr)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (char*)nullptr, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_add_x3_nullptr)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "add", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_dtype_mismatch_x1_x2)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_dtype_mismatch_y)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_nz_format_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 16});
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_relu_3d_2d_fp16_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_dim_mismatch_x_y)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_relu_shared_x2_batch_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_relu_3d_2d_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({50, 1, 448}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({448, 896}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({896}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({50, 1, 896}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", op::USE_HF32),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend910b_test_relu_3d_2d_failed)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x1_desc = TensorDesc({50, 1, 448}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({448, 896}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({50, 1, 896}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bmm_relu_fp16)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 512, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bmm_relu_with_bias)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({2, 256, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bmm_empty_fp16)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 512, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bmm_empty_with_bias)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 128, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({2, 256, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_relu_4d_2d_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 25, 1, 448}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({448, 896}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({896}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 25, 1, 896}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", op::USE_HF32),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_relu_shared_x2_4d_5d_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({2, 25, 1, 448}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({1, 1, 1, 448, 896}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({896}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 2, 25, 1, 896}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", op::USE_HF32),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_x3_mm_shape_mismatch)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_16cast32_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_16cast32_y_dtype_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_empty_op_type_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_bias_nz_format_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_x3_nz_format_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 4, 16, 16});
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "add", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_inner_precise_high_precision)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = op::USE_FP32_ADD;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_split_small_n_add)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_split_small_m_mul)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({8, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "mul", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_split_k_equal_1)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({32, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_split_x3_no_batch_2d)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_split_x3_no_batch_3d_batch1)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({1, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "mul", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_split_bmm_2d_add)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({4, 8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_empty_tensor_16cast32)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_empty_output_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({0, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_INNER_NULLPTR);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_transposed_x2_bias_uses_logical_n)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_BF16, ACL_FORMAT_ND, {1, 32}, 0, {64, 32});
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    for (auto bias_dtype : {ACL_BF16, ACL_FLOAT}) {
        TensorDesc bias_desc = TensorDesc({1, 64}, bias_dtype, ACL_FORMAT_ND);
        ScalarDesc one_desc = ScalarDesc(1.0F, ACL_FLOAT);
        auto ut = OP_API_UT(aclnnFusedMatmulV2,
                            INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, one_desc, one_desc, "sigmoid", 0),
                            OUTPUT(out_desc));
        uint64_t workspace_size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
    }
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_empty_inputs_failed)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    for (const auto& shape : vector<array<int64_t, 3>>{{0, 32, 64}, {4, 0, 64}, {4, 32, 0}}) {
        TensorDesc x1_desc = TensorDesc({shape[0], shape[1]}, ACL_FLOAT16, ACL_FORMAT_ND);
        TensorDesc x2_desc = TensorDesc({shape[1], shape[2]}, ACL_FLOAT16, ACL_FORMAT_ND);
        TensorDesc out_desc = TensorDesc({shape[0], shape[2]}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnFusedMatmul,
                            INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "sigmoid", 0),
                            OUTPUT(out_desc));
        uint64_t workspace_size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_INNER_NULLPTR);
    }
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_bias_shape)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    const vector<vector<int64_t>> bias_shapes = {{64}, {1, 64}, {64, 64}, {63}, {65}};
    for (size_t i = 0; i < bias_shapes.size(); ++i) {
        SCOPED_TRACE(i);
        TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
        TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
        TensorDesc bias_desc = TensorDesc(bias_shapes[i], ACL_FLOAT, ACL_FORMAT_ND);
        TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnFusedMatmulV2,
                            INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, (aclScalar*)nullptr,
                                  (aclScalar*)nullptr, "sigmoid", 0),
                            OUTPUT(out_desc));
        uint64_t workspace_size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), i < 2 ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_fp16_bf16_without_bias_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    for (auto dtype : {ACL_FLOAT16, ACL_BF16}) {
        TensorDesc x1_desc = TensorDesc({4, 32}, dtype, ACL_FORMAT_ND);
        TensorDesc x2_desc = TensorDesc({32, 64}, dtype, ACL_FORMAT_ND);
        TensorDesc out_desc = TensorDesc({4, 64}, dtype, ACL_FORMAT_ND);
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(aclnnFusedMatmul,
                            INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "sigmoid", 0),
                            OUTPUT(out_desc));
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
        auto ut_v2 = OP_API_UT(aclnnFusedMatmulV2,
                               INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr,
                                     (aclScalar*)nullptr, "sigmoid", 0),
                               OUTPUT(out_desc));
        EXPECT_EQ(ut_v2.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
    }
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_x3_failed)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x3_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, "sigmoid", 0),
                        OUTPUT(out_desc));
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
    auto ut_v2 = OP_API_UT(
        aclnnFusedMatmulV2,
        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, x3_desc, (aclScalar*)nullptr, (aclScalar*)nullptr, "sigmoid", 0),
        OUTPUT(out_desc));
    EXPECT_EQ(ut_v2.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_batch_shape_failed)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({1, 4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmulV2,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclScalar*)nullptr,
                              (aclScalar*)nullptr, "sigmoid", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_fp32_failed)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "sigmoid", op::USE_HF32),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, dav3510_test_sigmoid_nondefault_scale_failed)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    for (const auto& scales : vector<array<float, 2>>{{2.0F, 1.0F}, {1.0F, 2.0F}}) {
        TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
        TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
        TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
        ScalarDesc alpha_desc = ScalarDesc(scales[0], ACL_FLOAT);
        ScalarDesc beta_desc = ScalarDesc(scales[1], ACL_FLOAT);
        auto ut = OP_API_UT(
            aclnnFusedMatmulV2,
            INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, alpha_desc, beta_desc, "sigmoid", 0),
            OUTPUT(out_desc));
        uint64_t workspace_size = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_fusedmatmul_test, ascend960dt_test_empty_op_type_success)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend960dt_test_relu_with_bias_success)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend960dt_test_16cast32_success)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend960dt_test_16cast32_y_dtype_invalid)
{
    NpuArchManager archManager(NpuArch::DAV_9201);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, ascend960pr_test_empty_op_type_success)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend960pr_test_relu_with_bias_success)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend960pr_test_16cast32_success)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend960pr_test_16cast32_y_dtype_invalid)
{
    NpuArchManager archManager(NpuArch::DAV_9202);
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedMatmul,
                        INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, "16cast32", 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
