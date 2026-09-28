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

#include "../../../op_host/op_api/aclnn_matmul.h"
#include "op_api/op_api_def_nn.h"
#include "opdev/platform.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_matmulWeightNz_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_matmul_weight_nz_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_matmul_weight_nz_test TearDown" << endl; }
    static void MatMulCommonTest(TensorDesc a_desc, TensorDesc b_desc, TensorDesc out_desc, aclnnStatus expect_status,
                                 int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION)
    {
        auto ut = OP_API_UT(aclnnMatmulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cubeMathType);

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expect_status);
        // SAMPLE: precision simulate
        if (expect_status == ACL_SUCCESS) {
            // ut.TestPrecision();  // soc version  2. 二段接口
        }
    }
};

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_fp16_x2_not_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {}, 0, {32, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_fp16_out_fp32)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_bfp16_weight_nd)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_bf16_out_fp32)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_fp32_weight_nd)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_invalid_dtype)
{
    TensorDesc a2_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc b2_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out2_desc = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    MatMulCommonTest(a2_desc, b2_desc, out2_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_bf16_out_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_bf16_fp32_out_weight_nz)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS, KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_fp32_out_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_fp16_out_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmulWeightNz_test, ascend950_test_aligned_fp16_fp32_out_weight_nz)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS, KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, ascend910B2_test_aligned_fp32_not_support_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

// 接口整改异常用例 - 950
TEST_F(l2_matmulWeightNz_test, matmul_NZ_950_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_950_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_950_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_950_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

// 接口整改异常用例 - 310
TEST_F(l2_matmulWeightNz_test, matmul_NZ_310_FP32_FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_310_FP32_FP16_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_310_FP32_FP32_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, USE_HF32);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_310_FP32_FP16_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, USE_HF32);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_310_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_310_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

// out数据类型需与self、mat2推导之后的数据类型满足推导规则，FP16×FP16推导为FP16，out为BF16应报错
TEST_F(l2_matmulWeightNz_test, matmul_NZ_910B_FP16_FP16_out_BF16_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

// out数据类型需与self、mat2推导之后的数据类型满足推导规则，BF16×BF16推导为BF16，out为FP16应报错
TEST_F(l2_matmulWeightNz_test, matmul_NZ_910B_BF16_BF16_out_FP16_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

// 16进32出：FP16×FP16推导为FP16，out为FP32属设计允许场景，不应被一致性校验拦截
TEST_F(l2_matmulWeightNz_test, matmul_NZ_910B_FP16_FP16_out_FP32_valid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

// mat2的view stride非标准布局（[1,K_full]且K_full != K），需拒绝（issue 6020）
TEST_F(l2_matmulWeightNz_test, matmul_NZ_nonstandard_stride_rejected)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    // view [K,N]=[32,16]，stride [1,K_full]=[1,40]，既非[1,K]也非[N,1]
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {1, 40}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

// k1 == n1 时storage无法区分转置朝向，非标准stride同样需拒绝
TEST_F(l2_matmulWeightNz_test, matmul_NZ_nonstandard_stride_k1_eq_n1_rejected)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    // K=32, N=20 -> k1 = n1 = 2
    TensorDesc b_desc = TensorDesc({32, 20}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {1, 40}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 20}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

// 标准转置stride [1,K] 不应被误拒
TEST_F(l2_matmulWeightNz_test, matmul_NZ_transpose_stride_ok)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {1, 32}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

// 标准非转置stride [N,1] 不应被误拒
TEST_F(l2_matmulWeightNz_test, matmul_NZ_nontranspose_stride_ok)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {16, 1}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}
