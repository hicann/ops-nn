/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "op_api/op_api_def.h"
#include "opdev/platform.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_matmulWeightNz_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_matmul_weight_nz_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_matmul_weight_nz_test TearDown" << endl;
    }
    static void MatMulCommonTest(TensorDesc a_desc, TensorDesc b_desc, TensorDesc out_desc,
                                         aclnnStatus expect_status, int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION)
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

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_fp16_x2_not_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {}, 0, {32, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_fp16_out_fp32)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_bfp16_weight_nd)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_bf16_out_fp32)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_fp32_weight_nd)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).ValueRange(0, 2);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_invalid_dtype)
{
    TensorDesc a2_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND);
    TensorDesc b2_desc = TensorDesc({16, 32}, ACL_BOOL, ACL_FORMAT_ND, {}, 0, {2, 1, 16, 16});
    TensorDesc out2_desc = TensorDesc({16, 16}, ACL_BOOL, ACL_FORMAT_ND);
    MatMulCommonTest(a2_desc, b2_desc, out2_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_bf16_out_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_fp32_out_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmulWeightNz_test, ascend910_95_test_aligned_fp16_out_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACL_SUCCESS);
}

TEST_F(l2_matmulWeightNz_test, ascend910B2_test_aligned_fp32_not_support_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

// 接口整改异常用例 - 910_95
TEST_F(l2_matmulWeightNz_test, matmul_NZ_910_95_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_910_95_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_910_95_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    MatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_matmulWeightNz_test, matmul_NZ_910_95_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
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