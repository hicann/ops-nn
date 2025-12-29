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

#include "../../../op_host/op_api/aclnn_batch_matmul.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api/op_api_def.h"

using namespace std;
using namespace op;
class l2_batch_matmul_weight_nz_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "batch_matmul_weight_nz_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "batch_matmul_weight_nz_test TearDown" << endl;
    }
    static void BatchMatMulCommonTest(TensorDesc a_desc, TensorDesc b_desc, TensorDesc out_desc,
                                         aclnnStatus expect_status, int8_t cubeMathType = KEEP_DTYPE)
    {
        auto ut = OP_API_UT(aclnnBatchMatMulWeightNz, INPUT(a_desc, b_desc), OUTPUT(out_desc), cubeMathType);

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

TEST_F(l2_batch_matmul_weight_nz_test, ascend910B_test_aligned_fp32_out_weight_nd)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_weight_nz_test, ascend910B_test_aligned_fp16_out_weight_nd)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_weight_nz_test, ascend910B_test_aligned_bf16_out_weight_nd)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_BF16, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_weight_nz_test, ascend910B_test_aligned_fp32_out_weight_nz)
{
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({32, 16}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 1, 16, 16});
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID);
}

// 接口整改异常用例 910_95
TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_910_95_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_910_95_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_910_95_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_910_95_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_910_95_BF16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910_95);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_BF16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

// 接口整改异常用例 - 310
TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_310_FP32_FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, KEEP_DTYPE);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_310_FP32_FP16_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, KEEP_DTYPE);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_310_FP32_FP32_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, USE_HF32);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_310_FP32_FP16_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, USE_HF32);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_310_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}

TEST_F(l2_batch_matmul_weight_nz_test, batch_matmul_NZ_310_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    TensorDesc a_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 32, 32}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 2, 16, 16});
    TensorDesc out_desc = TensorDesc({2, 32, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    BatchMatMulCommonTest(a_desc, b_desc, out_desc, ACLNN_ERR_PARAM_INVALID, FP16FP32_KEEP_DTYPE);
}