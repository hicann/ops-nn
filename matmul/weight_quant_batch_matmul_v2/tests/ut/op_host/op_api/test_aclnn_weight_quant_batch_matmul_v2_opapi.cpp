/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_weight_quant_batch_matmul_v2.h"
#include "../../../../op_host/op_api/aclnn_weight_quant_batch_matmul_v3.h"
#include "../../../../op_host/op_api/aclnn_weight_quant_batch_matmul_nz.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_weight_quant_batch_matmul_v2_opapi_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_weight_quant_batch_matmul_v2_opapi_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_weight_quant_batch_matmul_v2_opapi_test TearDown" << endl; }
};

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_v2_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc antiquantOffset = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, antiquantOffset, nullptr, nullptr, bias, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_v2_invalid_dtype)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_v2_empty_x_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({0, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({0, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_TRUE(ret == ACLNN_ERR_PARAM_INVALID || ret == ACLNN_SUCCESS);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_v3_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV3,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    auto ret = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_TRUE(ret == ACLNN_SUCCESS || ret == ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_nz_invalid_weight_nd)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulNz,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_fp8_pergroup_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 128), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_bf16_uint64_scale_fixpipe_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_scale_dtype_mismatch)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_bf16_bias_wrong_dtype)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, bias, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_quant_offset_without_scale)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc quantOffset = TensorDesc({32}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, quantOffset, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_nz_func_wrong_platform)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulNz,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_RUNTIME_ERROR);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_int4_nz_transposed)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({32, 64}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 1), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_int8_nz_large_k)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 65536}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({32, 65536}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend910B_uint64_scale_y_not_fp16)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    TensorDesc x = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({64, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc quantScale = TensorDesc({32}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, quantScale, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_int8_nz_bias_wrong_dtype)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({4, 2, 16, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc antiquantScale = TensorDesc({1, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, bias, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_fp8_pergroup_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({128, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 64), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_int8_nz_pergroup_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({128, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc antiquantScale = TensorDesc({2, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulNz,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 64), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_int8_nz_pertensor_scale_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({128, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc antiquantScale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulNz,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_fp8_nz_weight_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({128, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc antiquantScale = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulNz,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_int8_nz_bf16_bias_invalid)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({128, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    TensorDesc antiquantScale = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc bias = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulNz,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, bias, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_float_weight_preprocess)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({128, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_weight_quant_batch_matmul_v2_opapi_test, ascend950_float_weight_uint8_scale_preprocess)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
    NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x = TensorDesc({16, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc weight = TensorDesc({128, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc antiquantScale = TensorDesc({64}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc y = TensorDesc({16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnWeightQuantBatchMatmulV2,
                        INPUT(x, weight, antiquantScale, nullptr, nullptr, nullptr, nullptr, 0), OUTPUT(y));
    uint64_t workspace_size = 0;
    ut.TestGetWorkspaceSize(&workspace_size);
}
