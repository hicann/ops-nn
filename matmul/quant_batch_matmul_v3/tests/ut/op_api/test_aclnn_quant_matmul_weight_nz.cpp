/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_quant_matmul_weight_nz.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

struct QuantBatchMatmulWeightNzTestParam {
    string caseName;
    vector<int64_t> x1;
    vector<int64_t> x2;
    vector<int64_t> scale;
    vector<int64_t> offset;
    vector<int64_t> bias;
    vector<int64_t> out;
    vector<int64_t> x1_stride;
    vector<int64_t> x2_stride;
    aclDataType scaleType;
    aclDataType outType;
    // out
    aclnnStatus expect_ret;
};

class l2_QuantBatchMatmulWeightNz_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_QuantBatchMatmulWeightNz_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_QuantBatchMatmulWeightNz_test TearDown" << endl; }
};


TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_normal_case_01)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_x2_nullptr_case_02)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, nullptr, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_x2_not_align_32_case_03)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_n_equal_k)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_x2_not_align_16_case_04)
{

    TensorDesc x1_desc = TensorDesc({4, 17}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({17, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_yscale_not_empty_case_05)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, y_scale_desc, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_x1Offset_not_empty_case_06)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc x1_offset_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, x1_offset_desc, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_yOffset_not_empty_case_06)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_offset_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, y_offset_desc, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_groupsize_not_0_case_06)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 11),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910B2_test_x2_not_align_16_x2_not_nz)
{
    TensorDesc x1_desc = TensorDesc({4, 17}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({17, 16}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {17, 16});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_x2_not_align_16_case_04)
{

    TensorDesc x1_desc = TensorDesc({4, 17}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({17, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_out_int32_fail)
// {

//     TensorDesc x1_desc = TensorDesc({4, 17}, ACL_INT8, ACL_FORMAT_ND);
//     TensorDesc x2_desc = TensorDesc({17, 16}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({4, 16}, ACL_INT32, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_out_nd_fail)
// {

//     TensorDesc x1_desc = TensorDesc({4, 17}, ACL_INT8, ACL_FORMAT_ND);
//     TensorDesc x2_desc = TensorDesc({17, 16}, ACL_INT8, ACL_FORMAT_ND, {}, 0, {1, 2, 16, 32});
//     TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_out_dtype_fail)
{

    TensorDesc x1_desc = TensorDesc({4, 17}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({17, 16}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {}, 0, {1, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_n_equal_1)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 1}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 1}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_a8w4_mx_0)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x1_desc = TensorDesc({1, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({8, 128}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ, {1, 8}, 0, {2, 8, 16, 4});
    TensorDesc x1_scale_desc = TensorDesc({1, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2_scale_desc = TensorDesc({128, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t groupSize = 32;
    TensorDesc out_desc = TensorDesc({1, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, groupSize),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_01_bf16)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({15360, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 960, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({15360}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 15360}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_a8w4_mx_1)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x1_desc = TensorDesc({16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({64, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0, {4, 4, 16, 32});
    TensorDesc x2_scale_desc = TensorDesc({2, 128}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_scale_desc = TensorDesc({1, 128}, ACL_INT64, ACL_FORMAT_ND);
    int64_t groupSize = 32;
    TensorDesc out_desc = TensorDesc({16, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, x2_scale_desc, y_scale_desc, nullptr, nullptr, nullptr, nullptr, false, false, groupSize),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_mxfp4_weight_nz_micro_scaling_enc_group)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x1_desc = TensorDesc({16, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({64, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0, {4, 4, 16, 32});
    TensorDesc x1_scale_desc = TensorDesc({16, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2_scale_desc = TensorDesc({1, 128, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    int64_t groupSize = 32;
    TensorDesc out_desc = TensorDesc({16, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr,
                              false, false, groupSize),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// FLOAT4+E8M0：内轴（view 最后一维）须为偶数
TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_mxfp4_weight_nz_invalid_odd_inner_axis)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x1_desc = TensorDesc({16, 63}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({128, 63}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 8, 16, 32});
    TensorDesc x1_scale_desc = TensorDesc({16, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2_scale_desc = TensorDesc({128, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    const int64_t groupSizeMx = 32;
    TensorDesc out_desc = TensorDesc({16, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr,
                              false, true, groupSizeMx),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FLOAT4+E8M0：k 维须大于 2
TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_mxfp4_weight_nz_invalid_k_le_2)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x1_desc = TensorDesc({8, 2}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({16, 2}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 32});
    TensorDesc x1_scale_desc = TensorDesc({8, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2_scale_desc = TensorDesc({16, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    const int64_t groupSizeMx = 32;
    TensorDesc out_desc = TensorDesc({8, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr,
                              false, true, groupSizeMx),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FLOAT4+E8M0 + x2 为 NZ：view 最后两维之一为 1（k 或 n 为 1）应拦截
TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_mxfp4_weight_nz_invalid_last_two_dim_one)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 1}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
    TensorDesc x1_scale_desc = TensorDesc({16, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2_scale_desc = TensorDesc({32, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    const int64_t groupSizeMx = 32;
    TensorDesc out_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr,
                              false, true, groupSizeMx),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// FLOAT4+E8M0 + x2 为 NZ：不允许 transposeX1=true
TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend950_test_mxfp4_weight_nz_invalid_transpose_x1_true)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    TensorDesc x1_desc = TensorDesc({16, 64}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({64, 128}, ACL_FLOAT4_E2M1, ACL_FORMAT_FRACTAL_NZ, {}, 0, {4, 4, 16, 32});
    TensorDesc x1_scale_desc = TensorDesc({16, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc x2_scale_desc = TensorDesc({1, 128, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    const int64_t groupSizeMx = 32;
    TensorDesc out_desc = TensorDesc({16, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr,
                              true, true, groupSizeMx),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_01_fp16)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({15360, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 960, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({15360}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 15360}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_02)
{
    TensorDesc x1_desc = TensorDesc({124, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({15360, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 960, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({124}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({15360}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({124, 15360}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_03_bf16)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({5120, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 320, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({5120}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({5120}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 5120}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_03_fp16)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({5120, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 320, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({5120}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 5120}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_04)
{
    TensorDesc x1_desc = TensorDesc({124, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({5120, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 320, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({124}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({5120}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({5120}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({124, 5120}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_05)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({20480, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 1280, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({20480}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({20480}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 20480}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_06)
{
    TensorDesc x1_desc = TensorDesc({6256, 20480}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({5120, 20480}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {640, 320, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({5120}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({5120}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 5120}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_07)
{
    TensorDesc x1_desc = TensorDesc({124, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({20480, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 1280, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({124}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({20480}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({20480}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({124, 20480}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_08)
{
    TensorDesc x1_desc = TensorDesc({124, 20480}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({5120, 20480}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {640, 320, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({124}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({5120}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({5120}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({124, 5120}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_09)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({15360, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 960, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({15360}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 15360}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_10)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({5120, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 320, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({5120}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({5120}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 5120}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_11)
{
    TensorDesc x1_desc = TensorDesc({6256, 5120}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({20480, 5120}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {160, 1280, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({20480}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({20480}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 20480}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a4w4_weight_nz_case_12)
{
    TensorDesc x1_desc = TensorDesc({6256, 20480}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({5120, 20480}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {}, 0, {640, 320, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({6256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2Scale_desc = TensorDesc({5120}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({5120}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({6256, 5120}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a8w4_weight_nz_case_1)
{
    // A8W4 msd方案 支持fp16输出
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {16, 512, 16, 8}).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a8w4_weight_nz_case_2)
{
    // A8W4 msd方案 支持fp16输出
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({1024, 1024}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {128, 64, 16, 8}).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a8w4_weight_nz_case_3)
{
    // A8W4 msd方案 支持fp16输出
    TensorDesc x1_desc = TensorDesc({1, 29568}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 3696}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {462, 512, 16, 8}).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({8192}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({8192}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 8192}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a8w4_weight_nz_case_4)
{
    // A8W4 msd方案 支持fp16输出
    TensorDesc x1_desc = TensorDesc({1, 3448}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({1024, 431}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {54, 64, 16, 8}).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a8w4_weight_nz_case_5)
{
    // A8W4 msd方案 支持fp16输出
    TensorDesc x1_desc = TensorDesc({8, 8}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({1, 1}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 1, 16, 8}).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({1}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({8, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulWeightNz_test, a8w4_weight_nz_case_6)
{
    // A8W4 msd方案 支持fp16输出
    TensorDesc x1_desc = TensorDesc({1, 30000}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({30000, 128}, ACL_INT32, ACL_FORMAT_FRACTAL_NZ, {}, 0, {16, 1875, 16, 8}).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}