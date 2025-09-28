/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_quant_matmul_weight_nz.h"

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
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
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

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910_95_test_x2_not_align_16_case_04)
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

// TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910_95_test_out_int32_fail)
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

// TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910_95_test_out_nd_fail)
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

TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910_95_test_out_dtype_fail)
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
TEST_F(l2_QuantBatchMatmulWeightNz_test, ascend910_95_test_n_equal_1)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 1}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {1, 2, 16, 32});
    TensorDesc scale_desc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 1}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulWeightNz, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}