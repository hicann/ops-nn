/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_quant_batch_matmul_inplace_add.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class quant_batch_matmul_inplace_add_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "quant_batch_matmul_inplace_add_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "quant_batch_matmul_inplace_add_test TearDown" << endl; }
};

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_illegal_format_case_01)
{
    int64_t m = 128;
    int64_t n = 2048;
    int64_t k = 4096;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_FLOAT8_E5M2, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({k / 64, m, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_illegal_shape_case_01)
{
    int64_t m = 128;
    int64_t n = 2048;
    int64_t k = 4096;

    TensorDesc x1_desc = TensorDesc({k, m, 2}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({k / 64, m, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_illegal_shape_case_02)
{
    int64_t m = 128;
    int64_t n = 2048;
    int64_t k = 4096;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({k / 64, m, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({k / 64, n, 3}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_illegal_shape_case_03)
{
    int64_t m = 128;
    int64_t n = 2048;
    int64_t k = 4096;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({k / 64, m, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 111;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_illegal_shape_case_04)
{
    int64_t m = 128;
    int64_t n = 2048;
    int64_t k = 4096;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({1000, m, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_illegal_shape_case_05)
{
    int64_t m = 128;
    int64_t n = 2048;
    int64_t k = 4096;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({k / 64, m, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = false;
    bool transposeX2 = false;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_mxfp8_illegal_transpose_false_true)
{
    int64_t m = 128;
    int64_t n = 256;
    int64_t k = 512;

    TensorDesc x1_desc = TensorDesc({m, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({n, k}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({m, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({n, k / 64, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = false;
    bool transposeX2 = true;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_mxfp8_support_transpose_true_false)
{
    int64_t m = 128;
    int64_t n = 256;
    int64_t k = 512;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({k / 64, m, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({k / 64, n, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_hif8_tt_pertensor_support)
{
    int64_t m = 128;
    int64_t n = 256;
    int64_t k = 512;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 0;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_hif8_tt_illegal_transpose_true_true)
{
    int64_t m = 128;
    int64_t n = 256;
    int64_t k = 512;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({n, k}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = true;
    int64_t groupsize = 0;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_hif8_tt_illegal_x1_scale_dtype_e8m0)
{
    int64_t m = 128;
    int64_t n = 256;
    int64_t k = 512;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({1}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 0;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_hif8_tt_illegal_group_size_nonzero)
{
    int64_t m = 128;
    int64_t n = 256;
    int64_t k = 512;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 32;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(quant_batch_matmul_inplace_add_test, ascend950_test_hif8_tt_illegal_x1_scale_shape_not_one)
{
    int64_t m = 128;
    int64_t n = 256;
    int64_t k = 512;

    TensorDesc x1_desc = TensorDesc({k, m}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({k, n}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1_scale_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc y_input_desc = TensorDesc({m, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupsize = 0;
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1_desc, x2_desc, x1_scale_desc, x2_scale_desc,
                              y_input_desc, transposeX1, transposeX2, groupsize),
                        OUTPUT());
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
