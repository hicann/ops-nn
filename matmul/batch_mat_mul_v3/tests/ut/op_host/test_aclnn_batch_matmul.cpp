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

#include "../../../op_host/op_api/aclnn_batch_matmul.h"
#include "opdev/platform.h"
#include "op_api/op_api_def_nn.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;
class l2_batch_matmul_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "batch_matmul_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "batch_matmul_test TearDown" << endl; }
};

TEST_F(l2_batch_matmul_test, case_1)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
}

TEST_F(l2_batch_matmul_test, case_2)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, case_3)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
}

TEST_F(l2_batch_matmul_test, case_4)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto
        out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, case_5)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_6)
{
    auto tensor_1_desc = TensorDesc({1, 1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND)
                               .ValueRange(-2, 2)
                               .Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_7)
{
    auto tensor_1_desc = nullptr;
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_batch_matmul_test, case_8)
{
    auto tensor_1_desc = TensorDesc({1, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 0, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_batch_matmul_test, case_9)
{
    auto tensor_1_desc = TensorDesc({1, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 0, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();  // soc version  2. 二段接口
}

TEST_F(l2_batch_matmul_test, case_10)
{
    auto tensor_1_desc = TensorDesc({0, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({0, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({0, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();  // soc version  2. 二段接口
}

TEST_F(l2_batch_matmul_test, case_11)
{
    auto tensor_1_desc = TensorDesc({1, 5, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 5, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 5, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_13)
{
    auto tensor_1_desc = TensorDesc({3, 5, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({2, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({6, 5, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 910 Fp16 16Aligin Nd
TEST_F(l2_batch_matmul_test, ascend910A_case_14)
{
    auto tensor_1_desc = TensorDesc({16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_1_desc_t = TensorDesc({16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {512, 1, 16}, 0, {16, 32, 16})
                               .ValueRange(-2, 2);

    auto tensor_2_desc = TensorDesc({16, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_2_desc_t = TensorDesc({16, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND, {2048, 1, 32}, 0, {16, 64, 32})
                               .ValueRange(-2, 2);

    auto out_tensor_desc = TensorDesc({16, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND)
                               .ValueRange(-2, 2)
                               .Precision(0.005, 0.005);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 910 Fp16 16notAligin Nd
TEST_F(l2_batch_matmul_test, ascend910A_case_18)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_1_desc_t = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {6, 1, 2}, 0, {1, 3, 2}).ValueRange(-2, 2);

    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_2_desc_t = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3})
                               .ValueRange(-2, 2);

    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 910 Fp32 16notAligin Nd
TEST_F(l2_batch_matmul_test, ascend910A_case_22)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_1_desc_t = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {6, 1, 2}, 0, {1, 3, 2}).ValueRange(-2, 2);

    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_2_desc_t = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3}).ValueRange(-2, 2);

    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;

    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, case_nz_mat2_k_axis_1)
{
    auto tensor_1_desc = TensorDesc({1, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({1, 1, 4}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_out_shape_mismatch)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 2, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_self_nz_format)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_weightnz_dtype_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({8, 32, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({8, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnBatchMatMulWeightNz, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                        cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_weightnz_out_dtype_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({8, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({8, 16, 64}, ACL_BF16, ACL_FORMAT_ND);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnBatchMatMulWeightNz, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                        cube_math_type);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 910 Fp32 16Aligin Nd
TEST_F(l2_batch_matmul_test, ascend910A_case_23)
{
    auto tensor_1_desc = TensorDesc({16, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_1_desc_t = TensorDesc({16, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {512, 1, 16}, 0, {16, 32, 16})
                               .ValueRange(-2, 2);

    auto tensor_2_desc = TensorDesc({16, 32, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_2_desc_t = TensorDesc({16, 32, 64}, ACL_FLOAT, ACL_FORMAT_ND, {2048, 1, 32}, 0, {16, 64, 32})
                               .ValueRange(-2, 2);

    auto out_tensor_desc = TensorDesc({16, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 910B Fp32 Nd
TEST_F(l2_batch_matmul_test, ascend910B2_case_24)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {6, 1, 2}, 0, {1, 3, 2}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3}).ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 910B Fp16 Nd  IsNdToNzOnTheFly = true,
TEST_F(l2_batch_matmul_test, ascend910B2_case_28)
{
    auto tensor_1_desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {512, 1, 16}, 0, {1, 32, 16})
                               .ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND, {2048, 1, 32}, 0, {1, 64, 32})
                               .ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({1, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 910B Fp16 Nd  IsNdToNzOnTheFly = false,
TEST_F(l2_batch_matmul_test, ascend910B2_case_32)
{
    auto tensor_1_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({1, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {20, 1, 4}, 0, {1, 5, 4}).ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({1, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

TEST_F(l2_batch_matmul_test, case_36)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, case_37)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, test_aligned_fp32_false_true_1D_storage_shape)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({864, 49, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_t_desc = TensorDesc({864, 32, 49}, ACL_FLOAT, ACL_FORMAT_ND, {1568, 1, 32}, 0, {1354752});
    TensorDesc out_desc = TensorDesc({864, 49, 49}, ACL_FLOAT, ACL_FORMAT_ND);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_t_desc), OUTPUT(out_desc), cube_math_type);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, test_null_tensor)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({2, 4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({2, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), cube_math_type);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_case_40)
{
    auto tensor_1_desc = TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {16384, 1, 128}, 0, {2097152})
                               .ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {16384, 1, 128}, 0, {2097152})
                               .ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// shape < 2
TEST_F(l2_batch_matmul_test, case_41)
{
    auto tensor_1_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, cubeMathType_0_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_4_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 4; // 会路由到0
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_1_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_2_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_3_fp16_fp16)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 3;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 3;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_1_fp32_fp16)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_2_fp32_fp16)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_1_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_2_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_1_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_2_fp32_fp32)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, cubeMathType_2_fp16_fp16_fp32)
{
    auto tensor_1_desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({1, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 2;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp32_bmm_V3)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({2400, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc_t = TensorDesc({2400, 8, 128}, ACL_FLOAT, ACL_FORMAT_ND, {1024, 1, 8}, 0, {2457600});
    TensorDesc out_desc = TensorDesc({2400, 4, 128}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc_t), OUTPUT(out_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp32_bmm_V2_with_a_trans)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc_t = TensorDesc({16, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {512, 1, 16}, 0, {8192});
    TensorDesc b_desc = TensorDesc({16, 32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc_t, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_self_bf16_mat2_dtype_not_matched)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc_t = TensorDesc({16, 16, 32}, ACL_BF16, ACL_FORMAT_ND, {512, 1, 16}, 0, {8192});
    TensorDesc b_desc = TensorDesc({16, 32, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 0;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc_t, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_self_bf16_out_dtype_not_matched)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc_t = TensorDesc({16, 16, 32}, ACL_BF16, ACL_FORMAT_ND, {512, 1, 16}, 0, {8192});
    TensorDesc b_desc = TensorDesc({16, 32, 8}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 0;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc_t, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp32_bmm_al1_fullload_boundary)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1, 256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({48, 256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({48, 256, 256}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp16_bmm_al1_fullload_boundary)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1, 256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({48, 512, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({48, 256, 256}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp16_bmm_multi_batch_unaligned)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({219277, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({219277, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({219277, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp32_bmm_multi_batch_AL1_full_load)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({1500, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1500, 128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1500, 1, 512}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp32_bmm_bl1_fullload_boundary)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({48, 256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 256, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({48, 256, 256}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_fp16_bmm_bl1_fullload_boundary)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({48, 256, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({1, 512, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({48, 256, 256}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_test_dtype_promotion_of_mix_bf16_fp32)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({48, 256, 512}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({48, 512, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({48, 256, 256}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_res = OP_API_UT(aclnnBatchMatMul, INPUT(a_desc, b_desc), OUTPUT(out_desc), math_type);
    aclRet = ut_res.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_0)
{
    auto tensor_1_desc = TensorDesc({4096, 128, 9}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({4096, 128, 9}, ACL_BF16, ACL_FORMAT_ND, {1152, 1, 128}, 0, {4096, 9, 128})
                               .ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({4096, 9, 8}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({4096, 9, 8}, ACL_BF16, ACL_FORMAT_ND, {72, 1, 9}, 0, {4096, 8, 9})
                               .ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({4096, 128, 8}, ACL_BF16, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_1)
{
    auto tensor_1_desc = TensorDesc({2048, 2, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({2048, 2, 64}, ACL_BF16, ACL_FORMAT_ND, {128, 1, 2}, 0, {2048, 64, 2})
                               .ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({2048, 64, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({2048, 64, 3}, ACL_BF16, ACL_FORMAT_ND, {192, 1, 64}, 0, {2048, 3, 64})
                               .ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({2048, 2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_2)
{
    auto tensor_1_desc = TensorDesc({3072, 1, 300}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({3072, 1, 300}, ACL_BF16, ACL_FORMAT_ND, {300, 1, 1}, 0, {3072, 300, 1})
                               .ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({3072, 300, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({3072, 300, 16}, ACL_BF16, ACL_FORMAT_ND, {4800, 1, 300}, 0, {3072, 16, 300})
                               .ValueRange(0, 2);

    auto
        out_tensor_desc = TensorDesc({3072, 1, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_3)
{
    auto tensor_1_desc = TensorDesc({3072, 1, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({3072, 1, 16}, ACL_BF16, ACL_FORMAT_ND, {16, 1, 1}, 0, {3072, 16, 1})
                               .ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({3072, 16, 60}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({3072, 16, 60}, ACL_BF16, ACL_FORMAT_ND, {960, 1, 16}, 0, {3072, 60, 16})
                               .ValueRange(0, 2);

    auto
        out_tensor_desc = TensorDesc({3072, 1, 60}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_4)
{
    auto tensor_1_desc = TensorDesc({8, 1536, 2048}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({8, 2048, 7680}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({8, 1536, 7680}, ACL_BF16, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_fp32_0)
{
    auto tensor_1_desc = TensorDesc({3000, 1, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t = TensorDesc({3000, 1, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {1024, 1, 1}, 0, {3000, 1024, 1})
                               .ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({3000, 1024, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t = TensorDesc({3000, 1024, 128}, ACL_FLOAT, ACL_FORMAT_ND, {131072, 1, 1024}, 0,
                                      {3000, 128, 1024})
                               .ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({3000, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend950_test_bmm2mm)
{
    auto tensor_1_desc = TensorDesc({69, 16, 224}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 224, 112}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({69, 16, 112}, ACL_FLOAT16, ACL_FORMAT_ND)
                               .ValueRange(-2, 2)
                               .Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend950_test_bmm2m)
{
    auto tensor_1_desc = TensorDesc({2400, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({2400, 1, 976}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto
        out_tensor_desc = TensorDesc({2400, 4, 976}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend950_test_bmm2m_N1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({2400, 20, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({2400, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({2400, 20, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend950_fp16_fp16_fp32_output)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({8, 16, 20}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 20}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend950_bf16_bf16_fp32_output)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({8, 16, 20}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 20}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 0;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_5)
{
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({8, 1, 20}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({8, 10, 20}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_6)
{
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto
        out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 接口整改异常用例 - 950
TEST_F(l2_batch_matmul_test, batch_matmul_950_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_950_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_950_FP32_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_950_FP16_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_950_FP16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_950_BF16_BF16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 接口整改异常用例 - 310
TEST_F(l2_batch_matmul_test, batch_matmul_310_FP32_FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = 0;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_310_FP32_FP16_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = 0;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_310_FP32_FP32_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = 3;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_310_FP32_FP16_USE_HF32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = 3;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_310_FP32_FP32_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_310_FP32_FP16_FP16FP32_KEEP_DTYPE)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({8, 10, 5000}, ACL_FLOAT, ACL_FORMAT_ND)
                               .ValueRange(0, 2)
                               .Precision(0.0001, 0.0001);
    int8_t cube_math_type = -1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    cube_math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_keep_dtype_aligned)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({8, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({8, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_keep_dtype_aligned)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({8, 16, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({8, 32, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({8, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_keep_dtype_not_aligned)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 7, 9}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({3, 5, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_keep_dtype_not_aligned)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({3, 5, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({3, 7, 9}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({3, 5, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_use_hf32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = USE_HF32;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_use_hf32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 32, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = USE_HF32;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_allow_fp32_down_precision)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_allow_fp32_down_precision)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 32, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = ALLOW_FP32_DOWN_PRECISION;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_large_batch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({128, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({128, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({128, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_large_batch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({128, 16, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({128, 32, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({128, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_batch_broadcast)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_batch_broadcast)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 32, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_k1_special)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 1, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_k1_special)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 1, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_empty_tensor)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({2, 0, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({2, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    int8_t cube_math_type = KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_fp16_fp16_fp32_transpose)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_1_desc_t = TensorDesc({4, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {512, 1, 16}, 0, {4, 32, 16})
                               .ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc_t = TensorDesc({4, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND, {64, 1, 32}, 0, {4, 16, 32})
                               .ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

    int8_t math_type = KEEP_DTYPE;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, batch_matmul_16in32out_bf16_bf16_fp32_transpose)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_1_desc = TensorDesc({4, 16, 32}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_1_desc_t = TensorDesc({4, 16, 32}, ACL_BF16, ACL_FORMAT_ND, {512, 1, 16}, 0, {4, 32, 16})
                               .ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({4, 32, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc_t = TensorDesc({4, 32, 16}, ACL_BF16, ACL_FORMAT_ND, {64, 1, 32}, 0, {4, 16, 32})
                               .ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({4, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.005, 0.005);

    int8_t math_type = KEEP_DTYPE;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc),
                                    math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                   math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true = OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc),
                                  math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, aclnnBatchMatMulExecute)
{
    auto self = TensorDesc({2, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr();
    auto mat2 = TensorDesc({2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr();
    auto out = TensorDesc({2, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr();
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnBatchMatMulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
    EXPECT_TRUE(ret == ACLNN_SUCCESS || ret == ACLNN_ERR_INNER_NULLPTR);
    if (executor != nullptr) {
        delete executor;
    }
}

TEST_F(l2_batch_matmul_test, aclnnBatchMatMulWeightNzExecute)
{
    auto self = TensorDesc({2, 64, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr();
    auto mat2 = TensorDesc({2, 128, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr();
    auto out = TensorDesc({2, 64, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclTypeRawPtr();
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnBatchMatMulWeightNzGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
    EXPECT_TRUE(ret == ACLNN_SUCCESS || ret == ACLNN_ERR_INNER_NULLPTR || ret == ACLNN_ERR_PARAM_INVALID);
    if (executor != nullptr) {
        delete executor;
    }
}

// ===================== l0op 直调白盒用例：覆盖 batch_matmul.cpp 90~145 行的三个 l0op 函数 =====================
//   - BatchMatMulNzFp162Fp16（输出 FP16、FORMAT_FRACTAL_NZ）
//   - BatchMatMulNdFp162Fp32（输出 FP32、FORMAT_ND）
//   - BatchMatMulNzFp162Fp32（输出 FP32、FORMAT_FRACTAL_NZ）
// 直调手法沿用本仓先例 matmul/common/tests/ut/test_batch_matmul.cpp：
//   直接 #include batch_matmul.cpp，使被测函数在用例目标内编译执行；
//   OP_TYPE_REGISTER 采用 inline + 函数局部静态（按算子名 GenOpTypeId 一次），
//   与 opapi so 内同源定义按 ODR 合并/符号拦截，不会重复注册，可安全直调；
//   该 .cpp 在整个 UT 可执行文件中仅被本文件 include 一次。
// 触发链路（供对照）：aclnnBatchMatMul/aclnnAddbmm/aclnnBaddbmm GetWorkspaceSize
//   -> ExecBmmOpV2(batch_matmul_util.cpp) -> GetBatchMatmulOp(L398)
//   -> 输入 FP16/BF16 且未命中 ascendc 场景时按 dtype/format 分发：
//      输出 FP16 + self_format=NZ -> BatchMatMulNzFp162Fp16（L442）
//      输出 FP32 + self_format=ND -> BatchMatMulNdFp162Fp32（L448）
//      输出 FP32 + self_format=NZ -> BatchMatMulNzFp162Fp32（L451）
// 分支覆盖说明：
//   主路径（AllocTensor + INFER_SHAPE + ADD_TO_LAUNCHER_LIST_AICORE 全成功）逐函数覆盖；
//   INFER_SHAPE 失败分支以 K 维/批维不一致输入确定性触发（ret != ACLNN_SUCCESS -> return nullptr）；
//   AllocTensor 失败与 launcher 注册失败分支在本 UT 框架下不可注入：
//     - op_api UT 桩 tests/ut/op_api/stub/opdev/op_executor.cpp 的 CreatAiCoreKernelLauncher
//       恒返回 ACLNN_SUCCESS（共享桩，不在本算子可改范围内）；
//     - executor 正常创建时 AllocTensor 不返回空，无故障注入点。
//   该两分支为对上一次框架调用的错误码透传，行级逻辑与 INFER_SHAPE 失败分支一致（return nullptr）。

#include <cstdint>
#include "opdev/op_executor.h"

// 直调被测源文件（覆盖目标：batch_matmul.cpp 90~145 行）
#include "../../../../common/op_host/op_api/batch_matmul.cpp"

using namespace l0op;

namespace {

aclTensor* MakeTensor(aclOpExecutor* executor, const std::initializer_list<int64_t>& dims, op::DataType dtype,
                      op::Format format)
{
    return executor->AllocTensor(op::Shape(dims), dtype, format);
}

constexpr int64_t kOpImplModeDefault = 0x1; // 与 GetBatchMatmulOpInfo 对 FP16 输入的默认取值一致

} // namespace

class BatchMatmulL0OpDirectTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "BatchMatmulL0OpDirectTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "BatchMatmulL0OpDirectTest TearDown" << endl; }

    // BatchMatMulNzFp162Fp16：FP16 入 NZ，输出 FP16/FRACTAL_NZ
    void TestNzFp162Fp16MainPath(bool adj)
    {
        auto uniqueExecutor = CREATE_EXECUTOR();
        ASSERT_NE(uniqueExecutor.get(), nullptr);
        auto* executor = uniqueExecutor.get();
        // K=128：非转置 x1 为 {b, m=64, k=128}，转置后为 {b, k=128, m=64}
        auto* x1 = adj ? MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ) :
                         MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        auto* x2 = adj ? MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ) :
                         MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        ASSERT_NE(x1, nullptr);
        ASSERT_NE(x2, nullptr);
        auto* out = BatchMatMulNzFp162Fp16(x1, x2, nullptr, nullptr, adj, adj, false, kOpImplModeDefault, executor);
        ASSERT_NE(out, nullptr);
        EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
        EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
        EXPECT_EQ(out->GetViewShape(), op::Shape({2, 64, 64}));
    }

    // BatchMatMulNdFp162Fp32：FP16 入 ND，输出 FP32/ND
    void TestNdFp162Fp32MainPath(bool adj)
    {
        auto uniqueExecutor = CREATE_EXECUTOR();
        ASSERT_NE(uniqueExecutor.get(), nullptr);
        auto* executor = uniqueExecutor.get();
        auto* x1 = adj ? MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND) :
                         MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
        auto* x2 = adj ? MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND) :
                         MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
        ASSERT_NE(x1, nullptr);
        ASSERT_NE(x2, nullptr);
        auto* out = BatchMatMulNdFp162Fp32(x1, x2, nullptr, nullptr, adj, adj, false, kOpImplModeDefault, executor);
        ASSERT_NE(out, nullptr);
        EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
        EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_ND);
        EXPECT_EQ(out->GetViewShape(), op::Shape({2, 64, 64}));
    }

    // BatchMatMulNzFp162Fp32：FP16 入 NZ，输出 FP32/FRACTAL_NZ
    void TestNzFp162Fp32MainPath(bool adj)
    {
        auto uniqueExecutor = CREATE_EXECUTOR();
        ASSERT_NE(uniqueExecutor.get(), nullptr);
        auto* executor = uniqueExecutor.get();
        auto* x1 = adj ? MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ) :
                         MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        auto* x2 = adj ? MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ) :
                         MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
        ASSERT_NE(x1, nullptr);
        ASSERT_NE(x2, nullptr);
        auto* out = BatchMatMulNzFp162Fp32(x1, x2, nullptr, nullptr, adj, adj, false, kOpImplModeDefault, executor);
        ASSERT_NE(out, nullptr);
        EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
        EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
        EXPECT_EQ(out->GetViewShape(), op::Shape({2, 64, 64}));
    }
};

// ===================== 主路径：AllocTensor + INFER_SHAPE + ADD_TO_LAUNCHER 全成功 =====================

// L90~L107 主路径
TEST_F(BatchMatmulL0OpDirectTest, L0_NzFp162Fp16_MainPath) { TestNzFp162Fp16MainPath(false); }

// L90~L107 转置入参（adjX1=adjX2=true，覆盖 L0_DFX 属性记录与 INFER_SHAPE 转置语义）
TEST_F(BatchMatmulL0OpDirectTest, L0_NzFp162Fp16_MainPathAdj) { TestNzFp162Fp16MainPath(true); }

// L109~L126 主路径
TEST_F(BatchMatmulL0OpDirectTest, L0_NdFp162Fp32_MainPath) { TestNdFp162Fp32MainPath(false); }

// L109~L126 转置入参
TEST_F(BatchMatmulL0OpDirectTest, L0_NdFp162Fp32_MainPathAdj) { TestNdFp162Fp32MainPath(true); }

// L128~L145 主路径
TEST_F(BatchMatmulL0OpDirectTest, L0_NzFp162Fp32_MainPath) { TestNzFp162Fp32MainPath(false); }

// L128~L145 转置入参
TEST_F(BatchMatmulL0OpDirectTest, L0_NzFp162Fp32_MainPathAdj) { TestNzFp162Fp32MainPath(true); }

// ===================== 异常分支：INFER_SHAPE 失败 -> return nullptr =====================

// L97~L102：K 维不一致（128 vs 100）触发 INFER_SHAPE 失败
TEST_F(BatchMatmulL0OpDirectTest, L0_NzFp162Fp16_InferShapeFail)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeTensor(executor, {2, 100, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    auto* out = BatchMatMulNzFp162Fp16(x1, x2, nullptr, nullptr, false, false, false, kOpImplModeDefault, executor);
    EXPECT_EQ(out, nullptr);
}

// L116~L121：K 维不一致触发 INFER_SHAPE 失败
TEST_F(BatchMatmulL0OpDirectTest, L0_NdFp162Fp32_InferShapeFail)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* x2 = MakeTensor(executor, {2, 100, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    auto* out = BatchMatMulNdFp162Fp32(x1, x2, nullptr, nullptr, false, false, false, kOpImplModeDefault, executor);
    EXPECT_EQ(out, nullptr);
}

// L135~L140：批维不可广播（2 vs 3）触发 INFER_SHAPE 失败
TEST_F(BatchMatmulL0OpDirectTest, L0_NzFp162Fp32_InferShapeFail)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeTensor(executor, {3, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    auto* out = BatchMatMulNzFp162Fp32(x1, x2, nullptr, nullptr, false, false, false, kOpImplModeDefault, executor);
    EXPECT_EQ(out, nullptr);
}

// ===================== batch_matmul_util.cpp 白盒用例（本轮新增，覆盖目标区间见下表） =====================
// 覆盖目标（batch_matmul_util.cpp 此前未覆盖行区间 -> 用例）：
//   L96~107  SetTensorToNDFormat                        -> Util_SetTensorToNDFormat_*
//   L327~347 CheckShapeEqualToMul                       -> Util_CheckShapeEqualToMul_*
//   L359~389 CheckArchIfBatchMatMulToMulDav3510         -> 基线已 100% 行覆盖（3510 k=1 既有用例触达），本轮回归确认
//   L441~453 GetBatchMatmulOp FP16/BF16 分发尾          -> Util_GetBatchMatmulOp_Fp16*（NZ16->16 / ND16->32 /
//   NZ16->32） L475~514 CheckTransNonContiguousShapeSupport        -> Util_CheckTransNonContig_*（L0
//   不容纳批时的负载均衡率两分支） L546~566 CheckMergeBatchNonContiguousShapeSupport   -> Util_CheckMergeBatch_*（adj
//   组合、L0c 溢出、批不等） L584~610 CheckStreamKNonContiguousShapeSupport      -> Util_CheckStreamK_*（fp32 大K / 小K
//   / fp16 / fp32 对齐32） L645~664 CheckNonContiguousTranspose                -> Util_CheckNonContigTranspose_*（5
//   个返回出口） L666~673 CheckSocIfBatchMatMulToMulDefault          -> Util_CheckSocIfBatchMatMulToMulDefault_*（直调
//   + 未映射 arch 路由） L1249~1324 checkFusedmm 尾部 + ExecFusedmmOp        -> Util_CheckFusedmm_* /
//   Util_ExecFusedmmOp_* / baddbmm_fusedmm_*（公开入口）
// 直调手法沿用上轮 BatchMatmulL0OpDirectTest 段与 matmul/common/tests/ut/test_batch_matmul.cpp 先例：
//   直接 #include batch_matmul_util.cpp，使匿名命名空间内的检查函数在本 TU 可见；
//   该 .cpp 同时被编译进 libophost_nn_opapi.so（公开入口用例经 so 内同源实现执行，gcov 按源文件路径合并统计）。
// 平台注入：UT 桩 tests/ut/op_api/stub/opdev/platform.{h,cpp} 提供
//   - NpuArchManager（3510/1001 等 arch 注入，RAII 恢复，经符号拦截对 so 内代码同样生效）；
//   - SetCubeCoreNum（AI Core 数注入，本段用 CoreNumGuard 保存/恢复避免用例间污染）。
//   rtGetSocSpec 由本 TU 内置确定性桩提供（宏改名仅重写被 include 的源文件副本，见下方说明），
//   返回 l0a/l0b=65536、l0c=262144、l1=524288（与本机 ascend910b 真实规格探针实测一致），
//   使直调白盒用例不依赖门禁机的真实运行时（CI 门禁机 rtGetSocSpec 查询失败会导致期望 true 用例回归）。
// 本 UT 框架下不可达分支（如实记录，未硬凑）：
//   - CheckStreamKNonContiguousShapeSupport L610（return true）：桩 GetVectorCoreNum 恒返回 0，
//     aiv==2*aic 仅当 aic==0 时成立，而 aic==0 时末行 batchNum*mCnt*nCnt>=1 > 0 恒成立必返 false；
//   - CheckNonContiguousTranspose L635~636（streamk 命中出口）：同上，CheckStreamK 无法返回 true。

#include <cstdint>
#include <dlfcn.h>
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "../../../op_host/op_api/aclnn_baddbmm.h"

// ---- aclnn_kernels/level0 外部符号的最小 hidden 桩 ----
// libopapi_math_stub.so 是 ophost_nn_opapi 的私有依赖，无法满足本 TU 的未定义符号；
// 故对 util 链路引用的 l0op 外部函数提供本 TU 内解析的 hidden 桩：
//   - hidden 可见性（首声明处标注，后续头文件声明继承）保证不进入可执行文件动态符号表，
//     so 侧对真实实现（LD_PRELOAD 的 libopapi_math.so）的绑定不受影响，避免污染既有用例；
//   - 本轮用例实际执行路径仅触达 ReFormat（ND 直通）与 Reshape（连续视图重排），
//     其余桩仅满足链接，不参与任何执行路径；
//   - BmmCheckHitV3Shape/MmCheckHitV3Shape 在本轮直调用例分支中不会执行到（见各用例注释），桩返回 false。
namespace l0op {
__attribute__((visibility("hidden"))) const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype,
                                                            aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Fill(const aclTensor* dims, const aclTensor* value,
                                                            const aclIntArray* outShape, aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* Mul(const aclTensor* self, const aclTensor* other,
                                                           aclOpExecutor* executor);
__attribute__((visibility("hidden"))) const aclTensor* ReFormat(const aclTensor* x, const op::Format& format,
                                                                aclOpExecutor* executor);
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
__attribute__((visibility("hidden"))) bool BmmCheckHitV3Shape(const aclTensor* x1, const aclTensor* x2,
                                                              const aclTensor* bias, const bool adjX1, const bool adjX2,
                                                              op::Format self_format, op::Format mat2_format,
                                                              const bool enableFp16Bf16InFp32Out);
__attribute__((visibility("hidden"))) bool MmCheckHitV3Shape(const aclTensor* x1, const aclTensor* x2,
                                                             const aclTensor* bias, const bool transposeX1,
                                                             const bool transposeX2, op::Format mat2_format,
                                                             bool supportSplitK);
} // namespace l0op

// 上述 hidden 首声明之后的头文件重复声明继承 hidden 可见性
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/fill.h"
#include "level0/mul.h"
#include "matmul/common/op_host/op_api/matmul_v2tov3.h"

namespace l0op {
const aclTensor* Cast(const aclTensor* self, op::DataType, aclOpExecutor*) { return self; }
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor*) { return x; }
const aclTensor* Fill(const aclTensor*, const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* Mul(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
const aclTensor* ReFormat(const aclTensor* x, const op::Format& format, aclOpExecutor*)
{
    // 本轮用例仅对已是目标格式的张量调用（ND->ND），语义等价直通；空入参直通空
    return (x != nullptr && x->GetStorageFormat() == format) ? x : nullptr;
}
const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)
{
    // 连续张量的同元素数形状重排等价于零偏移视图；与真实实现一致地处理 -1 维推断
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
    // aclIntArray 为不透明类型，无法在本 TU 读取元素；经 dlsym 转发至 LD_PRELOAD 的
    // libopapi_math.so 中真实实现（本桩为 hidden 符号，不在动态符号表，不会被 dlsym 命中，无递归风险）
    using ReshapeFn = const aclTensor* (*)(const aclTensor*, const aclIntArray*, aclOpExecutor*);
    static ReshapeFn realReshape = reinterpret_cast<ReshapeFn>(
        dlsym(RTLD_DEFAULT, "_ZN4l0op7ReshapeEPK9aclTensorPK11aclIntArrayP13aclOpExecutor"));
    if (realReshape == nullptr) {
        return nullptr;
    }
    return realReshape(x, shape, executor);
}
const aclTensor* TransData(const aclTensor*, op::Format, int64_t, aclOpExecutor*) { return nullptr; }
const aclTensor* Transpose(const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* ViewCopy(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
bool BmmCheckHitV3Shape(const aclTensor*, const aclTensor*, const aclTensor*, bool, bool, op::Format, op::Format, bool)
{
    return false;
}
bool MmCheckHitV3Shape(const aclTensor*, const aclTensor*, const aclTensor*, bool, bool, op::Format, bool)
{
    return false;
}
} // namespace l0op

// ---- 直调被测源文件及其依赖（覆盖目标：batch_matmul_util.cpp 96~673、1210~1324） ----
// batch_matmul_util.cpp 及其依赖的 matmul_util.cpp/cube_util.cpp/matmul.cpp/fused_matmul.cpp
// 在 libophost_nn_opapi.so 中为本地符号（不可跨 DSO 链接），公开入口用例经 so 内同源实现执行
// （gcov 按源文件路径合并统计），直调用例则在本 TU 内编译执行，故在此显式 include 下列源文件。
// 先于隔离命名空间补齐 batch_matmul_util.cpp 的 include 列表，使 wrapper 内同名 include 幂等空操作，
// 避免头文件声明被嵌套进隔离命名空间。
#include "runtime/runtime/base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "matmul/common/op_host/math_util_nn.h"

// libruntime.so 为可执行文件的传递依赖（经 libnnopbase 等引入），非直接链接输入。
// rtGetSocSpec 平台依赖收敛（CI 门禁环境回归根因修复）：
//   batch_matmul_util.cpp 的 CheckTransNonContiguousShapeSupport / CheckMergeBatchNonContiguousShapeSupport /
//   CheckArchIfBatchMatMulToMulDav3510 经 rtGetSocSpec 查询 l0/l1 容量。原先声明为 weak、运行期解析到
//   进程内真实 libruntime：本机（带设备）返回 0 与真实规格值，用例通过；CI 门禁机（ophost/opapi 环境，
//   无可用运行时探测）同一查询返回非 0，CHECK_RET(...==0, false) 短路令上述函数恒返回 false，
//   导致 8 个"期望 true"的白盒用例失败（CI run 28ec79867：仅此 8 例失败，全部为 rtGetSocSpec 成功依赖路径；
//   LD_PRELOAD 模拟 rtGetSocSpec 失败可 1:1 复现同一失败集合）。
//   故改为在本 TU 内置确定性桩并经宏改名（#define ... #undef）仅重写下方被 include 的源文件副本中的调用：
//   - 返回规格值与本机 ascend910b 真实规格一致（l0a=l0b=65536、l0c=262144、l1=524288，探针实测），
//     各用例注释中的容量推导与此前本机运行完全一致；
//   - 本 TU 不再引用真实 rtGetSocSpec 符号（亦不再有 weak 声明），libophost_nn_opapi.so 内同源实现的
//     动态解析不受影响，公开入口用例行为零变化；
//   - 未登记的 label/key 按查询失败处理（返回非 0），保持被测代码"查询失败即返回 false"的原语义。
#include <cstdio>
#include <cstring>

namespace {
int BmmUtRtGetSocSpec(const char* label, const char* key, char* val, unsigned int valLen)
{
    if (label == nullptr || key == nullptr || strcmp(label, "AICoreSpec") != 0) {
        return 1;
    }
    uint32_t size = 0;
    if (strcmp(key, "l0_a_size") == 0) {
        size = 65536U;
    } else if (strcmp(key, "l0_b_size") == 0) {
        size = 65536U;
    } else if (strcmp(key, "l0_c_size") == 0) {
        size = 262144U;
    } else if (strcmp(key, "l1_size") == 0) {
        size = 524288U;
    } else {
        return 1;
    }
    if (val == nullptr || valLen == 0) {
        return 2;
    }
    const int n = snprintf(val, valLen, "%u", size);
    return (n > 0 && static_cast<unsigned int>(n) < valLen) ? 0 : 3;
}
} // namespace

#define rtGetSocSpec BmmUtRtGetSocSpec // 宏作用域仅覆盖下列被 include 的源文件副本，include 结束后立即 #undef

#include "../../../../common/op_host/op_api/cube_util.cpp"
#include "../../../../common/op_host/op_api/matmul.cpp"
#include "../../../../common/op_host/op_api/fused_matmul.cpp"
#include "../../../../common/op_host/op_api/matmul_util.cpp"

// batch_matmul_util.cpp 与 matmul_util.cpp 的匿名命名空间静态符号同名
// （NUM_TWO/BLOCK_CUBE/SetTensorToNDFormat/CheckAscendCScenario 等），将前者包进命名命名空间隔离；
// 其匿名命名空间成员仍可经 bmm_util_direct_ut:: 限定访问，公开函数经 bmm_util_direct_ut::Ops::NN:: 访问。
namespace bmm_util_direct_ut {
// 预开 wrapper::Ops（直通全局 Ops::Base/Ops::NN）：使 cpp 内 Ops::Base::Ceil* using 声明、
// Ops::NN::IsTranspose* 限定调用（L950/L1249 等）与非限定助手调用（NeedEnableFp32Output 等）均可解析；
// cpp 自身 Ops::NN 定义随后重开同一命名空间。
namespace Ops {
using namespace ::Ops;
namespace NN {
using namespace ::Ops::NN;
} // namespace NN
} // namespace Ops
#include "../../../../common/op_host/op_api/batch_matmul_util.cpp"
} // namespace bmm_util_direct_ut
#undef rtGetSocSpec // 恢复宏，避免影响本 TU 其余代码

namespace bmmu = bmm_util_direct_ut;            // 匿名命名空间成员别名
namespace bmmunn = bmm_util_direct_ut::Ops::NN; // 隔离命名空间内公开函数别名

using op::NpuArchManager;

namespace {

// RAII 保存/恢复桩平台 AI Core 数，避免用例间相互污染
struct CoreNumGuard {
    explicit CoreNumGuard(uint32_t newNum) : old_(op::GetCurrentPlatformInfoMock().coreNum_)
    {
        op::SetCubeCoreNum(newNum);
    }
    ~CoreNumGuard() { op::SetCubeCoreNum(old_); }
    uint32_t old_;
};

// 构造 FP16 输入的 MmOpInfo（供 GetBatchMatmulOp 直调分发用例，字段与 GetBatchMatmulOpInfo 对 FP16 的产出对齐）
MmOpInfo MakeFp16MmInfo(op::DataType outDtype, op::Format selfFormat)
{
    MmOpInfo info{};
    info.ori_info.self_dtype = op::DataType::DT_FLOAT16;
    info.ori_info.self_format = selfFormat;
    info.ori_info.mat2_dtype = op::DataType::DT_FLOAT16;
    info.ori_info.mat2_format = op::Format::FORMAT_ND;
    info.ori_info.output_dtype = outDtype;
    info.ori_info.output_format = op::Format::FORMAT_ND;
    info.support_info = info.ori_info;
    info.opImplModeEnum = 0x1;
    return info;
}

// 在连续 storage 上创建指定 viewShape/strides 的非连续视图
const aclTensor* MakeViewTensor(aclOpExecutor* executor, op::DataType dtype, const op::Shape& storageShape,
                                const op::Shape& viewShape, const op::Strides& strides)
{
    auto* storage = executor->AllocTensor(storageShape, dtype, op::Format::FORMAT_ND);
    EXPECT_NE(storage, nullptr);
    return executor->CreateView(storage, viewShape, storageShape, strides, 0);
}

constexpr int8_t kCubeMathKeepDtype = 0; // KEEP_DTYPE
constexpr int8_t kCubeMathUseHf32 = 3;   // USE_HF32

} // namespace

// ===================== L96~107 SetTensorToNDFormat =====================

// L102~104：storage 格式非 NZ/ND（NCHW）时三种格式均被改写为 ND
TEST_F(BatchMatmulL0OpDirectTest, Util_SetTensorToNDFormat_NCHWToND)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* tensor = MakeTensor(uniqueExecutor.get(), {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_NCHW);
    ASSERT_NE(tensor, nullptr);
    auto* result = bmmu::SetTensorToNDFormat(tensor);
    EXPECT_EQ(result, tensor);
    EXPECT_EQ(result->GetStorageFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(result->GetViewFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(result->GetOriginalFormat(), op::Format::FORMAT_ND);
}

// L100~101：ND / NZ 两种格式直通不改写
TEST_F(BatchMatmulL0OpDirectTest, Util_SetTensorToNDFormat_NdAndNzUnchanged)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* ndTensor = MakeTensor(uniqueExecutor.get(), {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* nzTensor = MakeTensor(uniqueExecutor.get(), {2, 64, 128}, op::DataType::DT_FLOAT16,
                                op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(ndTensor, nullptr);
    ASSERT_NE(nzTensor, nullptr);
    EXPECT_EQ(bmmu::SetTensorToNDFormat(ndTensor)->GetStorageFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(bmmu::SetTensorToNDFormat(nzTensor)->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
}

// ===================== L327~347 CheckShapeEqualToMul =====================

// L340~346：batch>=128、n 不在 (32/dataSize, 256/dataSize] 区间、非 1、UB 容量满足、n 非 256B 对齐 -> true
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckShapeEqualToMul_AllPass_True)
{
    // alignM=16, alignN=16: (16+16+16*16)*2=576 <= 253952; 8%128=8 != 0
    EXPECT_TRUE(bmmu::CheckShapeEqualToMul(16, 8, 128, 2, 16));
}

// L340~344：alignM+alignN+alignM*alignN 超出 UB 容量 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckShapeEqualToMul_UbOverflow_False)
{
    // n=1280 跳过区间分支; (512+1280+512*1280)*2 远超 UB
    EXPECT_FALSE(bmmu::CheckShapeEqualToMul(512, 1280, 128, 2, 16));
}

// L346：n 为 256B 对齐（n % (256/dataSize) == 0，且 n>256/dataSize 跳过区间分支）-> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckShapeEqualToMul_NAligned256B_False)
{
    EXPECT_FALSE(bmmu::CheckShapeEqualToMul(16, 256, 128, 2, 16));
}

// ===================== L441~453 GetBatchMatmulOp FP16/BF16 分发尾 =====================

// L441~443：FP16 入、FP16 出、self_format=NZ -> BatchMatMulNzFp162Fp16
TEST_F(BatchMatmulL0OpDirectTest, Util_GetBatchMatmulOp_Fp16NzToFp16Nz)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    auto info = MakeFp16MmInfo(op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = bmmu::GetBatchMatmulOp(x1, x2, nullptr, info, false, false, false, executor, false);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT16);
    EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
}

// L447~449：FP16 入、FP32 出、self_format=ND，且 arch 非 2201（bmm16In32OutRorA2 不拦截）-> BatchMatMulNdFp162Fp32
TEST_F(BatchMatmulL0OpDirectTest, Util_GetBatchMatmulOp_Fp16NdToFp32Nd_ArchNot2201)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* x2 = MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    auto info = MakeFp16MmInfo(op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* out = bmmu::GetBatchMatmulOp(x1, x2, nullptr, info, false, false, false, executor, false);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
    EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_ND);
    EXPECT_EQ(out->GetViewShape(), op::Shape({2, 64, 64}));
}

// L450~452：FP16 入、FP32 出、self_format=NZ -> BatchMatMulNzFp162Fp32
TEST_F(BatchMatmulL0OpDirectTest, Util_GetBatchMatmulOp_Fp16NzToFp32Nz)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeTensor(executor, {2, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    auto* x2 = MakeTensor(executor, {2, 128, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    auto info = MakeFp16MmInfo(op::DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ);
    auto* out = bmmu::GetBatchMatmulOp(x1, x2, nullptr, info, false, false, false, executor, false);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_FLOAT);
    EXPECT_EQ(out->GetStorageFormat(), op::Format::FORMAT_FRACTAL_NZ);
}

// ===================== L475~514 CheckTransNonContiguousShapeSupport =====================

// L504~513：L0 装不下整批但 L1 可容纳，且批负载均衡率 < 0.8 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckTransNonContig_L0NotFit_BalanceLow_False)
{
    CoreNumGuard coreGuard(16);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // m=320,k=64,n=128: L0a 溢出(81920>65536)、L1 满足；iterBatchL1=4, avg=81/16, max=8, rate=0.63<0.8
    auto* self = MakeTensor(executor, {81, 320, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {81, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckTransNonContiguousShapeSupport(self, mat2, nullptr));
}

// L500~516：L0 装不下但均衡率达标 -> true
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckTransNonContig_L0NotFit_BalanceOk_True)
{
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // m=512,k=128,n=512(fp16): L0a=262144>65536 溢出; L1=(65536+65536)*4=524288 满足; iterBatchL1=1, rate=1.0
    auto* self = MakeTensor(executor, {256, 512, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {256, 128, 512}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_TRUE(bmmu::CheckTransNonContiguousShapeSupport(self, mat2, nullptr));
}

// ===================== L546~566 CheckMergeBatchNonContiguousShapeSupport =====================

// L546~569：默认 adj（false,false）全约束满足 -> true（含 rtGetSocSpec 与 L0 容量检查）
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckMergeBatch_Fit_True_DefaultAdj)
{
    CoreNumGuard coreGuard(2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {128, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {128, 64, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_TRUE(bmmu::CheckMergeBatchNonContiguousShapeSupport(self, mat2, nullptr, false, false, kCubeMathKeepDtype));
}

// L547~548：adjX1=true 且 m>1 时 tempAlignM 走 4*CeilAlign(m,16) 分支 -> true
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckMergeBatch_Fit_True_AdjX1)
{
    CoreNumGuard coreGuard(2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {128, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {128, 64, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_TRUE(bmmu::CheckMergeBatchNonContiguousShapeSupport(self, mat2, nullptr, true, true, kCubeMathKeepDtype));
}

// L551~553：adjX1=false 且 adjX2=true 时跳过 minBaseK 对齐分支 -> true
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckMergeBatch_Fit_True_AdjX2Only)
{
    CoreNumGuard coreGuard(2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {128, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {128, 64, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_TRUE(bmmu::CheckMergeBatchNonContiguousShapeSupport(self, mat2, nullptr, false, true, kCubeMathKeepDtype));
}

// L566~567：L0c 容量溢出（tempAlignM*tempAlignN*8 > l0cSize）-> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckMergeBatch_L0cOverflow_False)
{
    CoreNumGuard coreGuard(2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // m=64,k=64,n=128: tempAlignM=256, tempAlignN=512 -> 256*512*8=1048576 > 262144
    auto* self = MakeTensor(executor, {128, 64, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {128, 64, 128}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckMergeBatchNonContiguousShapeSupport(self, mat2, nullptr, false, false, kCubeMathKeepDtype));
}

// L527~530：左右矩阵 batch 维不相等 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckMergeBatch_BatchNotEqual_False)
{
    CoreNumGuard coreGuard(2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {2, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {4, 64, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckMergeBatchNonContiguousShapeSupport(self, mat2, nullptr, false, false, kCubeMathKeepDtype));
}

// ===================== L584~610 CheckStreamKNonContiguousShapeSupport =====================
// 说明：桩 GetVectorCoreNum 恒 0，仅 aicoreNum==0 时可通过 aiv==2*aic 检查（L581），
//       此时末行 batchNum*mCnt*nCnt>=1 > 0 恒成立（L607），故 L610 return true 在本框架不可达。

// L590~593：FP32 且未开 HF32 且 K > 200 万 -> false（保精度走基础模板）
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckStreamK_Fp32Keep_LargeK_False)
{
    CoreNumGuard coreGuard(0);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {4, 16, 2000001}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {4, 2000001, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckStreamKNonContiguousShapeSupport(self, mat2, kCubeMathKeepDtype));
}

// L595~597：K 过小（CeilAlign(k,256) < 阈值）-> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckStreamK_Fp16_SmallK_False)
{
    CoreNumGuard coreGuard(0);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {4, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {4, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckStreamKNonContiguousShapeSupport(self, mat2, kCubeMathKeepDtype));
}

// L600~608：FP16 大 K 通过阈值检查，但 bmn 切分数不满足 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckStreamK_Fp16_False)
{
    CoreNumGuard coreGuard(0);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {4, 16, 8192}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {4, 8192, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckStreamKNonContiguousShapeSupport(self, mat2, kCubeMathKeepDtype));
}

// L601~602：FP32 未开 HF32 时基本块对齐值改用 32 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckStreamK_Fp32Keep_Align32_False)
{
    CoreNumGuard coreGuard(0);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {4, 16, 8192}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {4, 8192, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckStreamKNonContiguousShapeSupport(self, mat2, kCubeMathKeepDtype));
}

// ===================== L613~664 CheckNonContiguousTranspose =====================
// 非连续视图约定：mat2 视图恒为 bkn；[1,0,2] 模式 strides={n, b*n, 1}（不需换轴），
// [2,0,1] 模式 strides={k, 1, b*k}（需要换内两轴）。IsTransposeNonContiguous 仅在 3510 arch 生效。

// L644~646：命中 mergebatch 模板 -> CONTINUOUS
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_MergeHit_Continuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // self 连续 {b=128,m=16,k=64}; mat2 视图 bkn {128,64,16} [2,0,1]
    auto* self = MakeTensor(executor, {128, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {16, 128, 64}, {128, 64, 16},
                                op::Strides({64, 1, 8192}));
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    // adj 语义与真实调用点一致：IsTransposeLastTwoDims(连续张量)=false
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, false, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::CONTINUOUS);
}

// L648~651：A 连续但 iterbatch 校验不过（batch <= aicore）-> CONTINUOUS
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_AContig_IterBatchMiss_Continuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // self 连续 {4,16,64}; mat2 视图 bkn {4,64,16} [1,0,2]: strides={n, b*n, 1}={16,64,1}
    auto* self = MakeTensor(executor, {4, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {64, 4, 16}, {4, 64, 16}, op::Strides({16, 64, 1}));
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, true, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::CONTINUOUS);
}

// L653~656：A 非连续转置且 iterbatch 校验通过 -> CONTINUOUS（仅支持多 batch 载入模板）
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_ATranspose_IterBatchHit_Continuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // self 视图 {b=128,m=64,k=16} [1,0,2]: strides={k, b*k, 1}={16,2048,1}, storage {m,b,k}
    auto* self = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {64, 128, 16}, {128, 64, 16},
                                op::Strides({16, 2048, 1}));
    // mat2 视图 bkn {128,16,16} [2,0,1]: strides={k, 1, b*k}={16,1,2048}, storage {n,b,k}
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {16, 128, 16}, {128, 16, 16},
                                op::Strides({16, 1, 2048}));
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, false, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::CONTINUOUS);
}

// L657~660：A 非连续转置且 iterbatch 不过 -> AB_NON_CONTINUOUS
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_ATranspose_ABNonContinuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {64, 4, 16}, {4, 64, 16}, op::Strides({16, 64, 1}));
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {16, 4, 16}, {4, 16, 16}, op::Strides({16, 1, 64}));
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, false, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::AB_NON_CONTINUOUS);
    EXPECT_FALSE(swapA);
    EXPECT_TRUE(swapB);
}

// L661~663：仅 B 非连续转置且 iterbatch 通过 -> B_NON_CONTINUOUS
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_BNonContinuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // self 连续 {128,64,16}（m>n 使 mergebatch 不命中）; mat2 视图 bkn {128,16,16} [1,0,2]
    auto* self = MakeTensor(executor, {128, 64, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {16, 128, 16}, {128, 16, 16},
                                op::Strides({16, 2048, 1}));
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, true, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::B_NON_CONTINUOUS);
    EXPECT_FALSE(swapB);
}

// L625~627（附加覆盖）：A/B 均非连续转置且 A 不需换轴、B 也不需换轴 -> CONTINUOUS 拦截
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_BothTransposedNoSwap_Continuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    // self [1,0,2] {4,16,64}: strides={k, b*k, 1}={64,256,1}; mat2 [1,0,2] {4,64,16}: strides={16,64,1}
    auto* self = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {16, 4, 64}, {4, 16, 64},
                                op::Strides({64, 256, 1}));
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT16, {64, 4, 16}, {4, 64, 16}, op::Strides({16, 64, 1}));
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, false, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::CONTINUOUS);
}

// L638~641（附加覆盖）：转置场景下左右 dtype 不一致 -> CONTINUOUS
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_DtypeMismatch_Continuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {128, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 128, 64}, {128, 64, 16},
                                op::Strides({64, 1, 8192}));
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, true, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::CONTINUOUS);
}

// ===================== L666~673 CheckSocIfBatchMatMulToMulDefault / L675~682 路由 =====================

// L666~672：默认 SoC 检查函数直调恒 false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckSocIfBatchMatMulToMulDefault_False)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* self = MakeTensor(uniqueExecutor.get(), {2, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(uniqueExecutor.get(), {2, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckSocIfBatchMatMulToMulDefault(self, mat2, false, false));
}

// L675~681：arch 不在函数映射表（DAV_1001）时路由到 CheckSocIfBatchMatMulToMulDefault
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckArchIfBatchMatMulToMul_UnmappedArch_RoutesDefault)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* self = MakeTensor(uniqueExecutor.get(), {2, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(uniqueExecutor.get(), {2, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckArchIfBatchMatMulToMul(self, mat2, false, false));
}

// ===================== L1210~1270 checkFusedmm =====================

// L1217~1219：空 tensor 拦截 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_EmptyTensor_False)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {2, 0, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {2, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    ASSERT_NE(alpha, nullptr);
    ASSERT_NE(beta, nullptr);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1231~1233：self/mat2 非三维 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_Non3Dim_False)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1235~1237：bias 维度非 2/3 维 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_BiasDimUnsupported_False)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {2, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {2, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1239~1242：NZ 格式拦截 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_NzFormat_False)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {2, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {2, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ);
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1248~1250：mat2 非连续转置校验不过（连续张量）-> false（需 3510 arch 与 iterbatch 通过）
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_Mat2Contiguous_False)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1258~1260：二维 bias 形状与 aM/bN 不匹配 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_2dBiasShapeMismatch_False)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {8, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    // mat2 视图 bkn {128,16,16} [2,0,1]: strides={k, 1, b*k}={16,1,2048}
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 128, 16}, {128, 16, 16},
                                op::Strides({16, 1, 2048}));
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1262~1265：三维 bias 形状与 aM/bN 不匹配 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_3dBiasShapeMismatch_False)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {1, 8, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 128, 16}, {128, 16, 16},
                                op::Strides({16, 1, 2048}));
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1252~1269：三维 bias 形状完全匹配 -> true（覆盖全部形状校验与成功日志）
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_3dBiasShapeMatch_True)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 128, 16}, {128, 16, 16},
                                op::Strides({16, 1, 2048}));
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_TRUE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
    EXPECT_TRUE(swapB);
}

// ===================== L1272~1324 ExecFusedmmOp =====================

// L1278~1287 换轴分支 + L1301~1303 转置 self 分支 + L1312~1314 bias 处理：全链路直调 -> 非空输出
TEST_F(BatchMatmulL0OpDirectTest, Util_ExecFusedmmOp_SwapAndBias)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {128, 16, 32}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    // mat2 视图 bkn {128,32,16} [2,0,1]: strides={32,1,4096}, storage {n,b,k}={16,128,32}
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 128, 32}, {128, 32, 16},
                                op::Strides({32, 1, 4096}));
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* out = bmmunn::ExecFusedmmOp(bias, self, mat2, kCubeMathUseHf32, true, executor);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape(), op::Shape({128, 16, 16}));
}

// L1276~1277 不换轴分支 + L1303~1305 Contiguous 分支（self 为 [1,0,2] 非连续视图，
// IsTransposeLastTwoDims=false）：全链路直调 -> 非空输出
TEST_F(BatchMatmulL0OpDirectTest, Util_ExecFusedmmOp_NoSwapViewSelf)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    // self 视图 {b,m,k}={128,16,32} [1,0,2]: strides={k, b*k, 1}={32,4096,1}, storage {m,b,k}={16,128,32}
    auto* self = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 128, 32}, {128, 16, 32},
                                op::Strides({32, 4096, 1}));
    auto* mat2 = MakeTensor(executor, {128, 32, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* out = bmmunn::ExecFusedmmOp(bias, self, mat2, kCubeMathUseHf32, false, executor);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape(), op::Shape({128, 16, 16}));
}

// ===================== 补充分支用例（覆盖编译变体差异与遗漏出口） =====================

// L331~333：batch 数不足 128 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckShapeEqualToMul_BatchTooSmall_False)
{
    EXPECT_FALSE(bmmu::CheckShapeEqualToMul(64, 64, 64, 2, 16));
}

// L334~336：nDim 落在 (32/dataSize, 256/dataSize] 区间 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckShapeEqualToMul_NInRange_False)
{
    EXPECT_FALSE(bmmu::CheckShapeEqualToMul(64, 64, 128, 2, 16));
}

// L337~339：nDim == 1 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckShapeEqualToMul_NDimOne_False)
{
    EXPECT_FALSE(bmmu::CheckShapeEqualToMul(64, 1, 128, 2, 16));
}

// L349~389：CheckArchIfBatchMatMulToMulDav3510 直调（batch 数小于 aicore 数 -> 不命中 iterbatch，
// CheckShapeEqualToMul 亦不满足 -> 返回 true 表示可转 Mul）
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckArchIfBMMToMulDav3510_Direct)
{
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {4, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    // batchNum=4 < aicoreNum=8 -> fitIterBatch=false；batchNum=4 < 128 -> fitBatchMatMulToMul=false
    EXPECT_TRUE(bmmu::CheckArchIfBatchMatMulToMulDav3510(self, mat2, false, false));
}

// L434~435：BF16 输入命中条件第二个子表达式，ND->ND 走 BatchMatMulNd（439）
TEST_F(BatchMatmulL0OpDirectTest, Util_GetBatchMatmulOp_Bf16NdToBf16Nd_ArchNot2201)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* x1 = MakeTensor(executor, {2, 64, 128}, op::DataType::DT_BF16, op::Format::FORMAT_ND);
    auto* x2 = MakeTensor(executor, {2, 128, 64}, op::DataType::DT_BF16, op::Format::FORMAT_ND);
    ASSERT_NE(x1, nullptr);
    ASSERT_NE(x2, nullptr);
    MmOpInfo info{};
    info.ori_info.self_dtype = op::DataType::DT_BF16;
    info.ori_info.self_format = op::Format::FORMAT_ND;
    info.ori_info.mat2_dtype = op::DataType::DT_BF16;
    info.ori_info.mat2_format = op::Format::FORMAT_ND;
    info.ori_info.output_dtype = op::DataType::DT_BF16;
    info.ori_info.output_format = op::Format::FORMAT_ND;
    info.support_info = info.ori_info;
    info.opImplModeEnum = 0x1;
    auto* out = bmmu::GetBatchMatmulOp(x1, x2, nullptr, info, false, false, false, executor, false);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDataType(), op::DataType::DT_BF16);
    EXPECT_EQ(out->GetViewShape(), op::Shape({2, 64, 64}));
}

// L466~467：aicoreNum 为 0（桩默认值）-> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckTransNonContig_ZeroAicore_False)
{
    CoreNumGuard coreGuard(0);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {128, 16, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckTransNonContiguousShapeSupport(self, mat2, nullptr));
}

// L524~525：FP32 输入且未开 HF32（KEEP_DTYPE）-> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckMergeBatch_Fp32KeepDtype_False)
{
    CoreNumGuard coreGuard(2);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {128, 16, 64}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {128, 64, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckMergeBatchNonContiguousShapeSupport(self, mat2, nullptr, false, false, kCubeMathKeepDtype));
}

// L574~577：左右矩阵 batch 维不相等 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckStreamK_BatchNotEqual_False)
{
    CoreNumGuard coreGuard(0);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {2, 16, 8192}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {4, 8192, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    EXPECT_FALSE(bmmu::CheckStreamKNonContiguousShapeSupport(self, mat2, kCubeMathKeepDtype));
}

// L630~631：mat2 非转置（连续张量）-> CONTINUOUS
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckNonContigTranspose_BNotTransposed_Continuous)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {4, 16, 64}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {4, 64, 16}, op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    bool swapA = false;
    bool swapB = false;
    auto mode = bmmu::CheckNonContiguousTranspose(self, mat2, swapA, swapB, nullptr, false, false, kCubeMathKeepDtype);
    EXPECT_EQ(mode, bmmunn::NonContiguousMode::CONTINUOUS);
}

// L1214~1215：bias 空指针拦截 -> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_NullBias_False)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* self = MakeTensor(executor, {2, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeTensor(executor, {2, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(nullptr, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1244~1246：iterbatch 校验不过（batch < aicore）-> false
TEST_F(BatchMatmulL0OpDirectTest, Util_CheckFusedmm_IterBatchMiss_False)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* self = MakeTensor(executor, {4, 16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 4, 16}, {4, 16, 16}, op::Strides({16, 1, 64}));
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* alpha = executor->AllocScalar(1.0f);
    auto* beta = executor->AllocScalar(1.0f);
    bool swapB = false;
    EXPECT_FALSE(bmmunn::checkFusedmm(bias, self, mat2, alpha, beta, kCubeMathUseHf32, swapB));
}

// L1301~1303：self 为转置视图（IsTransposeLastTwoDims=true）时 CreateView 换轴分支 -> 非空输出
TEST_F(BatchMatmulL0OpDirectTest, Util_ExecFusedmmOp_TransposedSelf)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    ASSERT_NE(uniqueExecutor.get(), nullptr);
    auto* executor = uniqueExecutor.get();
    auto* bias = MakeTensor(executor, {16, 16}, op::DataType::DT_FLOAT, op::Format::FORMAT_ND);
    // self 视图 {b,m,k}={128,16,32} 为 {b,k,m} 连续 storage 的转置视图: strides={mk, 1, m}={512,1,16}
    auto* self = MakeViewTensor(executor, op::DataType::DT_FLOAT, {128, 32, 16}, {128, 16, 32},
                                op::Strides({512, 1, 16}));
    // mat2 视图 bkn {128,32,16} [2,0,1]: strides={32,1,4096}, storage {16,128,32}
    auto* mat2 = MakeViewTensor(executor, op::DataType::DT_FLOAT, {16, 128, 32}, {128, 32, 16},
                                op::Strides({32, 1, 4096}));
    ASSERT_NE(bias, nullptr);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(mat2, nullptr);
    auto* out = bmmunn::ExecFusedmmOp(bias, self, mat2, kCubeMathUseHf32, true, executor);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetViewShape(), op::Shape({128, 16, 16}));
}

// ===================== 公开入口（aclnnBaddbmm -> checkFusedmm/ExecFusedmmOp，so 侧同源实现） =====================

// checkFusedmm 全通过 -> ExecFusedmmOp 全链路（3510 arch + iterbatch 满足 + mat2 非连续转置视图）
TEST_F(BatchMatmulL0OpDirectTest, baddbmm_fusedmm_3510_end2end)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto batch1 = TensorDesc({128, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    // mat2 视图 bkn {128,32,16} [2,0,1]: strides={32,1,4096}, storage {16,128,32}
    auto batch2 = TensorDesc({128, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND, {32, 1, 4096}, 0, {16, 128, 32})
                      .ValueRange(-2, 2);
    auto out = TensorDesc({128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = kCubeMathUseHf32;
    auto ut = OP_API_UT(aclnnBaddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// checkFusedmm 在 mat2 连续时于 L1249~1250 返回 false，回落常规 baddbmm 链路
TEST_F(BatchMatmulL0OpDirectTest, baddbmm_fusedmm_Mat2Contiguous_Fallback)
{
    NpuArchManager archManager(NpuArch::DAV_3510);
    CoreNumGuard coreGuard(8);
    auto self = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto batch1 = TensorDesc({128, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto batch2 = TensorDesc({128, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out = TensorDesc({128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto beta = ScalarDesc(1.0f);
    auto alpha = ScalarDesc(1.0f);
    int8_t cubeMathType = kCubeMathUseHf32;
    auto ut = OP_API_UT(aclnnBaddbmm, INPUT(self, batch1, batch2, beta, alpha), OUTPUT(out), cubeMathType);
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ===================== cube_util.cpp L194~312 dtype 映射直调（PromoteType/CubeMathType 系列） =====================
// 覆盖目标（cube_util.cpp 已随 L2105 显式 include 进入本 TU，被测函数位于 Ops::NN）：
//   CalcPromoteTypeCubemathtype L194~198（default：不支持 dtype -> OP_LOGE + DT_UNDEFINED）
//   CalcUseFp16PromoteType      L201~221（FP16/FLOAT->FP16；BF16 警告->FP16；HIF8/FP8_E4M3FN
//   警告保原；default->UNDEFINED） CalcUseHf32PromoteType      L223~240（非 FLOAT 四种 dtype 警告保原；FLOAT
//   直通；default->UNDEFINED） CalcAllowFp32DownPrecisionPromoteType L242~265（FP16/BF16 直通；FLOAT 按
//   IsCubeSupportHf32 两态分流；HIF8/FP8 警告保原；default->UNDEFINED） CalcKeepDtypePromoteType    L267~281（五种支持
//   dtype 直通；default 报错后仍返回原 dtype，非 UNDEFINED） CalcForceGrpAccForFp32PromoteType L283~293（FLOAT
//   直通；default 报错后仍返回原 dtype） CalcPromoteTypeCubeMathTypeNew L296~312（6 个 cubeMathType 路由 + 未匹配 LOGW
//   保原）
// 平台两态注入：IsCubeSupportHf32 为 cube_util.h 内联函数，取值由 NpuArchManager RAII 驱动的
//   GetCurNpuArch() 决定（DAV_2201/3510/3002 返 true，其余如 DAV_1001 返 false），手法同上文 Util_GetBatchMatmulOp_*
//   用例。
// 全部为纯 dtype 入参直调，无需构造 tensor/executor；default/警告分支仅产生日志，不影响断言。

// CalcPromoteTypeCubemathtype L194~198：不支持的 dtype（INT8/STRING）且 cubeMathType != USE_FP16 -> OP_LOGE +
// DT_UNDEFINED
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcPromoteTypeCubemathtype_UnsupportedDtype_Undefined)
{
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubemathtype(op::DataType::DT_INT8, op::KEEP_DTYPE), op::DataType::DT_UNDEFINED);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubemathtype(op::DataType::DT_STRING, op::ALLOW_FP32_DOWN_PRECISION),
              op::DataType::DT_UNDEFINED);
}

// CalcUseFp16PromoteType L204~206：FP16 / FP32 -> 统一降为 FP16
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcUseFp16PromoteType_Fp16AndFp32_ToFp16)
{
    EXPECT_EQ(Ops::NN::CalcUseFp16PromoteType(op::DataType::DT_FLOAT16), op::DataType::DT_FLOAT16);
    EXPECT_EQ(Ops::NN::CalcUseFp16PromoteType(op::DataType::DT_FLOAT), op::DataType::DT_FLOAT16);
}

// CalcUseFp16PromoteType L207~210：BF16 -> OP_LOGW 警告 + 返回 FP16
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcUseFp16PromoteType_Bf16WarnToFp16)
{
    EXPECT_EQ(Ops::NN::CalcUseFp16PromoteType(op::DataType::DT_BF16), op::DataType::DT_FLOAT16);
}

// CalcUseFp16PromoteType L211~215：HIF8 / FP8_E4M3FN -> OP_LOGW 警告 + 返回原 dtype（不降级）
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcUseFp16PromoteType_Hif8Fp8WarnKeepSelf)
{
    EXPECT_EQ(Ops::NN::CalcUseFp16PromoteType(op::DataType::DT_HIFLOAT8), op::DataType::DT_HIFLOAT8);
    EXPECT_EQ(Ops::NN::CalcUseFp16PromoteType(op::DataType::DT_FLOAT8_E4M3FN), op::DataType::DT_FLOAT8_E4M3FN);
}

// CalcUseFp16PromoteType L216~219：不支持的 dtype（INT8）-> OP_LOGE + DT_UNDEFINED
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcUseFp16PromoteType_UnsupportedDtype_Undefined)
{
    EXPECT_EQ(Ops::NN::CalcUseFp16PromoteType(op::DataType::DT_INT8), op::DataType::DT_UNDEFINED);
}

// CalcUseHf32PromoteType L226~232：FP16/BF16/HIF8/FP8_E4M3FN -> OP_LOGW 警告 + 返回原 dtype
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcUseHf32PromoteType_NonFloatWarnKeepSelf)
{
    EXPECT_EQ(Ops::NN::CalcUseHf32PromoteType(op::DataType::DT_FLOAT16), op::DataType::DT_FLOAT16);
    EXPECT_EQ(Ops::NN::CalcUseHf32PromoteType(op::DataType::DT_BF16), op::DataType::DT_BF16);
    EXPECT_EQ(Ops::NN::CalcUseHf32PromoteType(op::DataType::DT_HIFLOAT8), op::DataType::DT_HIFLOAT8);
    EXPECT_EQ(Ops::NN::CalcUseHf32PromoteType(op::DataType::DT_FLOAT8_E4M3FN), op::DataType::DT_FLOAT8_E4M3FN);
}

// CalcUseHf32PromoteType L233~234：FP32 -> 保持 FP32
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcUseHf32PromoteType_FloatPassthrough)
{
    EXPECT_EQ(Ops::NN::CalcUseHf32PromoteType(op::DataType::DT_FLOAT), op::DataType::DT_FLOAT);
}

// CalcUseHf32PromoteType L235~238：不支持的 dtype（INT8）-> OP_LOGE + DT_UNDEFINED
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcUseHf32PromoteType_UnsupportedDtype_Undefined)
{
    EXPECT_EQ(Ops::NN::CalcUseHf32PromoteType(op::DataType::DT_INT8), op::DataType::DT_UNDEFINED);
}

// CalcAllowFp32DownPrecisionPromoteType L245~247：FP16 / BF16 -> 返回原 dtype
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcAllowFp32DownPrecisionPromoteType_Fp16Bf16Passthrough)
{
    EXPECT_EQ(Ops::NN::CalcAllowFp32DownPrecisionPromoteType(op::DataType::DT_FLOAT16), op::DataType::DT_FLOAT16);
    EXPECT_EQ(Ops::NN::CalcAllowFp32DownPrecisionPromoteType(op::DataType::DT_BF16), op::DataType::DT_BF16);
}

// CalcAllowFp32DownPrecisionPromoteType L249~251：FP32 且 IsCubeSupportHf32()=true（DAV_2201）-> 保持 FP32
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcAllowFp32DownPrecisionPromoteType_Float_Hf32Supported_KeepFp32)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_EQ(Ops::NN::CalcAllowFp32DownPrecisionPromoteType(op::DataType::DT_FLOAT), op::DataType::DT_FLOAT);
}

// CalcAllowFp32DownPrecisionPromoteType L252~254：FP32 且 IsCubeSupportHf32()=false（DAV_1001）-> OP_LOGD + 降为 FP16
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcAllowFp32DownPrecisionPromoteType_Float_Hf32NotSupported_DownToFp16)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    EXPECT_EQ(Ops::NN::CalcAllowFp32DownPrecisionPromoteType(op::DataType::DT_FLOAT), op::DataType::DT_FLOAT16);
}

// CalcAllowFp32DownPrecisionPromoteType L255~259：HIF8 / FP8_E4M3FN -> OP_LOGW 警告 + 返回原 dtype
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcAllowFp32DownPrecisionPromoteType_Hif8Fp8WarnKeepSelf)
{
    EXPECT_EQ(Ops::NN::CalcAllowFp32DownPrecisionPromoteType(op::DataType::DT_HIFLOAT8), op::DataType::DT_HIFLOAT8);
    EXPECT_EQ(Ops::NN::CalcAllowFp32DownPrecisionPromoteType(op::DataType::DT_FLOAT8_E4M3FN),
              op::DataType::DT_FLOAT8_E4M3FN);
}

// CalcAllowFp32DownPrecisionPromoteType L260~263：不支持的 dtype（INT8）-> OP_LOGE + DT_UNDEFINED
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcAllowFp32DownPrecisionPromoteType_UnsupportedDtype_Undefined)
{
    EXPECT_EQ(Ops::NN::CalcAllowFp32DownPrecisionPromoteType(op::DataType::DT_INT8), op::DataType::DT_UNDEFINED);
}

// CalcKeepDtypePromoteType L270~275：FP16/BF16/FP32/HIF8/FP8_E4M3FN 五种支持 dtype -> 均返回原 dtype
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcKeepDtypePromoteType_SupportedDtypesPassthrough)
{
    EXPECT_EQ(Ops::NN::CalcKeepDtypePromoteType(op::DataType::DT_FLOAT16), op::DataType::DT_FLOAT16);
    EXPECT_EQ(Ops::NN::CalcKeepDtypePromoteType(op::DataType::DT_BF16), op::DataType::DT_BF16);
    EXPECT_EQ(Ops::NN::CalcKeepDtypePromoteType(op::DataType::DT_FLOAT), op::DataType::DT_FLOAT);
    EXPECT_EQ(Ops::NN::CalcKeepDtypePromoteType(op::DataType::DT_HIFLOAT8), op::DataType::DT_HIFLOAT8);
    EXPECT_EQ(Ops::NN::CalcKeepDtypePromoteType(op::DataType::DT_FLOAT8_E4M3FN), op::DataType::DT_FLOAT8_E4M3FN);
}

// CalcKeepDtypePromoteType L276~279：不支持的 dtype（INT8）-> OP_LOGE 后仍返回原 dtype（注意非 UNDEFINED）
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcKeepDtypePromoteType_UnsupportedDtype_ReturnSelf)
{
    EXPECT_EQ(Ops::NN::CalcKeepDtypePromoteType(op::DataType::DT_INT8), op::DataType::DT_INT8);
}

// CalcForceGrpAccForFp32PromoteType L286~287：FP32 -> 返回原 dtype
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcForceGrpAccForFp32PromoteType_FloatPassthrough)
{
    EXPECT_EQ(Ops::NN::CalcForceGrpAccForFp32PromoteType(op::DataType::DT_FLOAT), op::DataType::DT_FLOAT);
}

// CalcForceGrpAccForFp32PromoteType L288~291：非 FP32（FP16）-> OP_LOGE 后仍返回原 dtype
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcForceGrpAccForFp32PromoteType_UnsupportedDtype_ReturnSelf)
{
    EXPECT_EQ(Ops::NN::CalcForceGrpAccForFp32PromoteType(op::DataType::DT_FLOAT16), op::DataType::DT_FLOAT16);
}

// CalcPromoteTypeCubeMathTypeNew L299~311：6 个 cubeMathType 路由 + 未匹配分支
//   每个路由用与下游函数行为可区分的 dtype 断言，同时证明路由正确性与下游分支行为：
//   USE_FP16(2)+BF16 -> FP16（CalcUseFp16PromoteType 的 BF16 警告分支）；
//   USE_HF32(3)+FP16 -> FP16（CalcUseHf32PromoteType 的警告保原分支）；
//   ALLOW_FP32_DOWN_PRECISION(1)+FP32（DAV_2201，IsCubeSupportHf32=true）-> FP32；
//   KEEP_DTYPE(0)+HIF8 -> HIF8（CalcKeepDtypePromoteType 直通分支）；
//   USE_FP32_ADD(4)+FP16 -> FP16（CalcForceGrpAccForFp32PromoteType 的 default 报错保原分支）；
//   未匹配 0x7F+FP32 -> OP_LOGW + 返回原 dtype（L310~311）。
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcPromoteTypeCubeMathTypeNew_MathTypeRoutes)
{
    NpuArchManager archManager(NpuArch::DAV_2201);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubeMathTypeNew(op::DataType::DT_BF16, op::USE_FP16), op::DataType::DT_FLOAT16);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubeMathTypeNew(op::DataType::DT_FLOAT16, op::USE_HF32),
              op::DataType::DT_FLOAT16);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubeMathTypeNew(op::DataType::DT_FLOAT, op::ALLOW_FP32_DOWN_PRECISION),
              op::DataType::DT_FLOAT);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubeMathTypeNew(op::DataType::DT_HIFLOAT8, op::KEEP_DTYPE),
              op::DataType::DT_HIFLOAT8);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubeMathTypeNew(op::DataType::DT_FLOAT16, op::USE_FP32_ADD),
              op::DataType::DT_FLOAT16);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubeMathTypeNew(op::DataType::DT_FLOAT, static_cast<int8_t>(0x7F)),
              op::DataType::DT_FLOAT);
}

// CalcPromoteTypeCubeMathTypeNew L303~304：ALLOW_FP32_DOWN_PRECISION 路由 + IsCubeSupportHf32()=false（DAV_1001）
//   组合态：路由至 CalcAllowFp32DownPrecisionPromoteType 的 FP32 降级分支 -> FP16
TEST_F(BatchMatmulL0OpDirectTest, Util_CalcPromoteTypeCubeMathTypeNew_AllowDown_Hf32NotSupported_DownToFp16)
{
    NpuArchManager archManager(NpuArch::DAV_1001);
    EXPECT_EQ(Ops::NN::CalcPromoteTypeCubeMathTypeNew(op::DataType::DT_FLOAT, op::ALLOW_FP32_DOWN_PRECISION),
              op::DataType::DT_FLOAT16);
}

// NeedCubeGoHF32 L315~333（超出 194~312 目标区间，信息性补全）：IsCubeSupportHf32 两态 + BF16/FP16 警告分支
//   true 态（DAV_2201）：FP32+USE_HF32 -> true；FP32+ALLOW_FP32_DOWN_PRECISION -> true；
//     非 FP32（BF16/FP16，各触发 OP_LOGW）或非 HF32 类 cubeMathType（KEEP_DTYPE）-> false。
//   false 态（DAV_1001）：FP32+USE_HF32 / FP32+ALLOW_FP32_DOWN_PRECISION -> false。
TEST_F(BatchMatmulL0OpDirectTest, Util_NeedCubeGoHF32_TwoStates)
{
    {
        NpuArchManager archManager(NpuArch::DAV_2201);
        EXPECT_TRUE(Ops::NN::NeedCubeGoHF32(op::DataType::DT_FLOAT, op::USE_HF32));
        EXPECT_TRUE(Ops::NN::NeedCubeGoHF32(op::DataType::DT_FLOAT, op::ALLOW_FP32_DOWN_PRECISION));
        EXPECT_FALSE(Ops::NN::NeedCubeGoHF32(op::DataType::DT_BF16, op::USE_HF32));
        EXPECT_FALSE(Ops::NN::NeedCubeGoHF32(op::DataType::DT_FLOAT16, op::USE_HF32));
        EXPECT_FALSE(Ops::NN::NeedCubeGoHF32(op::DataType::DT_FLOAT, op::KEEP_DTYPE));
    }
    {
        NpuArchManager archManager(NpuArch::DAV_1001);
        EXPECT_FALSE(Ops::NN::NeedCubeGoHF32(op::DataType::DT_FLOAT, op::USE_HF32));
        EXPECT_FALSE(Ops::NN::NeedCubeGoHF32(op::DataType::DT_FLOAT, op::ALLOW_FP32_DOWN_PRECISION));
    }
}
