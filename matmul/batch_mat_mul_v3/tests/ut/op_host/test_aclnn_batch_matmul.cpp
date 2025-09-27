/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

using namespace std;
using namespace op;
class l2_batch_matmul_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "batch_matmul_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "batch_matmul_test TearDown" << endl;
    }
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
    auto out_tensor_desc =
        TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-2, 2).Precision(0.005, 0.005);

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
    auto out_tensor_desc =
        TensorDesc({1, 1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

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

// 1980 Fp16 16Aligin Nd
TEST_F(l2_batch_matmul_test, ascend910A_case_14)
{
    auto tensor_1_desc = TensorDesc({16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_1_desc_t =
        TensorDesc({16, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {512, 1, 16}, 0, {16, 32, 16}).ValueRange(-2, 2);

    auto tensor_2_desc = TensorDesc({16, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_2_desc_t =
        TensorDesc({16, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND, {2048, 1, 32}, 0, {16, 64, 32}).ValueRange(-2, 2);

    auto out_tensor_desc =
        TensorDesc({16, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 1980 Fp16 16notAligin Nd
TEST_F(l2_batch_matmul_test, ascend910A_case_18)
{
    auto tensor_1_desc = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_1_desc_t = TensorDesc({1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {6, 1, 2}, 0, {1, 3, 2}).ValueRange(-2, 2);

    auto tensor_2_desc = TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_2_desc_t =
        TensorDesc({1, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {12, 1, 3}, 0, {1, 4, 3}).ValueRange(-2, 2);

    auto out_tensor_desc = TensorDesc({1, 2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 1980 Fp32 16notAligin Nd
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

    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
}

// 1980 Fp32 16Aligin Nd
TEST_F(l2_batch_matmul_test, ascend910A_case_23)
{
    auto tensor_1_desc = TensorDesc({16, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_1_desc_t =
        TensorDesc({16, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND, {512, 1, 16}, 0, {16, 32, 16}).ValueRange(-2, 2);

    auto tensor_2_desc = TensorDesc({16, 32, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto tensor_2_desc_t =
        TensorDesc({16, 32, 64}, ACL_FLOAT, ACL_FORMAT_ND, {2048, 1, 32}, 0, {16, 64, 32}).ValueRange(-2, 2);

    auto out_tensor_desc = TensorDesc({16, 16, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 1971 Fp32 Nd
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
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 1971 Fp16 Nd  IsNdToNzOnTheFly = true,
TEST_F(l2_batch_matmul_test, ascend910B2_case_28)
{
    auto tensor_1_desc = TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t =
        TensorDesc({1, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND, {512, 1, 16}, 0, {1, 32, 16}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t =
        TensorDesc({1, 32, 64}, ACL_FLOAT16, ACL_FORMAT_ND, {2048, 1, 32}, 0, {1, 64, 32}).ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({1, 16, 64}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
}

// 1971 Fp16 Nd  IsNdToNzOnTheFly = false,
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
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
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

    auto tensor_1_desc_t =
        TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {16384, 1, 128}, 0, {2097152}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t =
        TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND, {16384, 1, 128}, 0, {2097152}).ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({128, 128, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
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

    auto tensor_1_desc_t =
        TensorDesc({4096, 128, 9}, ACL_BF16, ACL_FORMAT_ND, {1152, 1, 128}, 0, {4096, 9, 128}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({4096, 9, 8}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t =
        TensorDesc({4096, 9, 8}, ACL_BF16, ACL_FORMAT_ND, {72, 1, 9}, 0, {4096, 8, 9}).ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({4096, 128, 8}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_1)
{
    auto tensor_1_desc = TensorDesc({2048, 2, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t =
        TensorDesc({2048, 2, 64}, ACL_BF16, ACL_FORMAT_ND, {128, 1, 2}, 0, {2048, 64, 2}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({2048, 64, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t =
        TensorDesc({2048, 64, 3}, ACL_BF16, ACL_FORMAT_ND, {192, 1, 64}, 0, {2048, 3, 64}).ValueRange(0, 2);

    auto out_tensor_desc = TensorDesc({2048, 2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_2)
{
    auto tensor_1_desc = TensorDesc({3072, 1, 300}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t =
        TensorDesc({3072, 1, 300}, ACL_BF16, ACL_FORMAT_ND, {300, 1, 1}, 0, {3072, 300, 1}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({3072, 300, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t =
        TensorDesc({3072, 300, 16}, ACL_BF16, ACL_FORMAT_ND, {4800, 1, 300}, 0, {3072, 16, 300}).ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({3072, 1, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_3)
{
    auto tensor_1_desc = TensorDesc({3072, 1, 16}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t =
        TensorDesc({3072, 1, 16}, ACL_BF16, ACL_FORMAT_ND, {16, 1, 1}, 0, {3072, 16, 1}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({3072, 16, 60}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t =
        TensorDesc({3072, 16, 60}, ACL_BF16, ACL_FORMAT_ND, {960, 1, 16}, 0, {3072, 60, 16}).ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({3072, 1, 60}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_4)
{
    auto tensor_1_desc = TensorDesc({8, 1536, 2048}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({8, 2048, 7680}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({8, 1536, 7680}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_fp32_0)
{
    auto tensor_1_desc = TensorDesc({3000, 1, 1024}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_1_desc_t =
        TensorDesc({3000, 1, 1024}, ACL_FLOAT, ACL_FORMAT_ND, {1024, 1, 1}, 0, {3000, 1024, 1}).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({3000, 1024, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc_t =
        TensorDesc({3000, 1024, 128}, ACL_FLOAT, ACL_FORMAT_ND, {131072, 1, 1024}, 0, {3000, 128, 1024})
            .ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({3000, 1, 128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_false_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    auto ut_true_true =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc_t, tensor_2_desc_t), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_true_true.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910_95_test_bmm2mm)
{
    auto tensor_1_desc = TensorDesc({69, 16, 224}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensor_2_desc = TensorDesc({1, 224, 112}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc =
        TensorDesc({69, 16, 112}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
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
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_test, ascend910B2_bf16_6)
{
    auto tensor_1_desc = TensorDesc({8, 10, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto tensor_2_desc = TensorDesc({8, 1, 5000}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);

    auto out_tensor_desc =
        TensorDesc({8, 10, 5000}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2).Precision(0.0001, 0.0001);

    int8_t math_type = 1;
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = 0;
    auto ut_false_false =
        OP_API_UT(aclnnBatchMatMul, INPUT(tensor_1_desc, tensor_2_desc), OUTPUT(out_tensor_desc), math_type);
    aclRet = ut_false_false.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}