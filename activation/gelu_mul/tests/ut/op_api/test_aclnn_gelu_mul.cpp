/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "../../../op_api/aclnn_gelu_mul.h"

using namespace op;
using namespace std;

class l2_gelu_mul_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_gelu_mul_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_gelu_mul_test TearDown" << endl; }
};

// Phase one validates the non-contiguous execution plan; actual values and writeback need ST.
TEST_F(l2_gelu_mul_test, gelu_mul_non_contiguous_input_and_output)
{
    auto input = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND, {16, 2}, 0, {2, 16}).ValueRange(-3.0F, 3.0F);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {8, 2}, 0, {2, 8});
    char approximate[] = "none";
    auto ut = OP_API_UT(aclnnGeluMul, INPUT(input, approximate), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_gelu_mul_test, gelu_mul_non_contiguous_output_only)
{
    auto input = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-3.0F, 3.0F);
    auto out = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {8, 2}, 0, {2, 8});
    char approximate[] = "tanh";
    auto ut = OP_API_UT(aclnnGeluMul, INPUT(input, approximate), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_gelu_mul_test, gelu_mul_contiguous_base)
{
    auto input = TensorDesc({8, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-3.0F, 3.0F);
    auto out = TensorDesc({8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    char approximate[] = "none";
    auto ut = OP_API_UT(aclnnGeluMul, INPUT(input, approximate), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_gelu_mul_test, gelu_mul_self_nullptr)
{
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    char approximate[] = "none";
    auto ut = OP_API_UT(aclnnGeluMul, INPUT((aclTensor*)nullptr, approximate), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_gelu_mul_test, gelu_mul_out_dtype_mismatch)
{
    auto input = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-3.0F, 3.0F);
    auto out = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    char approximate[] = "none";
    auto ut = OP_API_UT(aclnnGeluMul, INPUT(input, approximate), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gelu_mul_test, gelu_mul_out_shape_not_match)
{
    auto input = TensorDesc({8, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-3.0F, 3.0F);
    auto out = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    char approximate[] = "none";
    auto ut = OP_API_UT(aclnnGeluMul, INPUT(input, approximate), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_gelu_mul_test, gelu_mul_out_last_dim_not_halved)
{
    auto input = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-3.0F, 3.0F);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    char approximate[] = "tanh";
    auto ut = OP_API_UT(aclnnGeluMul, INPUT(input, approximate), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
