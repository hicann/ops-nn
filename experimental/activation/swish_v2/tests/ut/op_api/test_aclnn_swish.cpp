/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>

#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_swish.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class swish_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "swish_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "swish_test TearDown" << std::endl; }
};

TEST_F(swish_test, case_001_fp32)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(swish_test, case_002_fp16)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto betaDesc = ScalarDesc(0.01f, ACL_FLOAT16);
    auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(swish_test, case_003_bf16)
{
    auto inputDesc = TensorDesc({19, 21}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto betaDesc = ScalarDesc(static_cast<int32_t>(-1));
    auto outDesc = TensorDesc({19, 21}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(swish_test, case_004_nullptr_beta)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto betaDesc = static_cast<aclScalar*>(nullptr);
    auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(swish_test, case_005_empty_tensor)
{
    auto inputDesc = TensorDesc({2, 0, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.0f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 0, 32}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(swish_test, case_006_uncontiguous)
{
    auto inputDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {16, 2}).ValueRange(-2, 2);
    auto betaDesc = ScalarDesc(static_cast<int8_t>(-1));
    auto outDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {16, 2}).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(swish_test, case_007_inconsistent_shape)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 16, 32, 18}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(swish_test, case_008_inconsistent_dtype)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(swish_test, case_009_invalid_input_dtype)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(swish_test, case_010_bool_beta)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(true);
    auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(swish_test, case_011_nullptr_input)
{
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwish, INPUT(static_cast<aclTensor*>(nullptr), betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(swish_test, case_012_nullptr_output)
{
    auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(static_cast<aclTensor*>(nullptr)));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(swish_test, case_013_input_invalid_dim)
{
    auto inputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(swish_test, case_014_output_invalid_dim)
{
    auto inputDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(swish_test, case_015_scalar_input)
{
    auto inputDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwish, INPUT(inputDesc, betaDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}
