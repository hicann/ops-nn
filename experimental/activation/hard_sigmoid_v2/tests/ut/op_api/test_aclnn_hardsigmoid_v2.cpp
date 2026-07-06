/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_hardsigmoid_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class l2_hardsigmoid_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "hardsigmoid_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "hardsigmoid_test TearDown" << std::endl; }
};

TEST_F(l2_hardsigmoid_v2_test, case_001_float)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_002_float16)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_003_bfloat16)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_004_int32)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hardsigmoid_v2_test, case_005_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_006_not_contiguous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_007_invalid_input_dtype)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_v2_test, case_008_invalid_output_dtype)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_v2_test, case_009_dtype_mismatch)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_v2_test, case_010_shape_mismatch)
{
    auto selfDesc = TensorDesc({2, 3, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_v2_test, case_011_input_invalid_dim)
{
    auto selfDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_v2_test, case_012_scalar_input)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_013_nullptr)
{
    auto outDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHardsigmoidV2, INPUT(static_cast<aclTensor*>(nullptr)), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_hardsigmoid_v2_test, case_014_inplace_float)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_015_inplace_float16)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_016_inplace_bfloat16)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_017_inplace_empty_tensor)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_018_inplace_not_contiguous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

TEST_F(l2_hardsigmoid_v2_test, case_019_inplace_invalid_dim)
{
    auto selfDesc = TensorDesc({2, 4, 5, 2, 3, 4, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_v2_test, case_020_inplace_invalid_dtype)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_hardsigmoid_v2_test, case_021_inplace_nullptr)
{
    auto ut = OP_API_UT(aclnnInplaceHardsigmoidV2, INPUT(static_cast<aclTensor*>(nullptr)), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
