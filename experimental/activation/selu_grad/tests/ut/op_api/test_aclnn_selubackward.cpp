/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <vector>

#include "../../../op_api/aclnn_selu_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class SeluBackwardOpApiTest : public testing::Test {};

TEST_F(SeluBackwardOpApiTest, float32_success)
{
    auto gradOutput = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto result = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gradInput = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(1.0e-4, 1.0e-4);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SeluBackwardOpApiTest, float16_success)
{
    auto gradOutput = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto result = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gradInput = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(1.0e-3, 1.0e-3);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SeluBackwardOpApiTest, bfloat16_success)
{
    auto gradOutput = TensorDesc({2, 17}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto result = TensorDesc({2, 17}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gradInput = TensorDesc({2, 17}, ACL_BF16, ACL_FORMAT_ND).Precision(1.0e-2, 1.0e-2);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SeluBackwardOpApiTest, integer_types_success)
{
    for (const aclDataType dataType : {ACL_INT32, ACL_INT8}) {
        auto gradOutput = TensorDesc({2, 16}, dataType, ACL_FORMAT_ND).ValueRange(-4, 4);
        auto result = TensorDesc({2, 16}, dataType, ACL_FORMAT_ND).ValueRange(-4, 4);
        auto gradInput = TensorDesc({2, 16}, dataType, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    }
}

TEST_F(SeluBackwardOpApiTest, empty_tensor_success)
{
    auto gradOutput = TensorDesc({0, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto result = TensorDesc({0, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInput = TensorDesc({0, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 1;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_F(SeluBackwardOpApiTest, non_contiguous_success)
{
    auto gradOutput = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto result = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto gradInput = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(1.0e-4, 1.0e-4);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SeluBackwardOpApiTest, higher_than_eight_dims_success)
{
    const std::vector<int64_t> shape = {2, 2, 2, 2, 2, 2, 2, 2, 3};
    auto gradOutput = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto result = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gradInput = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SeluBackwardOpApiTest, unsupported_dtype_failed)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto result = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto gradInput = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SeluBackwardOpApiTest, uint8_success)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 4);
    auto result = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 4);
    auto gradInput = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(SeluBackwardOpApiTest, shape_mismatch_failed)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto result = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SeluBackwardOpApiTest, input_dtype_mismatch_failed)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto result = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SeluBackwardOpApiTest, output_dtype_mismatch_failed)
{
    auto gradOutput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto result = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(gradOutput, result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(SeluBackwardOpApiTest, nullptr_failed)
{
    auto result = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInput = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSeluBackward, INPUT(static_cast<aclTensor*>(nullptr), result), OUTPUT(gradInput));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}
