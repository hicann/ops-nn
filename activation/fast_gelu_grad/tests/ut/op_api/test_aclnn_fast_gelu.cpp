/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_fast_gelu_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_fast_gelu_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "fast_gelu_backward_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "fast_gelu_backward_test TearDown" << endl; }
};

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_float32)
{
    auto self = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_float16)
{
    auto self = TensorDesc({2, 4, 6, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({2, 4, 6, 8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({2, 4, 6, 8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_3d_shape)
{
    auto self = TensorDesc({32, 64, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({32, 64, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({32, 64, 128}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_5d_shape)
{
    auto self = TensorDesc({2, 4, 8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({2, 4, 8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({2, 4, 8, 16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_empty_tensor)
{
    auto self = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_scalar)
{
    auto self = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_large_shape)
{
    auto self = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({1024, 1024}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_nchw_format)
{
    auto self = TensorDesc({2, 64, 112, 112}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto input_tensor = TensorDesc({2, 64, 112, 112}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto output_tensor = TensorDesc({2, 64, 112, 112}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_nullptr_grad)
{
    auto input_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT((aclTensor*)nullptr, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_nullptr_self)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, (aclTensor*)nullptr), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_nullptr_output)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_dtype_diff)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_shape_diff)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_unsupported_dtype)
{
    auto self = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto input_tensor = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto output_tensor = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_not_contiguous)
{
    auto self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto input_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto output_tensor = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_fast_gelu_backward_test, test_fast_gelu_backward_value_range)
{
    auto self = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto input_tensor = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto output_tensor = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFastGeluBackward, INPUT(self, input_tensor), OUTPUT(output_tensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}