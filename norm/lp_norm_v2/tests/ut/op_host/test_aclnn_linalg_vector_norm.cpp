/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/op_api/aclnn_linalg_vector_norm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_linalg_vector_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_linalg_vector_norm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_linalg_vector_norm_test TearDown" << endl;
    }
};

TEST_F(l2_linalg_vector_norm_test, case_float_float_normal)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_linalg_vector_norm_test, case_float_float_transpose_normal)
{
    auto self_desc =
        TensorDesc({3, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND, {210, 42, 1, 7}, 0, {3, 5, 6, 7}).ValueRange(0, 2);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_linalg_vector_norm_test, case_float_float16_normal)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT16;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_float16_float16_normal)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT16;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_linalg_vector_norm_test, case_float16_float_normal)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_linalg_vector_norm_test, case_self_empty)
{
    auto self_desc = TensorDesc({2, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({1});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_linalg_vector_norm_test, case_self_nullptr)
{
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT((aclTensor*)nullptr, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_linalg_vector_norm_test, case_out_nullptr)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;

    auto ut =
        OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT((aclTensor*)nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_linalg_vector_norm_test, case_self_9_dim)
{
    auto self_desc = TensorDesc({2, 2, 3, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_out_9_dim)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_self_int32)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_out_int32)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_dtype_int32)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_INT32;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_out_type_not_equal_with_dtype)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_dims_out_of_range)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({3});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, case_dims_appears_multiple)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc(vector<int64_t>{1, 1});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_linalg_vector_norm_test, ascend910B_bf16_bf16_normal)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_BF16;
    auto out_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_linalg_vector_norm_test, ascend910B2_float_float_normal)
{
    auto self_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ord = ScalarDesc(2.0f);
    auto dims = IntArrayDesc({0});
    bool keepdim = false;
    aclDataType dtype = ACL_FLOAT;
    auto out_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLinalgVectorNorm, INPUT(self_desc, ord, dims, keepdim, dtype), OUTPUT(out_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}