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

#include "level2/aclnn_heaviside.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_heaviside_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "heaviside_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "heaviside_test TearDown" << endl;
    }
};

// 测试aicore:FLOAT32类型支持
TEST_F(l2_heaviside_test, case_dtype_aicore_support)
{
    auto self_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试非连续支持
TEST_F(l2_heaviside_test, case_NonContiguous)
{
    auto self_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试空tensor
TEST_F(l2_heaviside_test, case_Empty_Tensor)
{
    auto self_tensor_desc = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({0, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试不支持类型CheckDtypeValid
TEST_F(l2_heaviside_test, case_CheckDtypeValid)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(tensor_desc, tensor_desc), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试broadcast
TEST_F(l2_heaviside_test, case_broadcast)
{
    auto self_tensor_desc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3, 5});

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试不支持broadcast
TEST_F(l2_heaviside_test, case_broadcast_not_support)
{
    auto self_tensor_desc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3, 5}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试broadcast失败
TEST_F(l2_heaviside_test, case_broadcast_wrong)
{
    auto self_tensor_desc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 1, 5}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试oushape与broadcastshape不相等
TEST_F(l2_heaviside_test, case_broadcast_wrong_outshape)
{
    auto self_tensor_desc = TensorDesc({2, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto other_tensor_desc = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc({2, 3, 5}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHeaviside, INPUT(self_tensor_desc, other_tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试入参空指针
TEST_F(l2_heaviside_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnHeaviside, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}