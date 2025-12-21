/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_tf_scatter_add.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_tf_scatter_add_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ScatterAdd Test Setup" << endl; }
    static void TearDownTestCase() { cout << "ScatterAdd Test TearDown" << endl; }
};

TEST_F(l2_tf_scatter_add_test, ascend910B2_case_1)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    
}

// checkNotNull var index updates
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_2)
{
    auto varRef_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indice_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(nullptr, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, nullptr, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, nullptr), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
// CheckDtypeValid for index
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_4)
{
    vector<aclDataType> ValidList = {
        ACL_INT32,
        ACL_INT64,
        ACL_DT_UNDEFINED};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        auto indice_desc = TensorDesc({2, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 3);
        auto updates_desc = TensorDesc({2, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        if (ValidList[i] != ACL_DT_UNDEFINED) {
            EXPECT_EQ(aclRet, ACL_SUCCESS);
            
        } else {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        }
    }
}

// CheckDtype different dtype of input
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_5)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test bf16
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_6)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    
}

// test AiCPU int8
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_7)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 3, 3, 3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    
}

// check updatesDimNum
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_8)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check updatesShape for indicesShape
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_9)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({3, 3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check updatesShape for varRefShape
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_10)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 3, 3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check updatesShape for indicesShape
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_11)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check updatesShape for varRefShape
TEST_F(l2_tf_scatter_add_test, ascend910B2_case_12)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_tf_scatter_add_test, ascend910B2_case_13)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    
}

// CheckDtypeValid for var and updates
TEST_F(l2_tf_scatter_add_test, ascend910_95_case_1)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_BF16,
        ACL_INT32,
        ACL_INT8,
        ACL_UINT8,
        ACL_DT_UNDEFINED};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto varRef_desc = TensorDesc({3, 3, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 1);
        auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
        auto updates_desc = TensorDesc({2, 3, 3, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 1);
        auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    }
}