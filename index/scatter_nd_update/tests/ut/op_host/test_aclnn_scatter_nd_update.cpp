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

#include "../../../op_host/op_api/aclnn_scatter_nd_update.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_scatter_nd_update_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "ScatterNdUpdate Test Setup" << endl; }
    static void TearDownTestCase() { cout << "ScatterNdUpdate Test TearDown" << endl; }
};

TEST_F(l2_scatter_nd_update_test, case_1)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    
}

// 空tensor
TEST_F(l2_scatter_nd_update_test, case_2)
{
    auto varRef_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indice_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// checkNotNull self index src
TEST_F(l2_scatter_nd_update_test, case_3)
{
    auto varRef_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indice_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(nullptr, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, nullptr, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, nullptr), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid for self and src
TEST_F(l2_scatter_nd_update_test, case_5)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_BOOL,
        ACL_DT_UNDEFINED};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto varRef_desc = TensorDesc({3, 3, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 1);
        auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
        auto updates_desc = TensorDesc({2}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 1);
        auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

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

// CheckDtypeValid for index
TEST_F(l2_scatter_nd_update_test, case_6)
{
    vector<aclDataType> ValidList = {
        ACL_INT32,
        ACL_INT64,
        ACL_DT_UNDEFINED};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        auto indice_desc = TensorDesc({2, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 3);
        auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

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
TEST_F(l2_scatter_nd_update_test, case_7)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 910B
TEST_F(l2_scatter_nd_update_test, ascend910B2_case_8)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    
}

// 空tensor
TEST_F(l2_scatter_nd_update_test, ascend910B2_case_9)
{
    auto varRef_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indice_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// checkNotNull self index src
TEST_F(l2_scatter_nd_update_test, ascend910B2_case_10)
{
    auto varRef_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indice_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(nullptr, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, nullptr, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, nullptr), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid for self and src
TEST_F(l2_scatter_nd_update_test, ascend910B2_case_12)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_BOOL,
        ACL_BF16,
        ACL_INT64,
        ACL_DT_UNDEFINED};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto varRef_desc = TensorDesc({3, 3, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 1);
        auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
        auto updates_desc = TensorDesc({2}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 1);
        auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

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

// CheckDtypeValid for index
TEST_F(l2_scatter_nd_update_test, ascend910B2_case_13)
{
    vector<aclDataType> ValidList = {
        ACL_INT32,
        ACL_INT64,
        ACL_DT_UNDEFINED};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        auto indice_desc = TensorDesc({2, 3}, ValidList[i], ACL_FORMAT_ND).ValueRange(0, 3);
        auto updates_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
        auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());

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
TEST_F(l2_scatter_nd_update_test, ascend910B2_case_14)
{
    auto varRef_desc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indice_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updates_desc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnScatterNdUpdate, INPUT(varRef_desc, indice_desc, updates_desc), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
