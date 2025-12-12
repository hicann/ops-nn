/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_inplace_scatter.h
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_scatter.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_scatter_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "scatter_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "scatter_test TearDown" << endl;
    }

    void test_run(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> indexDims, aclDataType indexDtype, aclFormat indexFormat, vector<int64_t> indexRange,
        vector<int64_t> srcDims, aclDataType srcDtype, aclFormat srcFormat, vector<int64_t> srcRange, int64_t dim)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat)
                        .ValueRange(selfRange[0], selfRange[1])
                        .Precision(0.00001, 0.00001);
        auto index = TensorDesc(indexDims, indexDtype, indexFormat).ValueRange(indexRange[0], indexRange[1]);
        auto src = TensorDesc(srcDims, srcDtype, srcFormat).ValueRange(srcRange[0], srcRange[1]);
        int64_t reduction = 0;

        auto ut = OP_API_UT(aclnnInplaceScatter, INPUT(self, dim, index, src, reduction), OUTPUT());
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();

        reduction = 1;
        auto ut2 = OP_API_UT(aclnnInplaceScatter, INPUT(self, dim, index, src, reduction), OUTPUT());
        aclnnStatus getWorkspaceResult2 = ut2.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult2, ACL_SUCCESS);
        // ut2.TestPrecision();

        reduction = 2;
        auto ut3 = OP_API_UT(aclnnInplaceScatter, INPUT(self, dim, index, src, reduction), OUTPUT());
        aclnnStatus getWorkspaceResult3 = ut3.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult3, ACL_SUCCESS);
        // ut3.TestPrecision();
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> indexDims, aclDataType indexDtype, aclFormat indexFormat, vector<int64_t> indexRange,
        vector<int64_t> srcDims, aclDataType srcDtype, aclFormat srcFormat, vector<int64_t> srcRange, int64_t dim)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat)
                        .ValueRange(selfRange[0], selfRange[1])
                        .Precision(0.00001, 0.00001);
        auto index = TensorDesc(indexDims, indexDtype, indexFormat).ValueRange(indexRange[0], indexRange[1]);
        auto src = TensorDesc(srcDims, srcDtype, srcFormat).ValueRange(srcRange[0], srcRange[1]);
        int64_t reduction = 0;

        auto ut = OP_API_UT(aclnnInplaceScatter, INPUT(self, dim, index, src, reduction), OUTPUT());

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

// 不支持未定义的dtype
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_15)
{
    test_run_invalid(
        {10, 15}, ACL_DT_UNDEFINED, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5},
        ACL_DT_UNDEFINED, ACL_FORMAT_ND, {-10, 10}, 1);
}

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_16)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({10, 15}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.00001, 0.00001);
    auto index = TensorDesc({2, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto src = TensorDesc({2, 5}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t reduction = 0;

    auto ut = OP_API_UT(aclnnInplaceScatter, INPUT(nullptr, 0, index, src, reduction), OUTPUT());
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnInplaceScatter, INPUT(self, 0, nullptr, src, reduction), OUTPUT());
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnInplaceScatter, INPUT(self, 0, index, nullptr, reduction), OUTPUT());
    getWorkspaceResult = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_18)
{
    // self + index + src empty
    test_run(
        {10, 15, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5, 0}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {2, 5, 0},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);
    // index + src empty
    test_run(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 0, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 0, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2);
    // index empty
    test_run(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 0, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2);

    // self empty
    test_run_invalid(
        {10, 15, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);
    // src empty
    test_run_invalid(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 5, 0},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);
}
