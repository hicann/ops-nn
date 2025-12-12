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
 * \file test_scatter.h
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

class l2_scatter_test : public testing::Test
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
        vector<int64_t> srcDims, aclDataType srcDtype, aclFormat srcFormat, vector<int64_t> srcRange,
        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat, int64_t dim)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto index = TensorDesc(indexDims, indexDtype, indexFormat).ValueRange(indexRange[0], indexRange[1]);
        auto src = TensorDesc(srcDims, srcDtype, srcFormat).ValueRange(srcRange[0], srcRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);
        int64_t reduction = 0;

        auto ut = OP_API_UT(aclnnScatter, INPUT(self, dim, index, src, reduction), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();

        reduction = 1;
        auto ut2 = OP_API_UT(aclnnScatter, INPUT(self, dim, index, src, reduction), OUTPUT(out));
        aclnnStatus getWorkspaceResult2 = ut2.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult2, ACL_SUCCESS);
        // ut2.TestPrecision();

        reduction = 2;
        auto ut3 = OP_API_UT(aclnnScatter, INPUT(self, dim, index, src, reduction), OUTPUT(out));
        aclnnStatus getWorkspaceResult3 = ut3.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult3, ACL_SUCCESS);
        // ut3.TestPrecision();

        reduction = 3;
        auto ut4 = OP_API_UT(aclnnScatter, INPUT(self, dim, index, src, reduction), OUTPUT(out));
        aclnnStatus getWorkspaceResult4 = ut4.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult4, ACL_SUCCESS);
        // ut4.TestPrecision();

        reduction = 4;
        auto ut5 = OP_API_UT(aclnnScatter, INPUT(self, dim, index, src, reduction), OUTPUT(out));
        aclnnStatus getWorkspaceResult5 = ut5.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult5, ACL_SUCCESS);
        // ut5.TestPrecision();
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> indexDims, aclDataType indexDtype, aclFormat indexFormat, vector<int64_t> indexRange,
        vector<int64_t> srcDims, aclDataType srcDtype, aclFormat srcFormat, vector<int64_t> srcRange,
        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat, int64_t dim)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto index = TensorDesc(indexDims, indexDtype, indexFormat).ValueRange(indexRange[0], indexRange[1]);
        auto src = TensorDesc(srcDims, srcDtype, srcFormat).ValueRange(srcRange[0], srcRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);
        int64_t reduction = 0;

        auto ut = OP_API_UT(aclnnScatter, INPUT(self, dim, index, src, reduction), OUTPUT(out));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }

public:
    aclTensor* CreateAclTensor(
        vector<int64_t> view_shape, vector<int64_t> stride, int64_t offset, vector<int64_t> storage_shape,
        aclDataType dataType = ACL_FLOAT)
    {
        return aclCreateTensor(
            view_shape.data(), view_shape.size(), dataType, stride.data(), offset, ACL_FORMAT_ND, storage_shape.data(),
            storage_shape.size(), nullptr);
    }
};

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_scatter_test, l2_scatter_test_18)
{
    // self + index + src empty
    test_run(
        {10, 15, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5, 0}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {2, 5, 0},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {10, 15, 0}, ACL_FLOAT16, ACL_FORMAT_ND, 0);
    // index + src empty
    test_run(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 0, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 0, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, 2);
    // index empty
    test_run(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 0, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, 2);

    // self empty
    test_run_invalid(
        {10, 15, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {10, 15, 0}, ACL_FLOAT16, ACL_FORMAT_ND, 0);
    // src empty
    test_run_invalid(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 2}, {3, 5, 0},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, 0);
}
