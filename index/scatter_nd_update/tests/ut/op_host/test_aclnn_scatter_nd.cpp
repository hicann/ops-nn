/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_scatter_nd.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_scatter_nd_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Scatter Nd Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Scatter Nd Test TearDown" << endl; }
};

TEST_F(l2_scatter_nd_test, case_1)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_scatter_nd_test, case_2)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int64_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_scatter_nd_test, case_3)
{
    auto self = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_scatter_nd_test, case_4)
{
    auto self = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int64_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_scatter_nd_test, case_5)
{
    auto self = TensorDesc({8}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int64_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{0, 0, 1, 1});
    auto out = TensorDesc({8}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_scatter_nd_test, case_6)
{
    auto self = TensorDesc({8}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int64_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{0, 0, 1, 1});
    auto out = TensorDesc({8}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 类型
TEST_F(l2_scatter_nd_test, case_7)
{
    auto self = TensorDesc({8}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// shape
TEST_F(l2_scatter_nd_test, case_8)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({2,2}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空指针
TEST_F(l2_scatter_nd_test, case_9)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 2, 3, 4});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, (aclTensor*)nullptr), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针
TEST_F(l2_scatter_nd_test, case_10)
{
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT((aclTensor*)nullptr, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针
TEST_F(l2_scatter_nd_test, case_11)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, (aclTensor*)nullptr, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空指针
TEST_F(l2_scatter_nd_test, case_12)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<double>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, (aclTensor*)nullptr, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 空tensor
TEST_F(l2_scatter_nd_test, case_null_tensor)
{
    auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indice_desc = TensorDesc({4, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto updates_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indice_desc, updates_desc, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 数据类型bf16
TEST_F(l2_scatter_nd_test, case_13)
{
    auto self = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B ||
        op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_93
    ) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// shape
TEST_F(l2_scatter_nd_test, case_14)
{
    auto self = TensorDesc({4,2}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int64_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{0, 0, 1, 1});
    auto out = TensorDesc({8}, ACL_BOOL, ACL_FORMAT_ND)
        .Value(vector<bool>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices数据类型
TEST_F(l2_scatter_nd_test, case_15)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_UINT32, ACL_FORMAT_ND)
        .Value(vector<uint32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输入NAN
TEST_F(l2_scatter_nd_test, case_16)
{
    auto self = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, NAN, 1, 1, 1, 1, 1});
    auto indices = TensorDesc({4,1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>{0, 1, 4, 6});
    auto updates = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{NAN, 2, 3, 4});
    auto out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输入大于8维
TEST_F(l2_scatter_nd_test, case_17)
{
    auto self = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>(2 * 2 * 2 * 2 * 2 * 2));
    auto indices = TensorDesc({4, 1, 1, 1, 1, 1, 1, 1}, ACL_INT32, ACL_FORMAT_ND)
        .Value(vector<int32_t>(4));
    auto updates = TensorDesc({4, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>(4 * 2 * 2 * 2 * 2 * 2));
    auto out = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>(2 * 2 * 2 * 2 * 2 * 2));
    auto ut = OP_API_UT(aclnnScatterNd, INPUT(self, indices, updates, out), OUTPUT());
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}