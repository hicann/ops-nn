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

#include "../../../op_host/op_api/aclnn_scatter_add.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_scatter_add_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Scatter Add Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Scatter Add Test TearDown" << endl; }

public:
    aclTensor *CreateAclTensor(vector<int64_t> view_shape, vector<int64_t> stride,
                               int64_t offset,vector<int64_t> storage_shape, aclDataType dataType=ACL_FLOAT) {
    return aclCreateTensor(view_shape.data(), view_shape.size(), dataType, stride.data(), offset,
                           ACL_FORMAT_ND, storage_shape.data(), storage_shape.size(), nullptr);
  }
};

// 空tensor
TEST_F(l2_scatter_add_test, case_2)
{
    auto self_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND)
        .Precision(0.0001, 0.0001);
    int64_t dim = 0;
    auto index_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto src_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterAdd, INPUT(self_desc, dim, index_desc, src_desc), OUTPUT(self_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// checkNotNull self index src
TEST_F(l2_scatter_add_test, case_3)
{
    auto self_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND)
        .Precision(0.0001, 0.0001);
    int64_t dim = 0;
    auto index_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto src_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterAdd, INPUT(nullptr, dim, index_desc, src_desc), OUTPUT(self_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnScatterAdd, INPUT(self_desc, dim, nullptr, src_desc), OUTPUT(self_desc));
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnScatterAdd, INPUT(self_desc, dim, index_desc, nullptr), OUTPUT(self_desc));
    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// checkNotNull out
TEST_F(l2_scatter_add_test, case_4)
{
    auto self_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND)
        .Precision(0.0001, 0.0001);
    int64_t dim = 0;
    auto index_desc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);
    auto src_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnScatterAdd, INPUT(nullptr, dim, index_desc, src_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtype different dtype of input
TEST_F(l2_scatter_add_test, case_7)
{
    auto self_desc = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1.111131123123, 1, 1, 1, 1, 1, 1, 1, 1})
        .Precision(0.0001, 0.0001);
    int64_t dim = 0;
    auto index_desc = TensorDesc({1, 3}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int>{0, 1, 2});
    auto src_desc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto ut = OP_API_UT(aclnnScatterAdd, INPUT(self_desc, dim, index_desc, src_desc), OUTPUT(self_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// CheckShape 1 index > self
TEST_F(l2_scatter_add_test, case_9)
{
    auto self_desc = TensorDesc({3, 3}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1.111131123123, 1, 1, 1, 1, 1, 1, 1, 1})
        .Precision(0.0001, 0.0001);
    int64_t dim = 0;
    auto index_desc = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int>{0, 1, 2, 2});
    auto src_desc = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto ut = OP_API_UT(aclnnScatterAdd, INPUT(self_desc, dim, index_desc, src_desc), OUTPUT(self_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape 1 index > src
TEST_F(l2_scatter_add_test, case_10)
{
    auto self_desc = TensorDesc({3, 4}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1.111131123123, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .Precision(0.0001, 0.0001);
    int64_t dim = 0;
    auto index_desc = TensorDesc({1, 4}, ACL_INT64, ACL_FORMAT_ND)
        .Value(vector<int>{0, 1, 2, 2});
    auto src_desc = TensorDesc({3, 2}, ACL_DOUBLE, ACL_FORMAT_ND)
        .Value(vector<double>{1, 2, 3, 4, 5, 6});
    auto ut = OP_API_UT(aclnnScatterAdd, INPUT(self_desc, dim, index_desc, src_desc), OUTPUT(self_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}