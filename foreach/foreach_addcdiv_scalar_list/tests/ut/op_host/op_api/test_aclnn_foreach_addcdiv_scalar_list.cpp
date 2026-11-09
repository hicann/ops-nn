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
#include <float.h>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_foreach_addcdiv_scalar_list.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>
#include "opdev/platform.h"

using namespace std;

class l2_foreach_addcdiv_scalar_list_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_addcdiv_scalar_list_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_addcdiv_scalar_list_test TearDown" << endl; }
};

// support non-contiguous input and output
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    vector<int64_t> outDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(1, 2);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// scalars element count must equal the number of tensors in x1
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_scalars_numel_mismatch)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// scalars dtype must match the dtype of the x1 tensor list
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_scalars_dtype_mismatch)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// tensor lists x1/x2 must contain the same number of tensors
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_tensorlist_size_mismatch)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// tensor lists x1/x3 must contain the same number of tensors
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_tensorlist_size_mismatch_x3)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// tensor lists x1/out must contain the same number of tensors
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_tensorlist_size_mismatch_out)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// non-contiguous scalars view is normalized through Contiguous
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_non_contiguous_scalars)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 0, {4}).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// a null x1 tensor list pointer is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_nullptr_x1)
{
    vector<int64_t> selfDims = {2, 4};
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(nullptr, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// a null x2 tensor list pointer is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_nullptr_x2)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, nullptr, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// a null x3 tensor list pointer is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_nullptr_x3)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, nullptr, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// a null scalars pointer is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_nullptr_scalars)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, nullptr), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// a null out tensor list pointer is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_nullptr_out)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// an x1[0] dtype outside the supported set {FLOAT, FLOAT16, BF16} is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_x1_dtype_unsupported)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// an x2 member dtype differing from the x1[0] dtype is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_x2_member_dtype_mismatch)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// an x3 member dtype differing from the x1[0] dtype is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_x3_member_dtype_mismatch)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// an out member dtype differing from the x1[0] dtype is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_out_member_dtype_mismatch)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// a scalars dtype outside the supported set {FLOAT, FLOAT16, BF16} is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_scalars_dtype_unsupported)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({1}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// a scalars element count above the number of tensors in x1 is rejected
TEST_F(l2_foreach_addcdiv_scalar_list_test, ascend910B2_foreach_addcdiv_scalar_list_test_scalars_numel_mismatch_more)
{
    vector<int64_t> selfDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scalars = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarList, INPUT(x1List, x2List, x3List, scalars), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
