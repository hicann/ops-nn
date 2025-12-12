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

#include "../../../op_host/op_api/aclnn_repeat_interleave.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "acl/acl.h"

using namespace op;
using namespace std;

class l2_repeat_interleave_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "repeat_interleave_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "repeat_interleave_test TearDown" << std::endl; }

    void test_run(vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> repeatsDims, aclDataType repeatsDtype, aclFormat repeatsFormat, vector<int64_t> repeatsRange,
        int64_t outputSize, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto repeats = TensorDesc(repeatsDims, repeatsDtype, repeatsFormat).ValueRange(repeatsRange[0], repeatsRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRepeatInterleave, INPUT(self, repeats, outputSize), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> repeatsDims, aclDataType repeatsDtype, aclFormat repeatsFormat, vector<int64_t> repeatsRange,
        int64_t outputSize, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto repeats = TensorDesc(repeatsDims, repeatsDtype, repeatsFormat).ValueRange(repeatsRange[0], repeatsRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRepeatInterleave, INPUT(self, repeats, outputSize), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

// self + out: 910A not support bfloat16
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_09)
{
    test_run_invalid({2, 3, 3}, ACL_BF16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_BF16, ACL_FORMAT_ND);
}

// self + out: 不支持float64, complex64, complex128
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_10)
{
    test_run_invalid({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_DOUBLE, ACL_FORMAT_ND);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_COMPLEX64, ACL_FORMAT_ND);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_COMPLEX128, ACL_FORMAT_ND);
}


///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_13)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto repeats = TensorDesc({18}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 2);
    auto out = TensorDesc({36}, ACL_INT32, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    int64_t outputSize = 36;

    auto ut = OP_API_UT(aclnnRepeatInterleave, INPUT(nullptr, repeats, outputSize), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRepeatInterleave, INPUT(self, nullptr, outputSize), OUTPUT(out));
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnRepeatInterleave, INPUT(self, repeats, outputSize), OUTPUT(nullptr));
    getWorkspaceResult = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

