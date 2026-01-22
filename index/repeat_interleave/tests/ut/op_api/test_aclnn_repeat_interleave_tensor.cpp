/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_repeat_interleave.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "acl/acl.h"

using namespace op;
using namespace std;

class l2_repeat_interleave_tensor_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "repeat_interleave_tensor_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "repeat_interleave_tensor_test TearDown" << std::endl;
    }

    void test_run(
        vector<int64_t> repeatsDims, aclDataType repeatsDtype, aclFormat repeatsFormat, vector<int64_t> repeatsRange,
        int64_t outputSize, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto repeats =
            TensorDesc(repeatsDims, repeatsDtype, repeatsFormat).ValueRange(repeatsRange[0], repeatsRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRepeatInterleaveTensor, INPUT(repeats, outputSize), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    }

    void test_run_invalid(
        vector<int64_t> repeatsDims, aclDataType repeatsDtype, aclFormat repeatsFormat, vector<int64_t> repeatsRange,
        int64_t outputSize, vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat)
    {
        auto repeats =
            TensorDesc(repeatsDims, repeatsDtype, repeatsFormat).ValueRange(repeatsRange[0], repeatsRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRepeatInterleaveTensor, INPUT(repeats, outputSize), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

// repeats + out: 不支持除int64和int32以外的
TEST_F(l2_repeat_interleave_tensor_test, l2_repeat_interleave_tensor_test_02)
{
    test_run_invalid({18}, ACL_UINT8, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_UINT8, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_INT8, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_INT8, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_INT16, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_INT16, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_BOOL, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_BOOL, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_FLOAT16, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_FLOAT, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_FLOAT, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_DOUBLE, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_DOUBLE, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_BF16, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_BF16, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_COMPLEX64, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_COMPLEX64, ACL_FORMAT_ND);
    test_run_invalid({18}, ACL_COMPLEX128, ACL_FORMAT_ND, {2, 2}, 36, {36}, ACL_COMPLEX128, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_tensor_test, l2_repeat_interleave_tensor_test_04)
{
    uint64_t workspaceSize = 0;
    auto repeats = TensorDesc({18}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 2);
    auto out = TensorDesc({36}, ACL_INT64, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    int64_t outputSize = 36;

    auto ut = OP_API_UT(aclnnRepeatInterleaveTensor, INPUT(nullptr, outputSize), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRepeatInterleaveTensor, INPUT(repeats, outputSize), OUTPUT(nullptr));
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_tensor_test, l2_repeat_interleave_tensor_test_05)
{
    // repeats empty
    test_run({0}, ACL_INT64, ACL_FORMAT_ND, {2, 2}, 0, {0}, ACL_INT64, ACL_FORMAT_ND);
}
