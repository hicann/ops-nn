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

#include "../../../op_host/op_api/aclnn_mv.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "acl/acl.h"

using namespace op;
using namespace std;

class l2_mv_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "mv_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "mv_test TearDown" << std::endl;
    }

    void test_run(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> vecDims, aclDataType vecDtype, aclFormat vecFormat, vector<int64_t> vecRange,
        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat, int8_t cubeMathType)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto vec = TensorDesc(vecDims, vecDtype, vecFormat).ValueRange(vecRange[0], vecRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnMv, INPUT(self, vec), OUTPUT(out), cubeMathType);
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(
        vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        vector<int64_t> vecDims, aclDataType vecDtype, aclFormat vecFormat, vector<int64_t> vecRange,
        vector<int64_t> outDims, aclDataType outDtype, aclFormat outFormat, int8_t cubeMathType)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto vec = TensorDesc(vecDims, vecDtype, vecFormat).ValueRange(vecRange[0], vecRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnMv, INPUT(self, vec), OUTPUT(out), cubeMathType);
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

// self + other + out: 不支持double、complex64、complex128 + bool、uint8、int8、int16、int32、int64、bfloat16、
TEST_F(l2_mv_test, l2_mv_test_03)
{
    test_run_invalid(
        {2, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_DOUBLE, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_DOUBLE,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-15, -10}, {2},
        ACL_COMPLEX64, ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-15, -10}, {2},
        ACL_COMPLEX128, ACL_FORMAT_ND, 1);

    test_run_invalid(
        {2, 3}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_BOOL, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_BOOL,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_UINT8, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_UINT8,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_INT8, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_INT8,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_INT16, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_INT16,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_INT32, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_COMPLEX64,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_INT64, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_COMPLEX128,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {2, 3}, ACL_BF16, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_BF16, ACL_FORMAT_ND, {-15, -10}, {2}, ACL_BF16,
        ACL_FORMAT_ND, 1);
}

// ///////////////////////////////////////
// /////          检查空指针          /////
// ///////////////////////////////////////

TEST_F(l2_mv_test, l2_mv_test_05)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto vec = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(3, 10);
    auto out = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    int8_t cubeMathType = 1;

    auto ut = OP_API_UT(aclnnMv, INPUT(nullptr, vec), OUTPUT(out), cubeMathType);
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnMv, INPUT(self, nullptr), OUTPUT(out), cubeMathType);
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnMv, INPUT(self, vec), OUTPUT(nullptr), cubeMathType);
    getWorkspaceResult = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         支持空tensor        /////
///////////////////////////////////////

// 支持空tensor
TEST_F(l2_mv_test, l2_mv_test_06)
{
    // self n x 0, vec 0, out n   n不为0
    test_run(
        {3, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_FLOAT16, ACL_FORMAT_ND, {-15, -10}, {3}, ACL_FLOAT16,
        ACL_FORMAT_ND, 1);
    // self 0 x m, vec m, out 0   m不为0
    test_run(
        {0, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_FLOAT16, ACL_FORMAT_ND, {-15, -10}, {0}, ACL_FLOAT16,
        ACL_FORMAT_ND, 1);
    // self 0 x 0, vec 0, out 0   m, n都为0
    test_run(
        {0, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_FLOAT16, ACL_FORMAT_ND, {-15, -10}, {0}, ACL_FLOAT16,
        ACL_FORMAT_ND, 1);

    // self为空tensor时，如果vec + out dtype一致，则支持运算
    // 1. self可以和vec/out dtype不一致
    test_run(
        {3, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_FLOAT, ACL_FORMAT_ND, {-15, -10}, {3}, ACL_FLOAT,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {3, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_FLOAT16, ACL_FORMAT_ND, {-15, -10}, {3}, ACL_FLOAT,
        ACL_FORMAT_ND, 1);
    // 2. 该场景下，out dtype为不支持的数据类型也行
    test_run(
        {3, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_BOOL, ACL_FORMAT_ND, {-15, -10}, {3}, ACL_BOOL,
        ACL_FORMAT_ND, 1);
    test_run_invalid(
        {3, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_FLOAT16, ACL_FORMAT_ND, {-15, -10}, {3}, ACL_BOOL,
        ACL_FORMAT_ND, 1);
}
