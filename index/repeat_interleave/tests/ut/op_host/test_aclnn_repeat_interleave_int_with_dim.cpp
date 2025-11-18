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

#include "../../../op_host/op_api/aclnn_repeat_interleave.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "acl/acl.h"

using namespace op;
using namespace std;

class l2_repeat_interleave_dim_int_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "repeat_interleave_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "repeat_interleave_test TearDown" << std::endl; }

    void test_run(vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        int64_t repeats, int64_t dim, int64_t outputSize, vector<int64_t> outDims, aclDataType outDtype,
        aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(self, repeats, dim, outputSize), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
        // ut.TestPrecision();
    }

    void test_run_invalid(vector<int64_t> selfDims, aclDataType selfDtype, aclFormat selfFormat, vector<int64_t> selfRange,
        int64_t repeats, int64_t dim, int64_t outputSize, vector<int64_t> outDims, aclDataType outDtype,
        aclFormat outFormat)
    {
        auto self = TensorDesc(selfDims, selfDtype, selfFormat).ValueRange(selfRange[0], selfRange[1]);
        auto out = TensorDesc(outDims, outDtype, outFormat).Precision(0.00001, 0.00001);

        auto ut = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(self, repeats, dim, outputSize), OUTPUT(out));
        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
};

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

// self + out: uint8
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_01)
{
    test_run({2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_UINT8, ACL_FORMAT_ND);
}

// self + out: int8
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_02)
{
    test_run({2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_INT8, ACL_FORMAT_ND);
}

// self + out: int16
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_03)
{
    test_run({2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_INT16, ACL_FORMAT_ND);
}

// self + out: int32
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_04)
{
    test_run({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_INT32, ACL_FORMAT_ND);
}

// self + out: int64
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_05)
{
    test_run({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_INT64, ACL_FORMAT_ND);
}

// self + out: bool
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_06)
{
    test_run({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_BOOL, ACL_FORMAT_ND);
}

// self + out: float16
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_07)
{
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
}

// self + out: float32
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_08)
{
    test_run({2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND);
}

// self + out: 不支持float64, complex64, complex128
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_10)
{
    test_run_invalid({2, 3, 3}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    test_run_invalid({2, 3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
}

// self.dtype = out.dtype
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_11)
{
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////          检查空指针          /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_12)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t repeats = 2;
    int64_t dim = 1;
    auto out = TensorDesc({2, 6, 3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    int64_t outputSize = 36;

    auto ut = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(nullptr, repeats, dim, outputSize), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(self, repeats, dim, outputSize), OUTPUT(nullptr));
    getWorkspaceResult = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

///////////////////////////////////////
/////         检查format          /////
///////////////////////////////////////

// 所有format都支持
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_13)
{
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        6, {2, 6, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    test_run({2, 2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {-10, 10}, 2, 1,
        4, {2, 4, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    test_run({2, 2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC, {-10, 10}, 2, 1,
        4, {2, 4, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    test_run({2, 2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_HWCN, {-10, 10}, 2, 1,
        4, {2, 4, 1, 2}, ACL_FLOAT16, ACL_FORMAT_HWCN);
    test_run({2, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC, {-10, 10}, 2, 1,
        2, {2, 2, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    test_run({2, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW, {-10, 10}, 2, 1,
        2, {2, 2, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_14)
{
    // self empty
    test_run({2, 3, 0}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        0, {2, 6, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_15)
{
    uint64_t workspaceSize = 0;
    auto selfT = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    int64_t repeats = 2;
    int64_t dim = 1;
    auto out = TensorDesc({5, 8}, ACL_INT32, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    int64_t outputSize = 8;

    // self not contiguous
    auto ut = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(selfT, repeats, dim, outputSize), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

///////////////////////////////////////
/////    检查repeats的取值范围     /////
///////////////////////////////////////

// repeats的值为自然数
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_16)
{
    // repeats为0 
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0, 1,
        0, {2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // repeats为5
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 5, 1,
        15, {2, 15, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    // repeats为负数
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, -2, 1,
        6, {2, 6, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
}


///////////////////////////////////////
/////          检查shape          /////
///////////////////////////////////////

// 算子计算出来的shape != out.shape
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_17)
{
    // 计算出来的shape != out.shape
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        36, {2, 5, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////          0维 ~ 8维          /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_18)
{
    // 0维, 不支持传入dim
//    test_run_invalid({}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 3, 0,
//        3, {3}, ACL_FLOAT16, ACL_FORMAT_ND);
//    test_run_invalid({}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 3, -1,
//        3, {3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 1维
    test_run({5}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 5, 0,
        25, {25}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 8维
    test_run({1, 2, 1, 2, 1, 1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 3, 2,
        3, {1, 2, 3, 2, 1, 1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 9维
    test_run_invalid({1, 2, 1, 2, 1, 1, 1, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 1,
        4, {1, 4, 1, 2, 1, 1, 1, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////       检查dim的取值范围       /////
///////////////////////////////////////

// dim为 [-N, N-1]
TEST_F(l2_repeat_interleave_dim_int_test, l2_repeat_interleave_dim_int_test_19)
{
    // dim为 -N 
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, -3,
        4, {4, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // dim为 0 
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2, 0,
        4, {4, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // dim为 N - 1
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 5, 2,
        15, {2, 3, 15}, ACL_FLOAT16, ACL_FORMAT_ND);

    // dim为 -N - 1
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, -2, -4,
        6, {2, 6, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    // dim为 N 
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, -2, 3,
        6, {2, 6, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
}

// 可以加个outputSize的校验


