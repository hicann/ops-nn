/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_inplace_scatter.h
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_scatter.h"

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

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

// self + src + out: uint8
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_01)
{
    test_run(
        {10, 15}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_UINT8,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: int8
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_02)
{
    test_run(
        {10, 15}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_INT8,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: int16
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_03)
{
    test_run(
        {10, 15}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_INT16,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: int32
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_04)
{
    test_run(
        {10, 15}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_INT32,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: int64
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_05)
{
    test_run(
        {10, 15}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_INT64,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: bool
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_06)
{
    test_run(
        {10, 15}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_BOOL,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: float16
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_07)
{
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: float32
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_08)
{
    test_run(
        {10, 15}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: double
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_09)
{
    test_run(
        {10, 15}, ACL_DOUBLE, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_DOUBLE,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: complex64
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_10)
{
    test_run(
        {10, 15}, ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5},
        ACL_COMPLEX64, ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: complex128
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_11)
{
    test_run(
        {10, 15}, ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5},
        ACL_COMPLEX128, ACL_FORMAT_ND, {-10, 10}, 1);
}

// self + src + out: 不支持bfloat16   aicpu新增支持
TEST_F(l2_inplace_scatter_test, Ascend910B2_l2_inplace_scatter_test_12)
{
    test_run(
        {10, 15}, ACL_BF16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_BF16,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// indices: 支持int64 + int32, 不支持其他的
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_13)
{
    // 支持int32 + int64
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT32, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);

    // 不支持其他的
    test_run_invalid(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT16, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);
    test_run_invalid(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT8, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);
    test_run_invalid(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_UINT8, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

// self.dtype = src.dtype = out.dtype
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_14)
{
    // self.dtype = src.dtype
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);

    // self.dtype != src.dtype
    test_run_invalid(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5}, ACL_FLOAT,
        ACL_FORMAT_ND, {-10, 10}, 1);
}

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
/////         检查format          /////
///////////////////////////////////////

// 所有format都支持
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_17)
{
    test_run(
        {10, 15, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 5, 2, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4},
        {2, 5, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 1);
    test_run(
        {10, 15, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW, {-10, 10}, {2, 5, 2, 3}, ACL_INT64, ACL_FORMAT_NCHW, {0, 4},
        {2, 5, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW, {-10, 10}, 1);
    test_run(
        {10, 15, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NHWC, {-10, 10}, {2, 5, 2, 3}, ACL_INT64, ACL_FORMAT_NHWC, {0, 4},
        {2, 5, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NHWC, {-10, 10}, 1);
    test_run(
        {10, 15, 2, 3}, ACL_FLOAT16, ACL_FORMAT_HWCN, {-10, 10}, {2, 5, 2, 3}, ACL_INT64, ACL_FORMAT_HWCN, {0, 4},
        {2, 5, 2, 3}, ACL_FLOAT16, ACL_FORMAT_HWCN, {-10, 10}, 1);
    test_run(
        {10, 15, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC, {-10, 10}, {2, 5, 2, 3, 2}, ACL_INT64, ACL_FORMAT_NDHWC,
        {0, 4}, {2, 5, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC, {-10, 10}, 1);
    test_run(
        {10, 15, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW, {-10, 10}, {2, 5, 2, 3, 2}, ACL_INT64, ACL_FORMAT_NCDHW,
        {0, 4}, {2, 5, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW, {-10, 10}, 1);

    // NCL
    test_run(
        {30, 32, 17}, ACL_FLOAT, ACL_FORMAT_NCL, {-10, 10}, {30, 32, 17}, ACL_INT64, ACL_FORMAT_NCL, {0, 4},
        {30, 32, 17}, ACL_FLOAT, ACL_FORMAT_NCL, {-10, 10}, 1);
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

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_19)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Precision(0.00001, 0.00001);
    auto selfT =
        TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10).Precision(0.00001, 0.00001);
    auto index = TensorDesc({5, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto indexT = TensorDesc({5, 4}, ACL_INT64, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(0, 3);
    auto src = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto srcT = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    int64_t reduction = 0;

    // self not contiguous
    auto ut = OP_API_UT(aclnnInplaceScatter, INPUT(selfT, 0, index, src, reduction), OUTPUT());
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();

    // index not contiguous
    reduction = 1;
    auto ut2 = OP_API_UT(aclnnInplaceScatter, INPUT(self, 0, indexT, src, reduction), OUTPUT());
    aclnnStatus getWorkspaceResult2 = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult2, ACL_SUCCESS);
    // ut2.TestPrecision();

    // src not contiguous
    reduction = 2;
    auto ut3 = OP_API_UT(aclnnInplaceScatter, INPUT(self, 0, index, srcT, reduction), OUTPUT());
    aclnnStatus getWorkspaceResult3 = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult3, ACL_SUCCESS);
    // ut3.TestPrecision();

    // all not contiguous
    auto ut4 = OP_API_UT(aclnnInplaceScatter, INPUT(selfT, 0, indexT, srcT, reduction), OUTPUT());
    aclnnStatus getWorkspaceResult4 = ut4.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult4, ACL_SUCCESS);
    // ut4.TestPrecision();
}

///////////////////////////////////////
/////          检查维度数          /////
///////////////////////////////////////

// 维度数: src = index = self
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_20)
{
    // src = index = self
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {3, 5}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 1);

    // self.dimsize != index.dimsize
    test_run_invalid(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {1, 4}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5, 5},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 1);
    // self.dimsize != src.dimsize
    test_run_invalid(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {1, 4, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 5, 3, 4},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 1);
}

///////////////////////////////////////
/////          检查shape          /////
///////////////////////////////////////

// 除了dim以外的维度， index的维度大小 <= self的维度大小
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_22)
{
    // dim维度, index > self
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {30, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {40, 5},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);
    // dim维度, index <= self
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {10, 15}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {10, 15},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);

    // 非dim维度, index <= self < src
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {5, 10}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {5, 25},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);
    // 非dim维度，index > self, 报错
    test_run_invalid(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {100, 16}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {100, 25},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);
}

// index.shape < src.shape时正常运行
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_23)
{
    // index.shape < self.shape < src.shape
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {15, 20},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);

    // index.shape < src.shape < self.shape
    test_run(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {4, 7}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 0);

    // index.shape > src.shape
    test_run_invalid(
        {10, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {2, 3}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 0);
}

///////////////////////////////////////
/////          检查dim            /////
///////////////////////////////////////

// dim得在[-N, N-1]之间
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_24)
{
    // -N + N-1 + 0
    test_run(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, -3);
    test_run(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2);
    test_run(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);

    // -N-1 + N
    test_run_invalid(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, -4);
    test_run_invalid(
        {10, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {3, 5, 3},
        ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 3);
}

///////////////////////////////////////
/////         检查维度大小         /////
///////////////////////////////////////

TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_25)
{
    // 1维
    test_run(
        {10}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3}, ACL_INT64, ACL_FORMAT_ND, {0, 4}, {3}, ACL_FLOAT16,
        ACL_FORMAT_ND, {-10, 10}, 0);
    // 4维
    test_run(
        {10, 15, 15, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3, 2}, ACL_INT64, ACL_FORMAT_ND, {0, 4},
        {3, 5, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 2);
    // 8维
    test_run(
        {10, 15, 4, 4, 4, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3, 3, 3, 3, 3, 3}, ACL_INT64,
        ACL_FORMAT_ND, {0, 4}, {3, 5, 3, 3, 3, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);

    // 9维
    test_run_invalid(
        {10, 15, 4, 4, 4, 4, 4, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {3, 5, 3, 3, 3, 3, 3, 3, 1}, ACL_INT64,
        ACL_FORMAT_ND, {0, 4}, {3, 5, 3, 3, 3, 3, 3, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, 0);
}

///////////////////////////////////////
/////     支持torch.tensor(a)     /////
///////////////////////////////////////

// torch.tensor(a)能正常计算
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_29)
{
    auto self = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(3, 5).Precision(0.00001, 0.00001);
    auto index = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 0);
    auto src = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(6, 10);
    int64_t reduction = 0;
    int64_t dim = 0;

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

///////////////////////////////////////
/////     ut覆盖率 const->删去     /////
///////////////////////////////////////

// torch.tensor(a)能正常计算
TEST_F(l2_inplace_scatter_test, l2_inplace_scatter_test_30)
{
    auto self = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(3, 5).Precision(0.00001, 0.00001);
    auto index = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 0);
    auto src = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(6, 10);
    int64_t reduction = 0;
    int64_t dim = 0;

    auto ut = OP_API_UT(aclnnInplaceScatter, INPUT(self, dim, index, src, reduction), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    ut.TestPrecision();
}