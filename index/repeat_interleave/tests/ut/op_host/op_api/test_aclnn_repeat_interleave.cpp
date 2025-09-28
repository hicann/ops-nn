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

#include "level2/aclnn_repeat_interleave.h"

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

///////////////////////////////////////
/////          检查dtype          /////
///////////////////////////////////////

// self + out: uint8
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_01)
{
    test_run({2, 3, 3}, ACL_UINT8, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_UINT8, ACL_FORMAT_ND);
}

// self + out: int8
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_02)
{
    test_run({2, 3, 3}, ACL_INT8, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_INT8, ACL_FORMAT_ND);
}

// self + out: int16
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_03)
{
    test_run({2, 3, 3}, ACL_INT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_INT16, ACL_FORMAT_ND);
}

// self + out: int32
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_04)
{
    test_run({2, 3, 3}, ACL_INT32, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_INT32, ACL_FORMAT_ND);
}

// self + out: int64
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_05)
{
    test_run({2, 3, 3}, ACL_INT64, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_INT64, ACL_FORMAT_ND);
}

// self + out: bool
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_06)
{
    test_run({2, 3, 3}, ACL_BOOL, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_BOOL, ACL_FORMAT_ND);
}

// self + out: float16
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_07)
{
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
}

// self + out: float32
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_08)
{
    test_run({2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT, ACL_FORMAT_ND);
}

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

// repeats只能为INT64，INT32
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_11)
{
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT32, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
}

// self.dtype = out.dtype
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_12)
{
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT, ACL_FORMAT_ND);
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

///////////////////////////////////////
/////         检查format          /////
///////////////////////////////////////

// 所有format都支持
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_14)
{
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    test_run({2, 2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW, {-10, 10}, {8}, ACL_INT64, ACL_FORMAT_NCHW, {2, 2},
        16, {16}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    test_run({2, 2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC, {-10, 10}, {8}, ACL_INT64, ACL_FORMAT_NHWC, {2, 2},
        16, {16}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    test_run({2, 2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_HWCN, {-10, 10}, {8}, ACL_INT64, ACL_FORMAT_HWCN, {2, 2},
        16, {16}, ACL_FLOAT16, ACL_FORMAT_HWCN);
    test_run({2, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NDHWC, {-10, 10}, {8}, ACL_INT64, ACL_FORMAT_NDHWC, {2, 2},
        16, {16}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    test_run({2, 1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCDHW, {-10, 10}, {8}, ACL_INT64, ACL_FORMAT_NCDHW, {2, 2},
        16, {16}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
}

///////////////////////////////////////
/////         支持空tensor         /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_15)
{
    // self + repeats empty
    test_run({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        0, {0}, ACL_FLOAT16, ACL_FORMAT_ND);
    // self empty, repeats not emtpy (size 1)
//     test_run({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
//     0, {0}, ACL_FLOAT16, ACL_FORMAT_ND);
    test_run({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {1}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        0, {0}, ACL_FLOAT16, ACL_FORMAT_ND);

    // repeats must have the same size as input along dim
    // 1. self empty, repeats not emtpy
    test_run_invalid({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {6}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        0, {0}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 2. self not empty, repeats empty
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {0}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        0, {0}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 1. self empty, repeats' dim is larger than one
    test_run_invalid({2, 0, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2, 1, 3}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        0, {0}, ACL_FLOAT16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////       支持非连续tensor       /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_16)
{
    uint64_t workspaceSize = 0;
    auto self = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto selfT = TensorDesc({5, 4}, ACL_INT32, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-10, 10);
    auto repeats = TensorDesc({20}, ACL_INT64, ACL_FORMAT_ND).ValueRange(2, 2);
    auto out = TensorDesc({40}, ACL_INT32, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    int64_t outputSize = 40;

    // self not contiguous
    auto ut = OP_API_UT(aclnnRepeatInterleave, INPUT(selfT, repeats, outputSize), OUTPUT(out));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();
}

///////////////////////////////////////
/////          检查维度数          /////
///////////////////////////////////////

//// repeats的维度数为 0D / 1D tensor
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_17)
{
    // repeats为0D tensor
//    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
//        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    // repeats为1D tensor, size为1
    test_run({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {1}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    // repeats为1D tensor, size为元素数
    auto self = TensorDesc({2, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto repeats = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1, 2, 3, 4});
    auto out = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

    auto ut = OP_API_UT(aclnnRepeatInterleave, INPUT(self, repeats, 10), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    // ut.TestPrecision();

    // repeats为非0D / 1D tensor
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {1, 4}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    // repeats为1D tensor, 但size不为1或者元素数
    // 不为1
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {2}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 不为元素数： 元素数-1
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {17}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {36}, ACL_FLOAT16, ACL_FORMAT_ND);
}


///////////////////////////////////////
/////          检查shape          /////
///////////////////////////////////////

// 算子计算出来的shape != out.shape
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_18)
{
    // 计算出来的shape != out.shape
    test_run_invalid({2, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {18}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        36, {35}, ACL_FLOAT16, ACL_FORMAT_ND);
}

///////////////////////////////////////
/////          0维 ~ 8维          /////
///////////////////////////////////////

TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_19)
{
    // 0维
//    test_run({}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {1}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
//        2, {2}, ACL_FLOAT16, ACL_FORMAT_ND);
//    test_run({}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {}, ACL_INT64, ACL_FORMAT_ND, {1, 1},
//        1, {1}, ACL_FLOAT16, ACL_FORMAT_ND);
//    test_run({}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {}, ACL_INT64, ACL_FORMAT_ND, {0, 0},
//        0, {0}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 1维
    test_run({5}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {5}, ACL_INT64, ACL_FORMAT_ND, {1, 1},
        5, {5}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 8维
    test_run({1, 2, 1, 2, 1, 1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {8}, ACL_INT64, ACL_FORMAT_ND, {3, 3},
        24, {24}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 9维
    test_run_invalid({1, 2, 1, 2, 1, 1, 1, 2, 1}, ACL_FLOAT16, ACL_FORMAT_ND, {-10, 10}, {8}, ACL_INT64, ACL_FORMAT_ND, {2, 2},
        16, {16}, ACL_FLOAT16, ACL_FORMAT_ND);
}

// 可以加个outputSize的校验

//test repeat=1
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_20)
{
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t repeats = 1;
  int64_t outputSize = 2;
  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnRepeatInterleaveInt, INPUT(selfDesc, repeats, outputSize), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

//test repeat=1
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_21)
{
  auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t repeats = 1;
  int64_t dim = 0;
  int64_t outputSize = 2;
  auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(selfDesc, repeats, dim, outputSize), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 910A, not support BF16
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_test_22)
{
  auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  int64_t repeats = 1;
  int64_t dim = 0;
  int64_t outputSize = 2;
  auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(selfDesc, repeats, dim, outputSize), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 910B, support BF16
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_ascend910B_test_23)
{
  auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  int64_t repeats = 1;
  int64_t dim = 0;
  int64_t outputSize = 2;
  auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnRepeatInterleaveIntWithDim, INPUT(selfDesc, repeats, dim, outputSize), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 910B, repeat = 1
TEST_F(l2_repeat_interleave_test, l2_repeat_interleave_ascend910B_test_24)
{
  auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  auto repeatsDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 1);
  int64_t outputSize = 2;
  auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnRepeatInterleave, INPUT(selfDesc, repeatsDesc, outputSize), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}