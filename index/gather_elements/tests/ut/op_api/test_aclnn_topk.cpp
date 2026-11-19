/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_topk.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "level0/sort_with_index.h"

using namespace std;

class l2_topk_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "topk_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "topk_test TearDown" << endl; }
};

// self为空指针
TEST_F(l2_topk_test, l2_topk_test_nullptr_self0)
{
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(nullptr, 1, 0, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// values为空指针
TEST_F(l2_topk_test, l2_topk_test_nullptr_values0)
{
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, 0, 1, 1), OUTPUT(nullptr, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// indices为空指针
TEST_F(l2_topk_test, l2_topk_test_nullptr_indices0)
{
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, 0, 1, 1), OUTPUT(valDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(getWorkspaceResult, ACLNN_SUCCESS);
}

// self为空tensor
TEST_F(l2_topk_test, l2_topk_test_nullptr_self)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 0, 0, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// values为空tensor
TEST_F(l2_topk_test, l2_topk_test_nullptr_values)
{
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 0, 0, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// indices为空tensor
TEST_F(l2_topk_test, l2_topk_test_nullptr_indices)
{
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 0, 0, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，0维tensor，dim为0
TEST_F(l2_topk_test, l2_topk_test_fp32_shape0_dim0)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, 0, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，0维tensor，dim为-1
TEST_F(l2_topk_test, l2_topk_test_fp32_shape0_dim_neg1)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, -1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 非正常路径，0维tensor，dim为-2
TEST_F(l2_topk_test, l2_topk_test_fp32_shape0_dim_neg2)
{
    auto selfDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, -2, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32
TEST_F(l2_topk_test, l2_topk_test_fp32)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,largest为False，sorted为True
TEST_F(l2_topk_test, l2_topk_test_fp32_lf_st)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 1, 0, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,largest为False，sorted为False
TEST_F(l2_topk_test, l2_topk_test_fp32_lf_sf)
{
    auto selfDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 0, 0, 0), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,largest为True，sorted为False
TEST_F(l2_topk_test, l2_topk_test_fp32_lt_sf)
{
    auto selfDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 0, 1, 0), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32，dim为-1
TEST_F(l2_topk_test, l2_topk_test_fp32_dim_neg1)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, -1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,dim为-2
TEST_F(l2_topk_test, l2_topk_test_fp32_dim_neg2)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, -2, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,dim为-3
TEST_F(l2_topk_test, l2_topk_test_fp32_dim_neg3)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1, 4, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, -3, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，float32,dim为-4
TEST_F(l2_topk_test, l2_topk_test_fp32_dim_neg4)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, -4, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32,k为负数
TEST_F(l2_topk_test, l2_topk_test_fp32_k_neg1)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, -1, -1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32,k大于self在dim上的size
TEST_F(l2_topk_test, l2_topk_test_fp32_k_10)
{
    auto selfDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 10, 2, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float16
TEST_F(l2_topk_test, l2_topk_test_fp16)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 0, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int8
TEST_F(l2_topk_test, l2_topk_test_int8)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_INT8, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 2, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，uint8
TEST_F(l2_topk_test, l2_topk_test_uint8)
{
    auto selfDesc = TensorDesc({2, 4, 5, 6}, ACL_UINT8, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 5, 4}, ACL_UINT8, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 5, 4}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 4, 3, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int16
TEST_F(l2_topk_test, l2_topk_test_int16)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_INT16, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 4, 3}, ACL_INT16, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 4, 3}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 3, 2, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int32
TEST_F(l2_topk_test, l2_topk_test_int32)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，int64
TEST_F(l2_topk_test, l2_topk_test_int64)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 用例不支持，bool
TEST_F(l2_topk_test, l2_topk_test_bool)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_BOOL, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 5}, ACL_BOOL, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，complex64
TEST_F(l2_topk_test, l2_topk_test_complex16)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，complex128
TEST_F(l2_topk_test, l2_topk_test_complex128)
{
    auto selfDesc = TensorDesc({2, 4, 5}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({2, 2, 5}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({2, 2, 5}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2, 1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 用例不支持，shape=9
TEST_F(l2_topk_test, l2_topk_test_shape9)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 1, 1, 1, 1), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径，float32，dim为-1
TEST_F(l2_topk_test, DAV_3510_9589_l2_topk_k_more_than_2k)
{
    auto selfDesc = TensorDesc({1, 5000}, ACL_FLOAT, ACL_FORMAT_ND);
    auto valDesc = TensorDesc({1, 2100}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indDesc = TensorDesc({1, 2100}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, 2100, -1, true, true), OUTPUT(valDesc, indDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

namespace {
thread_local int sortWithIndexNullOutputs = 0;
thread_local int sortWithIndexCalls = 0;
thread_local bool sortWithIndexSupported = true;
} // namespace

namespace l0op {
bool IsSortWithIndexA2A3Supported() { return sortWithIndexSupported; }

// LLT 只控制跨仓接口的返回张量，覆盖融合可用和独立升级时输出为空的情况。
const std::tuple<aclTensor*, aclTensor*> SortWithIndex(const aclTensor* self, const aclTensor* index, int64_t axis,
                                                       bool /*descending*/, bool stable, aclOpExecutor* executor)
{
    ++sortWithIndexCalls;
    EXPECT_EQ(axis, -1);
    EXPECT_TRUE(stable);
    EXPECT_EQ(index->GetDataType(), op::DataType::DT_INT32);
    auto values = (sortWithIndexNullOutputs & 1) != 0 ?
                      nullptr :
                      executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    auto indices = (sortWithIndexNullOutputs & 2) != 0 ?
                       nullptr :
                       executor->AllocTensor(index->GetViewShape(), index->GetDataType(), index->GetViewFormat());
    return std::make_tuple(values, indices);
}
} // namespace l0op

class l2_topk_radix_sort_with_index_test
    : public l2_topk_test,
      public testing::WithParamInterface<tuple<op::SocVersion, aclDataType, bool, bool, int>> {
protected:
    void SetUp() override
    {
        sortWithIndexNullOutputs = 0;
        sortWithIndexCalls = 0;
        sortWithIndexSupported = true;
    }

    void TearDown() override
    {
        sortWithIndexNullOutputs = 0;
        sortWithIndexCalls = 0;
        sortWithIndexSupported = true;
    }
};

// 覆盖 A2/A3、FP16/BF16、末轴/非末轴以及最大/最小值的融合排序接入。
TEST_P(l2_topk_radix_sort_with_index_test, get_workspace_size)
{
    const auto [soc, dtype, nonLastDim, largest, nullOutputs] = GetParam();
    op::SocVersionManager versionManager(soc);
    sortWithIndexNullOutputs = nullOutputs;
    sortWithIndexSupported = nullOutputs >= 0;
    constexpr int64_t sortLength = 12000000;
    constexpr int64_t k = 4096;
    constexpr int64_t rows = 2;
    const vector<int64_t> inputShape = nonLastDim ? vector<int64_t>{sortLength, rows} :
                                                    vector<int64_t>{rows, sortLength};
    const vector<int64_t> outputShape = nonLastDim ? vector<int64_t>{k, rows} : vector<int64_t>{rows, k};
    auto selfDesc = TensorDesc(inputShape, dtype, ACL_FORMAT_ND);
    auto valDesc = TensorDesc(outputShape, dtype, ACL_FORMAT_ND);
    auto indDesc = TensorDesc(outputShape, ACL_INT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnTopk, INPUT(selfDesc, k, nonLastDim ? 0 : -1, largest, true), OUTPUT(valDesc, indDesc));

    // 仅构建 Host 执行计划，不分配大输入数据，也不启动 NPU kernel。
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(sortWithIndexCalls, sortWithIndexSupported ? 1 : 0);
}

INSTANTIATE_TEST_SUITE_P(radix_sort_with_index, l2_topk_radix_sort_with_index_test,
                         testing::Combine(testing::Values(op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93),
                                          testing::Values(ACL_FLOAT16, ACL_BF16), testing::Bool(), testing::Bool(),
                                          testing::Values(-1, 0, 1, 2, 3)));
