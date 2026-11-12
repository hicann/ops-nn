/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_batch_norm_reduce.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

class l2BatchNormReduceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2l2BatchNormReduceTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2l2BatchNormReduceTest TearDown" << std::endl; }
};

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_bfloat16)
{
    auto x = TensorDesc({3, 5, 3, 8}, ACL_BF16, ACL_FORMAT_NCHW);
    auto sum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto squareSum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_noncontiguous_sum)
{
    auto x = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto sum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 0, {10});
    auto squareSum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_noncontiguous_square_sum)
{
    auto x = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto sum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto squareSum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 0, {10});

    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_noncontiguous_outputs_with_offset)
{
    auto x = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto sum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 1, {10});
    auto squareSum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND, {2}, 1, {10});

    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_rejects_nhwc)
{
    auto x = TensorDesc({3, 3, 8, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto sum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto squareSum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_rejects_empty_nhwc)
{
    auto x = TensorDesc({0, 3, 8, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto sum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto squareSum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_rejects_float64)
{
    auto x = TensorDesc({1, 2, 3, 4}, ACL_DOUBLE, ACL_FORMAT_NCHW);
    auto sum = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto squareSum = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_rejects_rank_nine)
{
    auto x = TensorDesc({1, 2, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto sum = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto squareSum = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormReduce, INPUT(x), OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_rejects_null_workspace_size)
{
    auto xDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* x = DescToAclContainer(xDesc);
    aclTensor* sum = DescToAclContainer(outDesc);
    aclTensor* squareSum = DescToAclContainer(outDesc);
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnBatchNormReduceGetWorkspaceSize(x, sum, squareSum, nullptr, &executor), ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
    Release(x);
    Release(sum);
    Release(squareSum);
}

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_rejects_null_executor)
{
    auto xDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* x = DescToAclContainer(xDesc);
    aclTensor* sum = DescToAclContainer(outDesc);
    aclTensor* squareSum = DescToAclContainer(outDesc);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(aclnnBatchNormReduceGetWorkspaceSize(x, sum, squareSum, &workspaceSize, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);
    Release(x);
    Release(sum);
    Release(squareSum);
}
