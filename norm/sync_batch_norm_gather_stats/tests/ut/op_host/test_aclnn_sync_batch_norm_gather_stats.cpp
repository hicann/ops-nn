/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_sync_batch_norm_gather_stats.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_sync_batch_norm_gather_stats_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_sync_batch_norm_gather_stats_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_sync_batch_norm_gather_stats_test TearDown" << endl;
    }
};

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Dim_Check_0)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Dim_Check_1)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Dim_Check_2)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Dim_Check_3)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Type_0)
{
    auto totalSum = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Type_1)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Type_2)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Type_3)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_Type_4)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_gather_stats_test, batch_norm_stats_ret_OK)
{
    auto totalSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto totalSquareSum = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto sampleCount = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto mean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto variance = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    float momentum = 1e-1;
    float eps = 1e-5;
    auto batchMean = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto batchInvstd = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});

    uint64_t workspace_size = 0;
    auto ut = OP_API_UT(
        aclnnSyncBatchNormGatherStats, INPUT(totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps),
        OUTPUT(batchMean, batchInvstd));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}