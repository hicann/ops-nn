/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_nll_loss_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_nll_loss_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "nll_loss_backward SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "nll_loss_backward TearDown" << endl;
    }
};

TEST_F(l2_nll_loss_backward_test, case_001)
{
    auto gradDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<int64_t>{0});

    auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_002)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({10}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 6);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<int64_t>{3});

    auto outDesc = TensorDesc({10, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_003)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 6);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 2;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<int64_t>{3});

    auto outDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_004)
{
    auto gradDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_005)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 3, 4});
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = 0;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_006)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({7, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({7}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 3, 4, 5, 6});
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 2;
    int64_t ignoreIndex = 5;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({7, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_007)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 3;
    int64_t ignoreIndex = 4;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss_backward_test, case_008)
{
    auto gradDesc = TensorDesc({12}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({12}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss_backward_test, case_009)
{
    auto gradDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto outDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss_backward_test, case_010)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 7}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1, 7}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss_backward_test, case_011)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, (aclTensor*)nullptr, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss_backward_test, case_012)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, (aclTensor*)nullptr, weightDesc, reduction, ignoreIndex, totalWeightDesc),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss_backward_test, case_013)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, (aclTensor*)nullptr, reduction, ignoreIndex, totalWeightDesc),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss_backward_test, case_014)
{
    auto selfDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT((aclTensor*)nullptr, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss_backward_test, case_015)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, (aclTensor*)nullptr),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss_backward_test, case_016)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss_backward_test, case_017)
{
    auto gradDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_018)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 2;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_019)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_020)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({10, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({10, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, case_021)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto selfDesc = TensorDesc({10, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({10, 4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss_backward_test, case_022)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 1, 2, 3, 4});
    auto weightDesc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = 0;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss_backward_test, Ascend910B2_case_023)
{
    auto gradDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({10}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 6);
    auto weightDesc = TensorDesc({7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND).Value(vector<int64_t>{3});

    auto outDesc = TensorDesc({10, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nll_loss_backward_test, ascend310P_case_024)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 1, 2, 3, 4});
    auto weightDesc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = 0;
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnNLLLossBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}