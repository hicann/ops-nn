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

#include "../../../op_host/op_api/aclnn_nll_loss2d.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_nll_loss2d_forward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_nll_loss2d_forward SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_nll_loss2d_forward TearDown" << endl;
    }
};

TEST_F(l2_nll_loss2d_forward_test, case_001)
{
    auto selfDesc = TensorDesc({1, 3, 150, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({1, 150, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1, 150, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_002)
{
    auto selfDesc = TensorDesc({1, 3, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc =
        TensorDesc({1, 5, 2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 0, 0, 2, 1, 0, 1, 0, 1, 1});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 5, 2.1});
    int64_t reduction = 0;
    int64_t ignoreIndex = 0;

    auto outDesc = TensorDesc({1, 5, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_003)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 3, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}
//===============================================================================

TEST_F(l2_nll_loss2d_forward_test, case_004)
{
    auto selfDesc = TensorDesc({3, 4, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 2, 5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({3, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_005)
{
    auto selfDesc = TensorDesc({3, 3, 2, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto targetDesc = TensorDesc({3, 2, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<int64_t>{3}).ValidCount(0);
    auto outDesc = TensorDesc({3, 2, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_006)
{
    auto selfDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(0, 1);
    auto weightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_007)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_HWCN).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_008)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_NDHWC).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_009)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// ==========================================================================================

TEST_F(l2_nll_loss2d_forward_test, case_010)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_INT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_INT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_INT16, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_INT16, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_forward_test, case_011)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_INT8, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_INT8, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_forward_test, case_012)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_INT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_INT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_INT16, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_INT16, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_forward_test, case_015)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = 2;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_016)
{
    auto selfDesc = TensorDesc({1, 3, 2, 1}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 2, 1}, ACL_INT64, ACL_FORMAT_NCDHW).Value(vector<int64_t>{0, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 3);
    int64_t reduction = 1;
    int64_t ignoreIndex = 2;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// ��tensor����
TEST_F(l2_nll_loss2d_forward_test, case_017)
{
    auto selfDesc = TensorDesc({3, 2, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 0, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0}).ValidCount(0);

    auto outDesc = TensorDesc({3, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_018)
{
    auto targetDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(nullptr, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_forward_test, case_019)
{
    auto selfDesc = TensorDesc({1, 7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0}).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, nullptr, weightDesc, reduction, ignoreIndex), OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_forward_test, case_020)
{
    auto selfDesc = TensorDesc({1, 7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0}).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, nullptr, reduction, ignoreIndex), OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_forward_test, case_021)
{
    auto selfDesc = TensorDesc({1, 7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex), OUTPUT(outDesc, nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_forward_test, case_022)
{
    auto selfDesc = TensorDesc({1, 7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(nullptr, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_forward_test, case_023)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 3, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_forward_test, case_024)
{
    auto selfDesc = TensorDesc({1, 3, 1, 1}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 1, 1}, ACL_INT64, ACL_FORMAT_HWCN).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_HWCN).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_forward_test, case_025)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 3, 4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({1, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_forward_test, case_026)
{
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_027)
{
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 1);
    int64_t reduction = 2;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_forward_test, case_029)
{
    auto selfDesc = TensorDesc({5, 3, 1, 10}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({5, 1, 10}, ACL_INT32, ACL_FORMAT_NCDHW).ValueRange(0, 3);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = 3;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2d, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
        OUTPUT(outDesc, totalWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
