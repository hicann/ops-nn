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

#include "../../../op_api/aclnn_nll_loss2d_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_nll_loss2d_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "nll_loss2d_backward SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "nll_loss2d_backward TearDown" << endl;
    }
};

TEST_F(l2_nll_loss2d_backward_test, case_001)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({30, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({30, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{150});

    auto outDesc = TensorDesc({30, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nll_loss2d_backward_test, case_002)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_backward_test, case_003)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({30, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({30, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 2;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{150});

    auto outDesc = TensorDesc({30, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nll_loss2d_backward_test, case_004)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_backward_test, case_007)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 3;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_backward_test, case_008)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_backward_test, case_009)
{
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(nullptr, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_backward_test, case_010)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_UINT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_UINT32, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_UINT32, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_UINT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续场景，由于GPU不支持输入非连续的grad无法生成golden，因此该用例不测试精度
TEST_F(l2_nll_loss2d_backward_test, case_011)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {4500, 30, 1, 5}, 0, {3, 150, 6, 5});
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5}).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor场景
TEST_F(l2_nll_loss2d_backward_test, case_012)
{
    auto gradDesc = TensorDesc({3, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 0, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_backward_test, case_013)
{
    auto gradDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_backward_test, case_014)
{
    auto gradDesc = TensorDesc({16, 16, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({16, 64, 16, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto targetDesc = TensorDesc({16, 16, 8}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 64);
    auto weightDesc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{0});

    auto outDesc = TensorDesc({16, 64, 16, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_backward_test, case_015)
{
    auto gradDesc = TensorDesc({8, 16, 16}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto selfDesc = TensorDesc({8, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto targetDesc = TensorDesc({8, 16, 16}, ACL_INT64, ACL_FORMAT_NHWC).ValueRange(0, 128);
    auto weightDesc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NHWC).Value(vector<float>{0});

    auto outDesc = TensorDesc({8, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_backward_test, case_016)
{
    auto gradDesc = TensorDesc({16, 8, 16}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto selfDesc = TensorDesc({16, 256, 8, 16}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto targetDesc = TensorDesc({16, 8, 16}, ACL_INT64, ACL_FORMAT_HWCN).ValueRange(0, 256);
    auto weightDesc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_HWCN).Value(vector<float>{0});

    auto outDesc = TensorDesc({16, 256, 8, 16}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nll_loss2d_backward_test, case_017)
{
    auto gradDesc = TensorDesc({16, 16, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto selfDesc = TensorDesc({16, 512, 16, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto targetDesc = TensorDesc({16, 16, 8}, ACL_INT64, ACL_FORMAT_NDHWC).ValueRange(0, 512);
    auto weightDesc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NDHWC).Value(vector<float>{0});

    auto outDesc = TensorDesc({16, 512, 16, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nll_loss2d_backward_test, case_018)
{
    auto gradDesc = TensorDesc({16, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({16, 1024, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({16, 8, 8}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 1024);
    auto weightDesc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Value(vector<float>{0});

    auto outDesc = TensorDesc({16, 1024, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nll_loss2d_backward_test, case_019)
{
    auto gradDesc = TensorDesc({16, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 32, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({16, 16, 16}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 32);
    auto weightDesc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({16, 32, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_backward_test, case_020)
{
    auto gradDesc = TensorDesc({8, 8, 16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto selfDesc = TensorDesc({8, 1024, 8, 16}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto targetDesc = TensorDesc({8, 8, 16}, ACL_INT64, ACL_FORMAT_NCDHW).ValueRange(0, 1024);
    auto weightDesc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_NCDHW).Value(vector<float>{0});

    auto outDesc = TensorDesc({8, 1024, 8, 16}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_nll_loss2d_backward_test, case_021)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, nullptr, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_backward_test, case_022)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward, INPUT(gradDesc, selfDesc, nullptr, weightDesc, reduction, ignoreIndex, totalWeightDesc),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_backward_test, case_023)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward, INPUT(gradDesc, selfDesc, targetDesc, nullptr, reduction, ignoreIndex, totalWeightDesc),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_backward_test, case_024)
{
    auto gradDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = -100;

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward, INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, nullptr),
        OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_nll_loss2d_backward_test, case_025)
{
    auto gradDesc = TensorDesc({3, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 1, 1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 2, 3});
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    int64_t ignoreIndex = 3;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<int64_t>{0});

    auto outDesc = TensorDesc({3, 5, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_nll_loss2d_backward_test, case_027)
{
    auto gradDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 3;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0});

    auto outDesc = TensorDesc({3, 150, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_nll_loss2d_backward_test, ascend910B2_case_028)
{
    auto gradDesc = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({30, 150, 5, 6}, ACL_BF16, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({30, 5, 6}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 150);
    auto weightDesc = TensorDesc({150}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 1);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{150});

    auto outDesc = TensorDesc({30, 150, 5, 6}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnNLLLoss2dBackward,
        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}