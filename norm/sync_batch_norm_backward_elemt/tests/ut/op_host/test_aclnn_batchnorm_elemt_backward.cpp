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
#include "../../../op_host/op_api/aclnn_batch_norm_elemt_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "op_api/op_api_def.h"

class l2BatchNormElemtBackwardTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2l2BatchNormElemtBackwardTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2l2BatchNormElemtBackwardTest TearDown" << std::endl;
    }
};

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_float32)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormElemtBackwardTest, ascend910B2_batch_norm_elemt_backward_bf16)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    } else {
        // EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_2d)
{
    auto gradOutDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_gradOut)
{
    auto gradOutDesc = nullptr;
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_input)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = nullptr;

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_meanDesc)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = nullptr;
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_invstd)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = nullptr;
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_weight)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = nullptr;
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_sumDy)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = nullptr;
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_sumDyXmn)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = nullptr;
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_null_gradInput)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = nullptr;

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_gradOut)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_input)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_mean)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_invstd)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_weight)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_sumDy)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_sumDyXmn)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_type_gradInput)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_format_gradOut)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_format_input)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_format_gradInput)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_input_empty)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 0, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_empty)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto gradOutDesc = TensorDesc({2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 1, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_shape_gradOut)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4, 2, 3, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_shape_input)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4, 2, 3, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_shape_input2)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_shape_gradInput)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4, 2, 3, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtBackwardTest, l2_batch_norm_elemt_backward_err_broadcast)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{8, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
