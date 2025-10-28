/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_batch_norm_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class l2BatchNormBackwardTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2BatchNormBackwardTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2BatchNormBackwardTest TearDown" << std::endl;
    }
};

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_false_bias)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, false});

    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_false_weight)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, false, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_false_input)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{false, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValidCount(0);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_train_5d)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // // ut.TestPrecision();  // comment bcz of timeout in model tests (282616 ms)
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_infer_5d)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, false, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_float32_4d)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_float16_4d)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_float32_3d)
{
    auto gradOutDesc = TensorDesc({3, 5, 20}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 20}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 20}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_float16_3d)
{
    auto gradOutDesc = TensorDesc({3, 5, 20}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 20}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 20}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_float32_2d)
{
    auto gradOutDesc = TensorDesc({90, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);
    auto selfDesc = TensorDesc({90, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({90, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_float16_2d)
{
    auto gradOutDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto selfDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_err_null)
{
    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_err_int32)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_INT32, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_INT32, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_INT32, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_weight_int32)
{
    auto gradOutDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto selfDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_mean_int32)
{
    auto gradOutDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto selfDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_var_int32)
{
    auto gradOutDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);
    auto selfDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({90, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_err_shape)
{
    auto gradOutDesc = TensorDesc({3, 4, 5, 6, 7, 8, 9, 10, 11}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 4, 5, 6, 7, 8, 9, 10, 11}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 4, 5, 6, 7, 8, 9, 10, 11}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_mask1_error)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(nullptr, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_mask1_int32)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_INT32, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_mask2_error)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, nullptr, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_mask2_int32)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_mask3_error)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_mask3_int32)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, nullptr, nullptr, nullptr, nullptr, nullptr, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_float32_7d)
{
    auto gradOutDesc = TensorDesc({3, 5, 4, 6, 7, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 4, 6, 7, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 4, 6, 7, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_err_dim)
{
    auto gradOutDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sErrDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, sErrDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);

    ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, sErrDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));
    getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);

    ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, sErrDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));
    getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, ascend910B2_l2_batch_norm_backward_train_5d)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // // ut.TestPrecision();  // comment bcz of timeout in model tests (282616 ms)
}

TEST_F(l2BatchNormBackwardTest, ascend910_9589_l2_batch_norm_backward_infer)
{
    auto gradOutDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sErrDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, false, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, sErrDesc, sMeanDesc, sVarDesc, false, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));
    getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormBackwardTest, l2_batch_norm_backward_empty_mask)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{});

    auto gradInDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, true, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormBackwardTest, ascend910_9589_l2_batch_norm_backward_shape_error)
{
    auto gradOutDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 10);

    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto sVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto sErrDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});
    auto output_mask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outputErrDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sErrDesc, sVarDesc, false, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sErrDesc, false, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, gradBiasDesc));
    getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, false, 1e-5, output_mask),
        OUTPUT(gradInDesc, outputErrDesc, gradBiasDesc));
    getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    ut = OP_API_UT(
        aclnnBatchNormBackward,
        INPUT(gradOutDesc, selfDesc, weightDesc, rMeanDesc, rVarDesc, sMeanDesc, sVarDesc, false, 1e-5, output_mask),
        OUTPUT(gradInDesc, gradWeightDesc, outputErrDesc));
    getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}