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
#include "../../../op_host/op_api/aclnn_batch_norm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class l2BatchNormTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2BatchNormTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2BatchNormTest TearDown" << std::endl;
    }
};

TEST_F(l2BatchNormTest, l2_batch_norm_5d_train)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_5d_infer)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, false, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // // ut.TestPrecision();  // comment bcz of timeout in model tests (986327 ms)
}

TEST_F(l2BatchNormTest, l2_batch_norm_float32_infer)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, false, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float16_infer)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, false, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float16_4d)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, ascend910_9589_l2_batch_norm_float16_NHWC)
{
    auto selfDesc = TensorDesc({3, 8, 3, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 8, 3, 5}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float32_4d)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float16_3d)
{
    auto selfDesc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 7}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float32_3d)
{
    auto selfDesc = TensorDesc({3, 5, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float16_2d)
{
    auto selfDesc = TensorDesc({18, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({18, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float32_2d)
{
    auto selfDesc = TensorDesc({18, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({18, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_null_input)
{
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(nullptr, nullptr, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_null_out)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(nullptr, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_null_mean)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, nullptr, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_null_var)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_int32)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_int32_var)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, meanDesc, varDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_int32_self)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, meanDesc, varDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_int32_out)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, meanDesc, varDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_int32_mean)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, meanDesc, varDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_int32_weight)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_int32_bias)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto biasDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, biasDesc, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_shape)
{
    auto selfDesc = TensorDesc({3, 4, 5, 6, 7, 8, 9, 10, 11}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({3, 4, 5, 6, 7, 8, 9, 10, 11}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_float32_strided)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW, {120, 8, 24, 1});
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_float32_7d)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6, 7, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 4, 6, 7, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_dim)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_format1)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_format2)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_err_format3)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, ascend910B2_l2_batch_norm_error_mean_shape)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_BF16, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_BF16, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormTest, l2_batch_norm_empty_tensor)
{
    auto selfDesc = TensorDesc({3, 5, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 4, 0}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, ascend910B2_l2_batch_norm_bfloat16_infer)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_BF16, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_BF16, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, false, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, ascend910B2_l2_batch_norm_bfloat16_infer1)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8, 9}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(0);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, false, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, ascend910B2_l2_batch_norm_bnv3)
{
    auto selfDesc = TensorDesc({1, 64, 1, 1, 8192}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto weightDesc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto biasDesc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rMeanDesc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rVarDesc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 64, 1, 1, 8192}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto meanDesc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormTest, ascend910_9589_l2_batch_norm_mixDTypeInput_4d)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto rVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, weightDesc, biasDesc, rMeanDesc, rVarDesc, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormTest, ascend910_9589_l2_batch_norm_nullptr)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNorm, INPUT(selfDesc, nullptr, nullptr, nullptr, nullptr, true, 0.1, 1e-5),
        OUTPUT(outDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}