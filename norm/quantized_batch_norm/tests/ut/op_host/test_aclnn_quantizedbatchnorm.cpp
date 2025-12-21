/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quantizedbatchnorm.cpp
 * \brief
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_quantized_batch_norm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"

class l2QuantizedBatchNormTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2QuantizedBatchNormTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2QuantizedBatchNormTest TearDown" << std::endl;
    }
};

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_int32_infer)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_int8_infer)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_INT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_uint8_infer)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_null_input)
{
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            nullptr, nullptr, nullptr, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_null_out)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_float)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_int32_var)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_int32_mean)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_int32_weight)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_int32_bias)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_UINT8, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_shape)
{
    auto selfDesc = TensorDesc({3, 4, 5, 6, 7, 8, 9, 10, 11}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 4, 5, 6, 7, 8, 9, 10, 11}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_dim)
{
    auto selfDesc = TensorDesc({3, 5, 3, 8}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 3, 8}, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_format1)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_INT32, ACL_FORMAT_NCDHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_format2)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2QuantizedBatchNormTest, l2_quantized_batch_norm_err_format3)
{
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_INT32, ACL_FORMAT_NCHW).ValueRange(1, 5);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1});
    auto biasDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto inScaleScalarDesc = ScalarDesc(3.0f);
    auto inZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto outScaleScalarDesc = ScalarDesc(3.0f);
    auto outZeroPointScalarDesc = ScalarDesc(static_cast<int32_t>(2));
    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0});
    auto varDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1});
    float eps = 1e-5;

    auto outDesc = TensorDesc({3, 5, 4, 6, 5}, ACL_INT32, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnQuantizedBatchNorm,
        INPUT(
            selfDesc, meanDesc, varDesc, inScaleScalarDesc, inZeroPointScalarDesc, outScaleScalarDesc,
            outZeroPointScalarDesc, weightDesc, biasDesc, eps),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
