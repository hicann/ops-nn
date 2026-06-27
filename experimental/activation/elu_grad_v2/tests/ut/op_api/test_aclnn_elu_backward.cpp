/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "../../../op_host/op_api/aclnn_elu_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_elu_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EluBackward API Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EluBackward API Test TearDown" << std::endl;
    }
};

TEST_F(l2_elu_backward_test, case_001_float32_workspace_success)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gradInputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto alphaDesc = ScalarDesc(1.3f);
    auto scaleDesc = ScalarDesc(0.8f);
    auto inputScaleDesc = ScalarDesc(1.1f);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, false, selfDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(l2_elu_backward_test, case_002_float16_workspace_success)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gradInputDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto alphaDesc = ScalarDesc(1.0f);
    auto scaleDesc = ScalarDesc(1.0f);
    auto inputScaleDesc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, false, selfDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(l2_elu_backward_test, case_003_bfloat16_workspace_success_on_910b)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gradInputDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.02, 0.02);
    auto alphaDesc = ScalarDesc(1.0f);
    auto scaleDesc = ScalarDesc(0.9f);
    auto inputScaleDesc = ScalarDesc(1.2f);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, false, selfDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(ret, ACL_SUCCESS);
    } else {
        EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_elu_backward_test, case_004_nullptr_validation)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto scaleDesc = ScalarDesc(1.0f);
    auto inputScaleDesc = ScalarDesc(1.0f);

    uint64_t workspaceSize = 0;

    auto utNullInput = OP_API_UT(
        aclnnEluBackward, INPUT(nullptr, alphaDesc, scaleDesc, inputScaleDesc, false, selfDesc),
        OUTPUT(gradInputDesc));
    EXPECT_EQ(utNullInput.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    auto utNullAlpha = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, (aclScalar*)nullptr, scaleDesc, inputScaleDesc, false, selfDesc),
        OUTPUT(gradInputDesc));
    EXPECT_EQ(utNullAlpha.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

    auto utNullOutput = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, false, selfDesc),
        OUTPUT((aclTensor*)nullptr));
    EXPECT_EQ(utNullOutput.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_elu_backward_test, case_005_dtype_validation)
{
    auto alphaDesc = ScalarDesc(1.0f);
    auto scaleDesc = ScalarDesc(1.0f);
    auto inputScaleDesc = ScalarDesc(1.0f);

    auto invalidGrad = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto invalidSelf = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto invalidOut = TensorDesc({2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto utInvalid = OP_API_UT(
        aclnnEluBackward, INPUT(invalidGrad, alphaDesc, scaleDesc, inputScaleDesc, false, invalidSelf),
        OUTPUT(invalidOut));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(utInvalid.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);

    auto gradFloat = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfHalf = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outFloat = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto utMismatch = OP_API_UT(
        aclnnEluBackward, INPUT(gradFloat, alphaDesc, scaleDesc, inputScaleDesc, false, selfHalf),
        OUTPUT(outFloat));
    EXPECT_EQ(utMismatch.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_elu_backward_test, case_006_shape_and_dim_validation)
{
    auto alphaDesc = ScalarDesc(1.0f);
    auto scaleDesc = ScalarDesc(1.0f);
    auto inputScaleDesc = ScalarDesc(1.0f);

    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto wrongSelfDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto utShapeMismatch = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, false, wrongSelfDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(utShapeMismatch.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);

    auto largeDimDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto largeDimOut = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto utLargeDim = OP_API_UT(
        aclnnEluBackward, INPUT(largeDimDesc, alphaDesc, scaleDesc, inputScaleDesc, false, largeDimDesc),
        OUTPUT(largeDimOut));
    EXPECT_EQ(utLargeDim.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_elu_backward_test, case_007_negative_alpha_rejected_for_result_mode)
{
    auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto negativeAlpha = ScalarDesc(-0.1f);
    auto scaleDesc = ScalarDesc(1.0f);
    auto inputScaleDesc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, negativeAlpha, scaleDesc, inputScaleDesc, true, selfDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_elu_backward_test, case_008_empty_tensor_success)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = ScalarDesc(1.0f);
    auto scaleDesc = ScalarDesc(1.0f);
    auto inputScaleDesc = ScalarDesc(1.0f);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, false, selfDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
