/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <vector>
#include <array>
#include <complex>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_elu_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_elu_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "elu_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "elu_backward_test TearDown" << std::endl;
    }
};

// 正常场景_FLOAT_ND
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_FLOAT_ND)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_FLOAT16_NCHW
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_FLOAT16_NCHW)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto gradInputDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_BF16_NHWC
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_BF16_NHWC)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NHWC);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NHWC);
    auto gradInputDesc = TensorDesc({2, 2, 2, 2}, ACL_BF16, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);

        // precision simulate
        ut.TestPrecision();
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 正常场景_ratio_uint8_HWCN
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_ratio_uint8_HWCN)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
    uint8_t alpha = 1;
    auto alphaDesc = ScalarDesc(alpha);
    uint8_t scale = 1;
    auto scaleDesc = ScalarDesc(scale);
    uint8_t inputScale = 1;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto gradInputDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_gradInput_COMPLEX64_NDHWC
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_gradInput_COMPLEX64_NDHWC)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto gradInputDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_alpha_2_NCDHW
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_alpha_2_NCDHW)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    float alpha = 2.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto gradInputDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_alpha_1/2_NCHW_NHWC
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_alpha_half_NCHW_NHWC)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float alpha = 0.5;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradInputDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_scale_2_ND
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_scale_2_ND)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 2.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_scale_1/2_ND
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_scale_half_ND)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 0.5;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_inputScale_2_ND
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_inputScale_2_ND)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 2.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_inputScale_1/2_ND
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_inputScale_half_ND)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 0.5;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_isResult_false_ND
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_isResult_false_ND)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = false;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// CheckNotNull_gradOutput_nullptr
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_nullptr)
{
    auto gradOutputDesc = nullptr;
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_alpha_nullptr
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_alpha_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = nullptr;
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_scale_nullptr
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_scale_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto scaleDesc = nullptr;
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_inputScale_nullptr
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_inputScale_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    auto inputScaleDesc = nullptr;
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_selfOrResult_nullptr
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_selfOrResult_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = nullptr;
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_gradInput_nullptr
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradInput_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = nullptr;

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_dtype_unequal
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_dtype_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_INT32
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_INT32)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_INT64
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_INT64)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_INT16
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_INT16)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_INT8
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_INT8)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_UINT8
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_UINT8)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_BOOL
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_BOOL)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_COMPLEX64
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_COMPLEX64)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_COMPLEX128
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_COMPLEX128)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_DOUBLE
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_DOUBLE)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_UNDEFINED
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradOutput_UNDEFINED)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType_gradInput_INT32
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_gradInput_INT32)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_gradOutput_selfOrResult_unequal
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_shape_gradOutput_selfOrResult_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_gradOutput_gradInput_unequal
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_shape_gradOutput_gradInput_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_10D
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_shape_10D)
{
    auto gradOutputDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckAttributeValue_alpha_negtive
TEST_F(l2_elu_backward_test, l2_elu_backward_abnormal_alpha_negtive)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = -1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 2.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_valuerange)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 非连续
TEST_F(l2_elu_backward_test, l2_elu_backward_normal_uncontiguous)
{
    auto gradOutputDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    float scale = 1.0;
    auto scaleDesc = ScalarDesc(scale);
    float inputScale = 1.0;
    auto inputScaleDesc = ScalarDesc(inputScale);
    bool isResultDesc = true;
    auto selfOrResultDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto gradInputDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(
        aclnnEluBackward, INPUT(gradOutputDesc, alphaDesc, scaleDesc, inputScaleDesc, isResultDesc, selfOrResultDesc),
        OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}