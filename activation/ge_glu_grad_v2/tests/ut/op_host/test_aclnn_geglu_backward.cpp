/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <complex>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_geglu_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_geglu_backward_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "geglu_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "geglu_backward_test TearDown" << std::endl;
    }
};

// 正常场景_FLOAT16_ND_SHAPE1
TEST_F(l2_geglu_backward_test, l2_geglu_backward_normal_FLOAT16_ND_SHAPE1)
{
    auto gradOutputDesc = TensorDesc({2, 920, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 920, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto geluDesc = TensorDesc({2, 920, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2, 920, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    // uint64_t workspaceSize = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    // ut.TestPrecision();
}

// 正常场景_FLOAT16_ND_SHAPE1 gegluv
TEST_F(l2_geglu_backward_test, ascend910B2_normal_FLOAT16_ND_SHAPE1)
{
    auto gradOutputDesc = TensorDesc({2, 920, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 920, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto geluDesc = TensorDesc({2, 920, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2, 920, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3Backward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc, activateLeft),
            OUTPUT(gradInputDesc));

        // only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);

        // precision simulate
        ut.TestPrecision();
    }
}

// 正常场景_FLOAT16_ND_SHAPE2
TEST_F(l2_geglu_backward_test, l2_geglu_backward_normal_FLOAT16_ND_SHAPE2)
{
    auto gradOutputDesc = TensorDesc({2, 230, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 230, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto geluDesc = TensorDesc({2, 230, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2, 230, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    // uint64_t workspaceSize = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    // ut.TestPrecision();
}

// 正常场景_FLOAT16_ND_SHAPE3
TEST_F(l2_geglu_backward_test, l2_geglu_backward_normal_FLOAT16_ND_SHAPE3)
{
    auto gradOutputDesc = TensorDesc({2, 50, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 50, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto geluDesc = TensorDesc({2, 50, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2, 50, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    // uint64_t workspaceSize = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    // ut.TestPrecision();
}

// 正常场景_FLOAT16_ND_SHAPE4
TEST_F(l2_geglu_backward_test, l2_geglu_backward_normal_FLOAT16_ND_SHAPE4)
{
    auto gradOutputDesc = TensorDesc({2, 10, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 10, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto geluDesc = TensorDesc({2, 10, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2, 10, 1024}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    // uint64_t workspaceSize = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    // ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_geglu_backward_test, l2_geglu_backward_normal_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// CheckNotNull_gradOutput_nullptr
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gradOutput_nullptr)
{
    auto gradOutputDesc = nullptr;
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_self_nullptr
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_self_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = nullptr;
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_gelu_nullptr
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gelu_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = nullptr;
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_gradInput_nullptr
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gradInput_nullptr)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = nullptr;

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_INT32
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_INT32)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_self_dtype_unequal
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gradOutput_self_dtype_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_gelu_dtype_unequal
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gradOutput_gelu_dtype_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_gradOutput_gradInput_dtype_unequal
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gradOutput_gradInput_dtype_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_gradOutput_gelu_shape_unequal
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gradOutput_gelu_shape_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_self_gradInput_shape_unequal
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_self_gradInput_shape_unequal)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_dim_error
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_dim_error)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -2;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_gradOutput_self_shape_error
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_gradOutput_self_shape_error)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckAttributeValue
TEST_F(l2_geglu_backward_test, l2_geglu_backward_abnormal_CheckAttributeValue)
{
    auto gradOutputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto geluDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dimDesc = -1;
    int64_t approximateDesc = 2;
    auto gradInputDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续
TEST_F(l2_geglu_backward_test, l2_geglu_backward_normal_uncontiguous)
{
    auto gradOutputDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto selfDesc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {8, 2});
    auto geluDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    int64_t dimDesc = -1;
    int64_t approximateDesc = 1;
    auto gradInputDesc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {8, 2});

    auto ut = OP_API_UT(
        aclnnGeGluBackward, INPUT(gradOutputDesc, selfDesc, geluDesc, dimDesc, approximateDesc), OUTPUT(gradInputDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}