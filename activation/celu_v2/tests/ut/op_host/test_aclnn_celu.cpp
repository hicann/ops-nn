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
 * \file test_celu.cpp
 * \brief
 */
#include <vector>
#include <array>
#include <complex>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_celu.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_celu_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "celu_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "celu_test TearDown" << std::endl;
    }
};

// 正常场景_FLOAT_ND
TEST_F(l2_celu_test, l2_celu_normal_FLOAT_ND)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_FLOAT_NHWC_INPLACE
TEST_F(l2_celu_test, l2_celu_normal_FLOAT_NHWC_INPLACE)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);

    auto ut = OP_API_UT(aclnnInplaceCelu, INPUT(selfDesc, alphaDesc), OUTPUT());

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_FLOAT16_NCHW
TEST_F(l2_celu_test, l2_celu_normal_FLOAT16_NCHW)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_out_COMPLEX64_NCDHW
TEST_F(l2_celu_test, l2_celu_normal_out_COMPLEX64_NCDHW)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_ratio_uint8_NDHWC
TEST_F(l2_celu_test, l2_celu_normal_ratio_uint8_NDHWC)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    uint8_t alpha = 1;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_alpha_1/2_NCHW_NHWC
TEST_F(l2_celu_test, l2_celu_normal_alpha_half_NCHW_NHWC)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW);
    float alpha = 0.5;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 正常场景_alpha_2_HWCN
TEST_F(l2_celu_test, l2_celu_normal_alpha_2_HWCN)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);
    float alpha = 2.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_celu_test, l2_celu_normal_empty_tensor)
{
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// CheckNotNull_self_nullptr
TEST_F(l2_celu_test, l2_celu_abnormal_self_nullptr)
{
    auto selfDesc = nullptr;
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_alpha_nullptr
TEST_F(l2_celu_test, l2_celu_abnormal_alpha_nullptr)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alphaDesc = nullptr;
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_out_nullptr
TEST_F(l2_celu_test, l2_celu_abnormal_out_nullptr)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_self_DOUBLE
TEST_F(l2_celu_test, l2_celu_abnormal_self_DOUBLE)
{
    auto selfDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_BF16
TEST_F(l2_celu_test, l2_celu_abnormal_self_BF16)
{
    auto selfDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDtypeValid_self_INT32
TEST_F(l2_celu_test, l2_celu_abnormal_self_INT32)
{
    auto selfDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_INT64
TEST_F(l2_celu_test, l2_celu_abnormal_self_INT64)
{
    auto selfDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_INT16
TEST_F(l2_celu_test, l2_celu_abnormal_self_INT16)
{
    auto selfDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_INT8
TEST_F(l2_celu_test, l2_celu_abnormal_self_INT8)
{
    auto selfDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_UINT8
TEST_F(l2_celu_test, l2_celu_abnormal_self_UINT8)
{
    auto selfDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_BOOL
TEST_F(l2_celu_test, l2_celu_abnormal_self_BOOL)
{
    auto selfDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_COMPLEX64
TEST_F(l2_celu_test, l2_celu_abnormal_self_COMPLEX64)
{
    auto selfDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_COMPLEX128
TEST_F(l2_celu_test, l2_celu_abnormal_self_COMPLEX128)
{
    auto selfDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_self_UNDEFINED
TEST_F(l2_celu_test, l2_celu_abnormal_self_UNDEFINED)
{
    auto selfDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckPromoteType_out_INT32
TEST_F(l2_celu_test, l2_celu_abnormal_out_INT32)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_unequal
TEST_F(l2_celu_test, l2_celu_abnormal_shape_unequal)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_10D
TEST_F(l2_celu_test, l2_celu_abnormal_shape_10D)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckAttributeValue_alpha_0
TEST_F(l2_celu_test, l2_celu_abnormal_alpha_0)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 0.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 数据范围[-1，1]
TEST_F(l2_celu_test, l2_celu_normal_valuerange)
{
    auto selfDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    ut.TestPrecision();
}

// 非连续
TEST_F(l2_celu_test, l2_celu_normal_uncontiguous)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    float alpha = 1.0;
    auto alphaDesc = ScalarDesc(alpha);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnCelu, INPUT(selfDesc, alphaDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // precision simulate
    // ut.TestPrecision();
}