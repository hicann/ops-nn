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

#include "opdev/op_log.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "../../../op_api/aclnn_swish_backward.h"

using namespace op;
using namespace std;

class l2_swish_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "swish_backward Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "swish_backward Test TearDown" << std::endl;
    }
};

TEST_F(l2_swish_backward_test, swish_backward_testcase_001_normal_float32)
{
    auto gradOutputDesc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// float16
TEST_F(l2_swish_backward_test, swish_backward_testcase_002_normal_float16)
{
    auto gradOutputDesc =
        TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = ScalarDesc(0.01f, ACL_FLOAT16);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// bf16
TEST_F(l2_swish_backward_test, Ascend910B2_swish_backward_testcase_002_normal_bf16)
{
    auto gradOutputDesc =
        TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = (aclScalar*)nullptr;
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// empty
TEST_F(l2_swish_backward_test, swish_backward_testcase_003_normal_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(gradOutputDesc);
    auto betaDesc = ScalarDesc(static_cast<uint8_t>(0));
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull gradOutput
TEST_F(l2_swish_backward_test, swish_backward_testcase_004_exception_null_gradOutput)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT((aclTensor*)nullptr, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull self
TEST_F(l2_swish_backward_test, swish_backward_testcase_005_exception_null_self)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut =
        OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, (aclTensor*)nullptr, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull betaDesc
TEST_F(l2_swish_backward_test, swish_backward_testcase_006_exception_null_gradInput)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = (aclScalar*)nullptr;
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull gradInput
TEST_F(l2_swish_backward_test, swish_backward_testcase_007_exception_null_gradInput)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_swish_backward_test, swish_backward_testcase_008_exception_float64_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_swish_backward_test, swish_backward_testcase_009_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_swish_backward_test, swish_backward_testcase_010_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat 含ND不校验格式 TODO
TEST_F(l2_swish_backward_test, swish_backward_testcase_011_normal_float32)
{
    auto gradOutputDesc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = ScalarDesc(static_cast<uint8_t>(1));
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// CheckFormat 含ND不校验格式
TEST_F(l2_swish_backward_test, swish_backward_testcase_012_normal_format)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto betaDesc = ScalarDesc(static_cast<double>(0.0));
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckShape
TEST_F(l2_swish_backward_test, swish_backward_testcase_013_exception_different_shape)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_swish_backward_test, swish_backward_testcase_014_normal_not_contiguous_float)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                              .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto betaDesc = ScalarDesc(static_cast<int8_t>(-1));
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// not contiguous
TEST_F(l2_swish_backward_test, swish_backward_testcase_015_normal_not_contiguous_float16)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                              .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto betaDesc = ScalarDesc(static_cast<int8_t>(-1));
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_swish_backward_test, swish_backward_testcase_016_exception_complex64_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_swish_backward_test, swish_backward_testcase_017_exception_complex128_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// largeDim
TEST_F(l2_swish_backward_test, swish_backward_testcase_018_normal_large_dims)
{
    auto gradOutputDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto selfDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto betaDesc = ScalarDesc(static_cast<bool>(false));
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// empty
TEST_F(l2_swish_backward_test, swish_backward_testcase_019_normal_self_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// empty
TEST_F(l2_swish_backward_test, swish_backward_testcase_020_exception_gradOutput_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// InvalidDim
TEST_F(l2_swish_backward_test, swish_backward_testcase_021_exception_gradOutput_invalid_dim)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1, 1, 1, 1, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// InvalidDim
TEST_F(l2_swish_backward_test, swish_backward_testcase_022_exception_self_invalid_dim)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1, 1, 1, 1, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// InvalidDim
TEST_F(l2_swish_backward_test, swish_backward_testcase_022_exception_gradInput_invalid_dim)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(1.1f, ACL_FLOAT);
    auto gradInputDesc = TensorDesc({1, 16, 1, 1, 1, 1, 1, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSwishBackward, INPUT(gradOutputDesc, selfDesc, betaDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
