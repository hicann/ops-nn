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
#include "level2/aclnn_gelu_backward_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_gelu_backward_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Gelu Backward V2 Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "Gelu Backward V2 Test TearDown" << std::endl;
    }
};

TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_001_normal_float32)
{
    auto gradOutputDesc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// float16
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_002_normal_float16)
{
    auto gradOutputDesc =
        TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.01, 0.01);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// empty
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_003_normal_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(gradOutputDesc);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_004_exception_null_gradOutput)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT((aclTensor*)nullptr, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_005_exception_null_self)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "none";
    auto ut =
        OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, (aclTensor*)nullptr, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_006_exception_null_approximate)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, nullptr), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull grad_input
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_007_exception_null_gradInput)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = (aclTensor*)nullptr;
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_008_exception_float64_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_009_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckDtype different dtype of input
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_010_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(selfDesc).Precision(0.001, 0.001);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckAttr 不合法
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_011_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(selfDesc).Precision(0.001, 0.001);
    char* approximate = "test";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat 含ND不校验格式
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_012_normal_format)
{
    vector<aclFormat> ValidList = {
        ACL_FORMAT_ND,
        ACL_FORMAT_NC1HWC0,
        ACL_FORMAT_FRACTAL_NZ,
    };

    int length = ValidList.size();
    char* approximate = "none";
    for (int i = 0; i < length; i++) {
        auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);

        auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// CheckFormat 含ND不校验格式
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_013_normal_format)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_014_normal_different_format)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckShape
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_015_exception_different_shape)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_016_normal_not_contiguous_float)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                              .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// not contiguous
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_017_normal_not_contiguous_float16)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                              .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.01, 0.01);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_018_exception_complex64_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_019_exception_complex128_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// largeDim
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_020_normal_large_dims)
{
    auto gradOutputDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto selfDesc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// empty
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_021_normal_self_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// empty
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_022_exception_gradOutput_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// empty
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_023_normal_broadcast_shape)
{
    auto gradOutputDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(selfDesc);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty
TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_024_normal_broadcast_shape)
{
    auto gradOutputDesc = TensorDesc({16, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    char* approximate = "tanh";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}