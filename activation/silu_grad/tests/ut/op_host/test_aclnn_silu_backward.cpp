/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "../../../op_host/op_api/aclnn_silu_backward.h"

using namespace op;
using namespace std;


class l2_silu_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "silu_backward Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "silu_backward Test TearDown" << std::endl; }
};

TEST_F(l2_silu_backward_test, silu_backward_testcase_001_normal_float32)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1,1,1,1,1,1,1,1,1,1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// float16
TEST_F(l2_silu_backward_test, silu_backward_testcase_002_normal_float16)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1,1,1,1,1,1,1,1,1,1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// bf16
TEST_F(l2_silu_backward_test, ascend910b2_silu_backward_testcase_002_normal_bf16)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1,1,1,1,1,1,1,1,1,1});
    auto selfDesc = TensorDesc({2, 5}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// empty
TEST_F(l2_silu_backward_test, silu_backward_testcase_003_normal_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc(gradOutputDesc);
    auto gradInputDesc = TensorDesc(gradOutputDesc);

    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull gradOutput
TEST_F(l2_silu_backward_test, silu_backward_testcase_004_exception_null_gradOutput)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT((aclTensor*)nullptr, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull self
TEST_F(l2_silu_backward_test, silu_backward_testcase_005_exception_null_self)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);

    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, (aclTensor*)nullptr), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull grad_input
TEST_F(l2_silu_backward_test, silu_backward_testcase_006_exception_null_gradInput)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = (aclTensor*)nullptr;
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT((aclTensor*)nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_silu_backward_test, silu_backward_testcase_007_exception_float64_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_silu_backward_test, silu_backward_testcase_008_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(selfDesc);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_silu_backward_test, silu_backward_testcase_009_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat 含ND不校验格式
TEST_F(l2_silu_backward_test, silu_backward_testcase_010_normal_format)
{
    vector<aclFormat> ValidList = {
        ACL_FORMAT_ND,
        ACL_FORMAT_NC1HWC0,
        ACL_FORMAT_FRACTAL_NZ,
        };

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

        // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// CheckFormat 含ND不校验格式
TEST_F(l2_silu_backward_test, silu_backward_testcase_011_normal_format)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckFormat
TEST_F(l2_silu_backward_test, silu_backward_testcase_012_normal_different_format)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    // EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckShape
TEST_F(l2_silu_backward_test, silu_backward_testcase_013_exception_different_shape)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// not contiguous
TEST_F(l2_silu_backward_test, silu_backward_testcase_014_normal_not_contiguous_float)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND,
        {1, 2}, 0, {5, 2}).Value(vector<float>{1,1,1,1,1,1,1,1,1,1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// not contiguous
TEST_F(l2_silu_backward_test, silu_backward_testcase_015_normal_not_contiguous_float16)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND,
        {1, 2}, 0, {5, 2}).Value(vector<float>{1,1,1,1,1,1,1,1,1,1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {5, 2}).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_silu_backward_test, silu_backward_testcase_016_exception_complex64_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_silu_backward_test, silu_backward_testcase_017_exception_complex128_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);
    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// largeDim
// TEST_F(l2_silu_backward_test, silu_backward_testcase_018_normal_large_dims)
// {
//     auto gradOutputDesc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1,1);
//     auto selfDesc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.0001, 0.0001);
//     auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

//     // SAMPLE: only test GetWorkspaceSize
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);

//     // SAMPLE: precision simulate
//     // ut.TestPrecision();
// }

// empty
TEST_F(l2_silu_backward_test, silu_backward_testcase_019_normal_self_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = TensorDesc(gradOutputDesc);

    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// empty
TEST_F(l2_silu_backward_test, silu_backward_testcase_020_exception_gradOutput_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc);

    auto ut = OP_API_UT(aclnnSiluBackward, INPUT(gradOutputDesc, selfDesc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
