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

#include "../../../op_host/op_api/aclnn_prelu_backward.h"

using namespace op;
using namespace std;


class l2_prelu_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "prelu_backward Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "prelu_backward Test TearDown" << std::endl; }
};

TEST_F(l2_prelu_backward_test, prelu_backward_testcase_001_normal_float32)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// float16
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_002_normal_float16_nchw)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_NCHW).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // CPU 不支持fp16
    // ut.TestPrecision();
}

// float16
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_003_normal_float16_nhwc)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC).Precision(0.001, 0.001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_NHWC).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    // CPU 不支持fp16
    // ut.TestPrecision();
}

TEST_F(l2_prelu_backward_test, prelu_backward_testcase_004_normal_ndhwc)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_backward_test, prelu_backward_testcase_005_normal_ncdhw)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// empty
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_006_normal_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckNotNull
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_007_exception_null_gradOutput)
{
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(nullptr, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_008_exception_null_self)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, nullptr, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_009_exception_null_weight)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, nullptr), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_010_exception_null_gradInput)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(nullptr, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_011_exception_null_gradweight)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_012_exception_float64_dtype_not_supported)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_013_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_014_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_015_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_016_normal_dtype_not_the_same)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// empty
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_017_normal_self_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// empty
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_018_exception_gradOutput_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({0, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// empty
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_019_exception_weight_invalid_shape)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_020_exception_different_shape)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_021_exception_weight_shape_invalid)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_022_weight_one_element)
{
    auto gradOutputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// CheckShape, weight的只有一个元素时, gradweight需要与weight的shape相同
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_023_weight_one_element)
{
    auto gradOutputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// CheckShape, weight的只有一个元素时, gradWeight需要与weight的shape相同
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_024_weight_one_element_multi_dim)
{
    auto gradOutputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// CheckShape
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_025_weight_shape_multi_dim)
{
    auto gradOutputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// not contiguous
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_026_normal_not_contiguous_float)
{
    auto gradOutputDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
    auto selfDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// largeDim
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_027_normal_large_dims)
{
    auto gradOutputDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// largeDim
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_028_normal_large_dims)
{
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check gradoutput shape invalid
TEST_F(l2_prelu_backward_test, prelu_backward_testcase_029_exception_gradOutput_shape)
{
    auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradOutputDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc({1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto gradWeightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnPreluBackward, INPUT(gradOutputDesc, selfDesc, weightDesc), OUTPUT(gradInputDesc, gradWeightDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
