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

#include "../../../../op_host/op_api/aclnn_sigmoid_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_sigmoid_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Sigmoid Backward Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Sigmoid Backward Test TearDown" << endl; }
};

TEST_F(l2_sigmoid_backward_test, case_1)
{
    auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                            .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                            .Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 空tensor
TEST_F(l2_sigmoid_backward_test, case_2)
{
    auto grad_output_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc(grad_output_desc);
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull grad_output output
TEST_F(l2_sigmoid_backward_test, case_3)
{
    auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_desc = TensorDesc(grad_output_desc);
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(nullptr, output_desc), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, nullptr), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    workspaceSize = 0;
    aclRet = ut_2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull grad_input
TEST_F(l2_sigmoid_backward_test, case_4)
{
    auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradInputDesc = nullptr;
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_sigmoid_backward_test, case_5)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_DOUBLE,
        ACL_COMPLEX64,
        ACL_COMPLEX128,
        ACL_DT_UNDEFINED};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ValidList[i], ACL_FORMAT_ND);
        auto output_desc = TensorDesc({1, 16, 1, 1}, ValidList[i], ACL_FORMAT_ND);
        auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        //  ai_cpu文档中表示支持complex类型，实际测试时发现kernel并不支持，因此hostapi不开发complex类型的支持
        if (ValidList[i] != ACL_DT_UNDEFINED) {
            EXPECT_EQ(aclRet, ACL_SUCCESS);
        } else {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        }
    }
}

// CheckDtype different dtype of input
TEST_F(l2_sigmoid_backward_test, case_6)
{
    auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// CheckFormat
TEST_F(l2_sigmoid_backward_test, case_7)
{
    vector<aclFormat> ValidList = {
        ACL_FORMAT_UNDEFINED,
        ACL_FORMAT_NCHW,
        ACL_FORMAT_NHWC,
        ACL_FORMAT_ND,
        ACL_FORMAT_NC1HWC0,
        ACL_FORMAT_FRACTAL_Z,
        ACL_FORMAT_NC1HWC0_C04,
        ACL_FORMAT_HWCN,
        ACL_FORMAT_NDHWC,
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_NCDHW,
        ACL_FORMAT_NDC1HWC0,
        ACL_FRACTAL_Z_3D};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }

    // different format between input and output
    auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckShape
TEST_F(l2_sigmoid_backward_test, case_8)
{
    auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Checkprecision of different dtype
TEST_F(l2_sigmoid_backward_test, case_9)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_DOUBLE,
        ACL_COMPLEX64,
        ACL_COMPLEX128};

    vector<float> Precision_d = {
        0.0001,
        0.001,
        0.0001,
        0.0001,
        0.0001};
    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ValidList[i], ACL_FORMAT_ND).ValueRange(-100, 100);
        auto output_desc = TensorDesc({1, 16, 1, 1}, ValidList[i], ACL_FORMAT_ND).ValueRange(-100, 100);
        auto grad_input_desc = TensorDesc(grad_output_desc).Precision(Precision_d[i], Precision_d[i]);
        auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // SAMPLE: precision simulate
        // ut.TestPrecision();
    }
}

// not contiguous
TEST_F(l2_sigmoid_backward_test, case_10)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_DOUBLE};

    vector<float> Precision_d = {
        0.0001,
        0.001,
        0.0001};
    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto grad_output_desc = TensorDesc({5, 4}, ValidList[i], ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);
        auto output_desc = TensorDesc({5, 4}, ValidList[i], ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(1, 10);
        auto grad_input_desc = TensorDesc(grad_output_desc).Precision(Precision_d[i], Precision_d[i]);
        auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        // SAMPLE: precision simulate
        ut.TestPrecision();
    }
}

// CheckDim
TEST_F(l2_sigmoid_backward_test, case_11)
{
    auto grad_output_desc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc(grad_output_desc);
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype different dtype of input
TEST_F(l2_sigmoid_backward_test, case_12)
{
    auto grad_output_desc = TensorDesc({1, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_input_desc = TensorDesc({1, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check bf16
TEST_F(l2_sigmoid_backward_test, case_13)
{
    auto grad_output_desc = TensorDesc({1, 16, 1, 1}, ACL_BF16, ACL_FORMAT_ND)
                            .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto output_desc = TensorDesc({1, 16, 1, 1}, ACL_BF16, ACL_FORMAT_ND)
                            .Value(vector<float>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2});
    auto grad_input_desc = TensorDesc(grad_output_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSigmoidBackward, INPUT(grad_output_desc, output_desc), OUTPUT(grad_input_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}