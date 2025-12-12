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
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_sigmoid.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_sigmoid_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Sigmoid Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Sigmoid Test TearDown" << endl; }
};

TEST_F(l2_sigmoid_test, case_1)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND)
        .ValueRange(-2, 2)
        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 空tensor
TEST_F(l2_sigmoid_test, case_2)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull self
TEST_F(l2_sigmoid_test, case_3)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(l2_sigmoid_test, case_4)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(tensor_desc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_sigmoid_test, case_5)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_INT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
}

// CheckDtypeValid
TEST_F(l2_sigmoid_test, case_6)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1},  ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_sigmoid_test, case_7)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1},  ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat
TEST_F(l2_sigmoid_test, case_8)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto out_desc = TensorDesc({1, 16, 1, 1},  ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFormat
TEST_F(l2_sigmoid_test, case_9)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1},  ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckShape
TEST_F(l2_sigmoid_test, case_10)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 1, 4, 4},  ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check format
TEST_F(l2_sigmoid_test, case_11)
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
        auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ValidList[i]);
        auto out_desc = TensorDesc({1, 16, 1, 1},  ACL_FLOAT, ValidList[i]);

        auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// CheckDtypeValid
TEST_F(l2_sigmoid_test, case_12)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckDtypeValid
TEST_F(l2_sigmoid_test, case_13)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_DOUBLE, ACL_FORMAT_ND)
        .ValueRange(-2, 2)
        .Value(vector<double>{1, 2, 3, 4, 5, -1000, 7, 8, 9, 100000, 11, 12, 13, 14, 461, 16});;
    auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_sigmoid_test, case_14)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10000, 10000);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// CheckDtypeValid
TEST_F(l2_sigmoid_test, case_15)
{
    auto self_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-100000, 100000);
    auto out_desc = TensorDesc({1, 16, 1, 1}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_16)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_BF16, ACL_FORMAT_ND)
        .ValueRange(-2, 2)
        .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto out_tensor_desc = TensorDesc(tensor_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// AICORE not contiguous
TEST_F(l2_sigmoid_test, case_17)
{
    auto self_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1000, 1000);
    auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// AICPU not contiguous
TEST_F(l2_sigmoid_test, case_18)
{
    auto self_desc = TensorDesc({5, 4}, ACL_COMPLEX64, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1000, 1000);
    auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// CheckDim
TEST_F(l2_sigmoid_test, case_19)
{
    auto self_tensor_desc = TensorDesc({1,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnSigmoid, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self的数据类型不在支持范围内
TEST_F(l2_sigmoid_test, l2_inplace_Sigmoid_test_int64) {
  auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}