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

#include "../../../op_host/op_api/aclnn_addmv.h"
#include "common/op_api_def.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_addmv_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "addmv_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "addmv_test TearDown" << endl;
    }
};

// 输入float
TEST_F(l2_addmv_test, input_float32)
{
    auto input_tensor_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto vec_tensor_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto alpha_scalar_desc = ScalarDesc(5.9f);
    auto beta_scalar_desc = ScalarDesc(4.9f);

    auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.005, 0.005);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入float16
TEST_F(l2_addmv_test, input_float16)
{
    auto input_tensor_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto vec_tensor_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto alpha_scalar_desc = ScalarDesc(5.9f);
    auto beta_scalar_desc = ScalarDesc(4.9f);

    auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.005, 0.005);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

// 输入nullptr
TEST_F(l2_addmv_test, input_nullptr)
{
    auto input_tensor_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto mat_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto vec_tensor_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto alpha_scalar_desc = ScalarDesc(5.9f);
    auto beta_scalar_desc = ScalarDesc(4.9f);

    auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.0001, 0.0001);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    uint64_t workspace_size = 0;

    auto ut = OP_API_UT(
        aclnnAddmv, INPUT(nullptr, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, nullptr, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, nullptr, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, nullptr, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, nullptr),
        OUTPUT(out_tensor_desc), cubeMathType);
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut5 = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(nullptr), cubeMathType);
    aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输入数据类型遍历
TEST_F(l2_addmv_test, input_alldtype_in_list)
{
    vector<aclDataType> vaild_dtype_list{ACL_FLOAT, ACL_FLOAT16, ACL_DOUBLE, ACL_INT8, ACL_INT32,
                                         ACL_UINT8, ACL_INT16,   ACL_INT64,  ACL_BOOL};
    vector<aclDataType> invaild_dtype_list{ACL_COMPLEX64, ACL_COMPLEX128};
    for (auto dtype : vaild_dtype_list) {
        auto input_tensor_desc = TensorDesc({10}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto mat_tensor_desc = TensorDesc({10, 5}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto vec_tensor_desc = TensorDesc({5}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto alpha_scalar_desc = ScalarDesc(static_cast<bool>(1));
        auto beta_scalar_desc = ScalarDesc(static_cast<bool>(1));
        auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.005, 0.005);
        int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(
            aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
            OUTPUT(out_tensor_desc), cubeMathType);
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);

        // SAMPLE: precision simulate
        // ut.TestPrecision();
    }
    for (auto dtype : invaild_dtype_list) {
        auto input_tensor_desc = TensorDesc({10}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto mat_tensor_desc = TensorDesc({10, 5}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto vec_tensor_desc = TensorDesc({5}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto alpha_scalar_desc = ScalarDesc(static_cast<bool>(1));
        auto beta_scalar_desc = ScalarDesc(static_cast<bool>(1));
        auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.005, 0.005);
        int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(
            aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
            OUTPUT(out_tensor_desc), cubeMathType);
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 输入shape校验
TEST_F(l2_addmv_test, input_shape_check)
{
    auto input_tensor_desc = TensorDesc({10, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto mat_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto vec_tensor_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto alpha_scalar_desc = ScalarDesc(5.9f);
    auto beta_scalar_desc = ScalarDesc(4.9f);

    auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.0001, 0.0001);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;
    uint64_t workspace_size = 0;

    auto ut = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    input_tensor_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    mat_tensor_desc = TensorDesc({8, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto ut1 = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_addmv_test, alpha_0_beta_0)
{
    auto input_tensor_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat_tensor_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto vec_tensor_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto alpha_scalar_desc = ScalarDesc(0);
    auto beta_scalar_desc = ScalarDesc(0);

    auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.005, 0.005);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_addmv_test, input_empty)
{
    auto input_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat_tensor_desc = TensorDesc({0, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto vec_tensor_desc = TensorDesc({5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto alpha_scalar_desc = ScalarDesc(1.0f);
    auto beta_scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.005, 0.005);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_addmv_test, mat_empty)
{
    auto input_tensor_desc = TensorDesc({10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto mat_tensor_desc = TensorDesc({10, 0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto vec_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto alpha_scalar_desc = ScalarDesc(1.0f);
    auto beta_scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc = TensorDesc(input_tensor_desc).Precision(0.005, 0.005);
    int8_t cubeMathType = ALLOW_FP32_DOWN_PRECISION;

    auto ut = OP_API_UT(
        aclnnAddmv, INPUT(input_tensor_desc, mat_tensor_desc, vec_tensor_desc, alpha_scalar_desc, beta_scalar_desc),
        OUTPUT(out_tensor_desc), cubeMathType);

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}