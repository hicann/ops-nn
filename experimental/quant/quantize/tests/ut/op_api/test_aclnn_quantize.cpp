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
 * \file test_aclnn_quantize.cpp
 * \brief Quantize aclnnQuantizeGetWorkspaceSize return-code UT (dtype/shape/axis matrix + error codes).
 */
#include <float.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_quantize.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_quantize_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_quantize_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_quantize_test TearDown" << endl; }
};

// ---- null pointer checks ----
TEST_F(l2_quantize_test, quantize_null_x)
{
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT((aclTensor*)nullptr, scalesDesc, zeroPointsDesc, ACL_INT8, 1),
                        OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_quantize_test, quantize_null_scales)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, (aclTensor*)nullptr, zeroPointsDesc, ACL_INT8, 1),
                        OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_quantize_test, quantize_null_out)
{
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto ut = OP_API_UT(aclnnQuantize, INPUT((aclTensor*)nullptr, scalesDesc, zeroPointsDesc, ACL_INT8, 1),
                        OUTPUT((aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// ---- shape / dtype / axis error checks ----
TEST_F(l2_quantize_test, quantize_scales_not_1d)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_scales_numel_not_equal_axis)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_scales_zp_numel_mismatch)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_axis_out_of_range)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 5), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_x_dtype_unsupported)
{
    auto xDesc = TensorDesc({3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quantize_test, quantize_y_dtype_illegal_float)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_FLOAT, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// ---- valid dtype matrix: x in {fp32,fp16,bf16}, y in {int8,uint8,int32}, zp in {none,int8,int32,bf16} ----
TEST_F(l2_quantize_test, quantize_ok_fp32_per_channel_int8_zp_int8)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_quantize_test, quantize_ok_fp32_per_tensor_uint8_no_zp)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_UINT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, (aclTensor*)nullptr, ACL_UINT8, 1),
                        OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_quantize_test, quantize_ok_fp32_per_channel_int32_zp_int32)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT32, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT32, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_quantize_test, quantize_ok_fp16_per_channel_int8_zp_int32)
{
    auto xDesc = TensorDesc({3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_quantize_test, quantize_ok_bf16_all_consistent_int8_zp_bf16)
{
    auto xDesc = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto zeroPointsDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 5);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT8, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, zeroPointsDesc, ACL_INT8, 1), OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_quantize_test, quantize_ok_bf16_x_fp32_scales_int32_no_zp)
{
    auto xDesc = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalesDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outTensorDesc = TensorDesc({3, 2}, ACL_INT32, ACL_FORMAT_ND).ValidCount(6);
    auto ut = OP_API_UT(aclnnQuantize, INPUT(xDesc, scalesDesc, (aclTensor*)nullptr, ACL_INT32, 1),
                        OUTPUT(outTensorDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}
