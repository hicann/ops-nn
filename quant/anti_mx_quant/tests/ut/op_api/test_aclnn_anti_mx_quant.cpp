/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_anti_mx_quant.cpp
 * \brief
 */
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_anti_mx_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_anti_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_anti_mx_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_anti_mx_quant_test TearDown" << endl; }
};

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_e4m3_bf16_success)
{
    TensorDesc x_desc = TensorDesc({1, 4}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({1, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND);
    int64_t axis = -1;
    int64_t dstType = static_cast<int64_t>(ACL_BF16);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_e4m3_fp32_axis1_success)
{
    TensorDesc x_desc = TensorDesc({2, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({2, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t axis = 1;
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_empty_tensor_success)
{
    TensorDesc x_desc = TensorDesc({0, 4}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({0, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({0, 4}, ACL_BF16, ACL_FORMAT_ND);
    int64_t axis = -1;
    int64_t dstType = static_cast<int64_t>(ACL_BF16);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspace_size, 0);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_e4m3_fp32_axis1_block_boundary_success)
{
    TensorDesc x_desc = TensorDesc({2, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({2, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t axis = 1;
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_mxscale_block_boundary_fail)
{
    TensorDesc x_desc = TensorDesc({2, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({2, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 128}, ACL_BF16, ACL_FORMAT_ND);
    int64_t axis = 1;
    int64_t dstType = static_cast<int64_t>(ACL_BF16);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_mxscale_zero_dim_fail)
{
    TensorDesc x_desc = TensorDesc({1, 4}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({1, 1, 0}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND);
    int64_t axis = -1;
    int64_t dstType = static_cast<int64_t>(ACL_BF16);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_mxscale_shape_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({1, 4}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({1, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND);
    int64_t axis = -1;
    int64_t dstType = static_cast<int64_t>(ACL_BF16);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_y_shape_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({1, 4}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({1, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);
    int64_t axis = -1;
    int64_t dstType = static_cast<int64_t>(ACL_BF16);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_anti_mx_quant_test, ascend950_anti_mx_quant_axis_out_of_range_fail)
{
    TensorDesc x_desc = TensorDesc({1, 4}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc mxscale_desc = TensorDesc({1, 1, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({1, 4}, ACL_BF16, ACL_FORMAT_ND);
    int64_t axis = 2;
    int64_t dstType = static_cast<int64_t>(ACL_BF16);

    class SocVersionManager testSocVersion(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnAntiMxQuant, INPUT(x_desc, mxscale_desc, axis, dstType), OUTPUT(y_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
