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
 * \file test_quant_matmul_dequant.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_quant_matmul_dequant.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
class l2_quant_matmul_dequant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_quant_matmul_dequant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_quant_matmul_dequant_test TearDown" << endl; }
};

TEST_F(l2_quant_matmul_dequant_test, ascend310P_success_nd_comboA)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto smooth_scale_desc = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(aclnnQuantMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, x_scale_desc, nullptr, smooth_scale_desc,
                              quantMode, true),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_success_nz_comboB)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({8, 32, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto weight_scale_desc = TensorDesc({512}, ACL_INT64, ACL_FORMAT_ND);
    auto x_scale_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto smooth_scale_desc = TensorDesc({256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(aclnnQuantMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, x_scale_desc, nullptr, smooth_scale_desc,
                              quantMode, true),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_success_nd_pad_k16)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({32, 16}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto smooth_scale_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(aclnnQuantMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, x_scale_desc, nullptr, smooth_scale_desc,
                              quantMode, true),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_success_empty)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({0, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({0, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, nullptr, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_success_nd_comboA_no_optional_scale)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, nullptr, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_invalid_transpose_false)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, nullptr, nullptr, quantMode, false),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_invalid_bias_not_null)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({512}, ACL_INT32, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, bias_desc, nullptr, nullptr, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_invalid_xoffset_not_null)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x_offset_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, x_offset_desc, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_invalid_out_dtype_mismatch)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, nullptr, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_invalid_smooth_scale_dtype)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({8, 32, 16, 32}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ);
    auto weight_scale_desc = TensorDesc({512}, ACL_INT64, ACL_FORMAT_ND);
    auto x_scale_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto smooth_scale_desc = TensorDesc({256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(aclnnQuantMatmulDequant,
                        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, x_scale_desc, nullptr, smooth_scale_desc,
                              quantMode, true),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_SUCCESS);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_invalid_weight_not_align16)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 255}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 255}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, nullptr, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_dequant_test, ascend310P_invalid_weight_3d)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({1, 512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, nullptr, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_quant_matmul_dequant_test, unsupported_soc)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    auto x_desc = TensorDesc({64, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({512, 256}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({64, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    char* quantMode = "pertoken";
    auto ut = OP_API_UT(
        aclnnQuantMatmulDequant,
        INPUT(x_desc, weight_desc, weight_scale_desc, nullptr, nullptr, nullptr, nullptr, quantMode, true),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}
} // namespace
