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

#include "../../../../op_host/op_api/aclnn_add_layer_norm_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_add_layer_norm_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "add_layer_norm_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "add_layer_norm_quant_test TearDown" << endl; }
};

TEST_F(l2_add_layer_norm_quant_test, ascend950PR_9589_case_dyn_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_bias = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc(
        {
            8,
        },
        ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "dynamic";
    double eps = 1e-5;
    bool xout = true;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, tensor_desc_bias, tensor_desc_s1,
              tensor_desc_s2, (aclTensor*)nullptr, (aclTensor*)nullptr, quantMode, eps, xout, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_os1, tensor_desc_os2));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_layer_norm_quant_test, ascend950PR_9589_case_stc_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_bias = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z1 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_z2 = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "static";
    double eps = 1e-5;
    bool xout = true;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, tensor_desc_bias, tensor_desc_s1,
              tensor_desc_s2, tensor_desc_z1, tensor_desc_z2, quantMode, eps, xout, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_os1, tensor_desc_os2));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// static场景：外轴为空（如 x = [0, 64]），所有输出与 x1 同形必为空，无需下发算子
TEST_F(l2_add_layer_norm_quant_test, static_outer_axis_empty)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({0, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({0, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({0, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({0, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "static";
    auto ut = OP_API_UT(
        aclnnAddLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, (aclTensor*)nullptr, tensor_desc_s1,
              (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, quantMode, 1e-5, false, true),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_os1, tensor_desc_os2));

    uint64_t workspaceSize = 1;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}

// static场景：归一化轴为空（如 x = [2, 0]），输出同样必为空，无需下发算子
TEST_F(l2_add_layer_norm_quant_test, static_normalized_axis_empty)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({2, 0}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "static";
    auto ut = OP_API_UT(
        aclnnAddLayerNormQuant,
        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, (aclTensor*)nullptr, tensor_desc_s1,
              (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, quantMode, 1e-5, false, true),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_os1, tensor_desc_os2));

    uint64_t workspaceSize = 1;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0);
}
