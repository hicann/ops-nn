/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "norm/rms_norm_grad_quant/op_api/aclnn_rms_norm_grad_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_rms_norm_grad_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "rms_norm_grad_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "rms_norm_grad_quant_test TearDown" << endl; }
};

TEST_F(l2_rms_norm_grad_quant_test, case_fp16_quant_hifloat8_001)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_bf16_quant_hifloat8_002)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_fp16_quant_int8_003)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_fp16_quant_int8_div_mode_false_004)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = false;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ==================== Additional dtype combinations ====================

TEST_F(l2_rms_norm_grad_quant_test, case_bf16_quant_int8_005)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_fp32_quant_hifloat8_006)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_fp32_quant_int8_007)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_fp16_gamma_fp16_scales_fp16_hifloat8_008)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_bf16_gamma_bf16_scales_bf16_int8_div_false_009)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = false;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_fp16_scales_fp16_int8_010)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_bf16_gamma_bf16_hifloat8_011)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ==================== Optional parameter: offsetXOptional=nullptr ====================

TEST_F(l2_rms_norm_grad_quant_test, case_offset_nullptr_012)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              nullptr, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ==================== Higher dimension shapes ====================

TEST_F(l2_rms_norm_grad_quant_test, case_3d_shape_013)
{
    auto tensor_desc_dy = TensorDesc({2, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({2, 4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({2, 4, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_rms_norm_grad_quant_test, case_4d_shape_014)
{
    auto tensor_desc_dy = TensorDesc({2, 2, 4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({2, 2, 4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({2, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({2, 2, 4, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ==================== Empty tensor (gamma dim=0) ====================

TEST_F(l2_rms_norm_grad_quant_test, case_empty_gamma_015)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ==================== Null pointer negative cases ====================

TEST_F(l2_rms_norm_grad_quant_test, case_nullptr_dy_016)
{
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(nullptr, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rms_norm_grad_quant_test, case_nullptr_x_017)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, nullptr, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rms_norm_grad_quant_test, case_nullptr_rstd_018)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, nullptr, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rms_norm_grad_quant_test, case_nullptr_gamma_019)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, nullptr, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rms_norm_grad_quant_test, case_nullptr_scales_x_020)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, nullptr,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rms_norm_grad_quant_test, case_nullptr_dx_out_021)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dgamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, nullptr, tensor_desc_dgamma),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_rms_norm_grad_quant_test, case_nullptr_dgamma_out_022)
{
    auto tensor_desc_dy = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_rstd = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scales_x = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_offset_x = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto tensor_desc_dx = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    char quantMode[] = "static";
    bool divMode = true;

    auto ut = OP_API_UT(aclnnRmsNormGradQuant,
                        INPUT(tensor_desc_dy, tensor_desc_x, tensor_desc_rstd, tensor_desc_gamma, tensor_desc_scales_x,
                              tensor_desc_offset_x, quantMode, divMode, tensor_desc_dx, nullptr),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

namespace {
enum TensorIndex : size_t { DY, X, RSTD, GAMMA, SCALES, OFFSET, DX, DGAMMA, TENSOR_COUNT };
constexpr std::array<const char*, TENSOR_COUNT> TENSOR_NAMES = {"dy",     "x",      "rstd", "gamma",
                                                                "scales", "offset", "dx",   "dgamma"};

struct DtypeCase {
    std::string name;
    std::array<aclDataType, TENSOR_COUNT> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
                                                    ACL_FLOAT,   ACL_INT32,   ACL_INT8,  ACL_FLOAT};
    int64_t rows = 2;
    int64_t cols = 64;
    bool hasOffset = true;
    aclnnStatus expected = ACLNN_ERR_PARAM_INVALID;
    int nullIndex = -1;
    op::SocVersion soc = op::SocVersion::ASCEND950;
};

std::vector<DtypeCase> MakeDtypeCases()
{
    std::vector<DtypeCase> cases;
    for (size_t i = 0; i < TENSOR_COUNT; ++i) {
        DtypeCase param{"unsupported_" + std::string(TENSOR_NAMES[i])};
        param.dtypes[i] = ACL_UINT8;
        cases.push_back(param);
    }
    for (auto index : {X, GAMMA, SCALES}) {
        DtypeCase param{"mismatched_" + std::string(TENSOR_NAMES[index])};
        param.dtypes[index] = ACL_BF16;
        cases.push_back(param);
    }
    for (auto index : {GAMMA, SCALES}) {
        DtypeCase param{"fp32_dy_mismatched_" + std::string(TENSOR_NAMES[index])};
        param.dtypes[DY] = param.dtypes[X] = ACL_FLOAT;
        param.dtypes[index] = ACL_FLOAT16;
        cases.push_back(param);
    }
    DtypeCase issue{"issue_8105_uint8_dy_and_x"};
    issue.dtypes[DY] = issue.dtypes[X] = ACL_UINT8;
    issue.dtypes[GAMMA] = ACL_FLOAT16;
    issue.hasOffset = false;
    cases.push_back(issue);

    // Nine input combinations from the op definition, each with both output types and offset forms.
    struct InputDtypes {
        const char* name;
        aclDataType dy;
        aclDataType gamma;
        aclDataType scales;
    };
    const std::array<InputDtypes, 9> inputs = {{{"fp16_fp32_fp32", ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT},
                                                {"fp16_fp32_fp16", ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16},
                                                {"bf16_fp32_fp32", ACL_BF16, ACL_FLOAT, ACL_FLOAT},
                                                {"bf16_fp32_bf16", ACL_BF16, ACL_FLOAT, ACL_BF16},
                                                {"fp16_fp16_fp32", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT},
                                                {"fp32_fp32_fp32", ACL_FLOAT, ACL_FLOAT, ACL_FLOAT},
                                                {"bf16_bf16_fp32", ACL_BF16, ACL_BF16, ACL_FLOAT},
                                                {"fp16_fp16_fp16", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16},
                                                {"bf16_bf16_bf16", ACL_BF16, ACL_BF16, ACL_BF16}}};
    for (const auto& input : inputs) {
        for (auto dx : {ACL_INT8, ACL_HIFLOAT8}) {
            for (bool hasOffset : {false, true}) {
                DtypeCase param{"valid_" + std::string(input.name) + (dx == ACL_INT8 ? "_int8" : "_hifloat8") +
                                (hasOffset ? "_offset" : "_no_offset")};
                param.dtypes = {input.dy, input.dy, ACL_FLOAT, input.gamma, input.scales, ACL_INT32, dx, ACL_FLOAT};
                param.hasOffset = hasOffset;
                param.expected = ACL_SUCCESS;
                cases.push_back(param);
            }
        }
    }

    for (bool emptyGamma : {false, true}) {
        DtypeCase empty{emptyGamma ? "empty_gamma" : "empty_batch"};
        empty.rows = emptyGamma ? 2 : 0;
        empty.cols = emptyGamma ? 0 : 64;
        empty.expected = ACL_SUCCESS;
        cases.push_back(empty);
        for (auto index : {DY, GAMMA, DX}) {
            auto invalid = empty;
            invalid.name += "_unsupported_" + std::string(TENSOR_NAMES[index]);
            invalid.dtypes[index] = ACL_UINT8;
            invalid.expected = ACLNN_ERR_PARAM_INVALID;
            cases.push_back(invalid);
        }
    }
    for (auto index : {DY, X, RSTD, GAMMA, SCALES, DX, DGAMMA}) {
        DtypeCase param{"nullptr_before_dtype_" + std::string(TENSOR_NAMES[index])};
        param.dtypes[DY] = param.dtypes[X] = ACL_UINT8;
        param.nullIndex = static_cast<int>(index);
        param.expected = ACLNN_ERR_PARAM_NULLPTR;
        cases.push_back(param);
    }
    DtypeCase legacy{"ascend910b_empty_gamma_keeps_dtype_behavior"};
    legacy.cols = 0;
    legacy.dtypes.fill(ACL_UINT8);
    legacy.soc = op::SocVersion::ASCEND910B;
    legacy.expected = ACL_SUCCESS;
    cases.push_back(legacy);
    return cases;
}

class RmsNormGradQuantDtypeTest : public testing::TestWithParam<DtypeCase> {};

TEST_P(RmsNormGradQuantDtypeTest, GetWorkspaceSize)
{
    const auto& param = GetParam();
    op::SocVersionManager platform(param.soc);
    const std::array<std::vector<int64_t>, TENSOR_COUNT> shapes = {{{param.rows, param.cols},
                                                                    {param.rows, param.cols},
                                                                    {param.rows},
                                                                    {param.cols},
                                                                    {1},
                                                                    {1},
                                                                    {param.rows, param.cols},
                                                                    {param.cols}}};
    std::vector<std::unique_ptr<aclTensor, void (*)(aclTensor*)>> tensors;
    std::array<aclTensor*, TENSOR_COUNT> args = {};
    for (size_t i = 0; i < TENSOR_COUNT; ++i) {
        tensors.push_back(TensorDesc(shapes[i], param.dtypes[i], ACL_FORMAT_ND).ToAclType());
        args[i] = static_cast<int>(i) == param.nullIndex ? nullptr : tensors.back().get();
    }
    if (!param.hasOffset) {
        args[OFFSET] = nullptr;
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnRmsNormGradQuantGetWorkspaceSize(args[DY], args[X], args[RSTD], args[GAMMA], args[SCALES],
                                                        args[OFFSET], "static", true, args[DX], args[DGAMMA],
                                                        &workspaceSize, &executor);
    std::unique_ptr<aclOpExecutor> executorOwner(executor);
    EXPECT_EQ(status, param.expected);
    if (param.expected == ACL_SUCCESS && param.cols == 0) {
        EXPECT_EQ(workspaceSize, 0U);
    }
}

INSTANTIATE_TEST_SUITE_P(DtypeValidation, RmsNormGradQuantDtypeTest, testing::ValuesIn(MakeDtypeCases()),
                         [](const testing::TestParamInfo<DtypeCase>& info) { return info.param.name; });
} // namespace
