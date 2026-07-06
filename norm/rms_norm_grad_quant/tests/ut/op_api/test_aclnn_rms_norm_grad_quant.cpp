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
#include <vector>

#include "gtest/gtest.h"

#include "norm/rms_norm_grad_quant/op_api/aclnn_rms_norm_grad_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

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
