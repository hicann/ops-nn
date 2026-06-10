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

#include "norm/add_layer_norm_quant_v2/op_api/aclnn_add_layer_norm_quant_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_add_layer_norm_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "add_layer_norm_quant_v2_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "add_layer_norm_quant_v2_test TearDown" << endl;
    }
};

TEST_F(l2_add_layer_norm_quant_v2_test, ascend910b_case_dyn_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_bias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_layernorm_res = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "dynamic";
    double eps = 1e-5;
    bool xout = true;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddLayerNormQuantV2,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, tensor_desc_bias, tensor_desc_s1,
            tensor_desc_s2, (aclTensor*)nullptr, (aclTensor*)nullptr, quantMode, eps, xout, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_layernorm_res, tensor_desc_os1, tensor_desc_os2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_add_layer_norm_quant_v2_test, ascend910b_case_stc_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_bias = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o1 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_o2 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_layernorm_res = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "static";
    double eps = 1e-5;
    bool xout = true;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddLayerNormQuantV2,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, tensor_desc_bias, tensor_desc_s1,
            tensor_desc_s2, tensor_desc_o1, tensor_desc_o2, quantMode, eps, xout, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_layernorm_res, tensor_desc_os1, tensor_desc_os2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_add_layer_norm_quant_v2_test, ascend910b_case_bf16_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_bias = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_layernorm_res = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "dynamic";
    double eps = 1e-5;
    bool xout = true;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddLayerNormQuantV2,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, tensor_desc_bias, tensor_desc_s1,
            (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, quantMode, eps, xout, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_layernorm_res, tensor_desc_os1, tensor_desc_os2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_add_layer_norm_quant_v2_test, ascend910b_case_fp32_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_bias = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_layernorm_res = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "dynamic";
    double eps = 1e-5;
    bool xout = true;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddLayerNormQuantV2,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, tensor_desc_bias, tensor_desc_s1,
            tensor_desc_s2, (aclTensor*)nullptr, (aclTensor*)nullptr, quantMode, eps, xout, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_layernorm_res, tensor_desc_os1, tensor_desc_os2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_add_layer_norm_quant_v2_test, ascend910b_case_no_bias_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_layernorm_res = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_os1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_os2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* quantMode = "dynamic";
    double eps = 1e-5;
    bool xout = true;
    bool divMode = true;

    auto ut = OP_API_UT(
        aclnnAddLayerNormQuantV2,
        INPUT(
            tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_beta, (aclTensor*)nullptr, tensor_desc_s1,
            (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, quantMode, eps, xout, divMode),
        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_layernorm_res, tensor_desc_os1, tensor_desc_os2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
