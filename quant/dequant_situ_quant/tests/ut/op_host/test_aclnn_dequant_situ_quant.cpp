/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_dequant_situ_quant.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_dequant_situ_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_dequant_situ_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_dequant_situ_quant_test TearDown" << endl; }
};

// issue 8380: x为空指针
TEST_F(l2_dequant_situ_quant_test, x_nullptr)
{
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(
        aclnnDequantSituQuant,
        INPUT((aclTensor*)nullptr, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8380: yOut为空指针
TEST_F(l2_dequant_situ_quant_test, y_nullptr)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT((aclTensor*)nullptr, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8380: yScaleOut为空指针
TEST_F(l2_dequant_situ_quant_test, y_scale_nullptr)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8380: INT8时weightScaleOptional为空指针
TEST_F(l2_dequant_situ_quant_test, int8_weight_scale_nullptr)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8380: quantTypeOptional为static时quantScaleOptional为空指针
TEST_F(l2_dequant_situ_quant_test, int8_static_quant_scale_nullptr)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8380: INT8 + dynamic量化时weightScaleOptional为空指针
TEST_F(l2_dequant_situ_quant_test, int8_dynamic_weight_scale_nullptr)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("dynamic");

    auto ut = OP_API_UT(
        aclnnDequantSituQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8380: INT32时weightScaleOptional为空指针
TEST_F(l2_dequant_situ_quant_test, int32_weight_scale_nullptr)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT32, ACL_FORMAT_ND);
    auto activation_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("dynamic");

    auto ut = OP_API_UT(
        aclnnDequantSituQuant,
        INPUT(x_desc, (aclTensor*)nullptr, activation_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8380: INT32时activationScaleOptional为空指针
TEST_F(l2_dequant_situ_quant_test, int32_activation_scale_nullptr)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT32, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("dynamic");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// issue 8381: x、yOut、yScaleOut全为空tensor
TEST_F(l2_dequant_situ_quant_test, empty_tensor_all)
{
    auto x_desc = TensorDesc({0, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({0, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue 8381: 仅x为空tensor，yOut/yScaleOut非空时也需返回161002，不能落到tiling
TEST_F(l2_dequant_situ_quant_test, empty_tensor_only_x)
{
    auto x_desc = TensorDesc({0, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 161002: quantTypeOptional不在支持范围内
TEST_F(l2_dequant_situ_quant_test, quant_type_invalid)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("unknown");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 161002: beta为0
TEST_F(l2_dequant_situ_quant_test, beta_zero)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 0.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 161002: x最后一维不是2的倍数
TEST_F(l2_dequant_situ_quant_test, x_last_dim_not_even)
{
    auto x_desc = TensorDesc({16, 63}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({63}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 31}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// quantTypeOptional为nullptr时按默认dynamic处理：INT8下weightScale仍为必选
TEST_F(l2_dequant_situ_quant_test, quant_type_nullptr_default_dynamic)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = nullptr;

    auto ut = OP_API_UT(
        aclnnDequantSituQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// quantTypeOptional大小写不敏感："STATIC"按static处理，需校验quantScaleOptional必选
TEST_F(l2_dequant_situ_quant_test, quant_type_uppercase_static)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("STATIC");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// yOut数据格式为非线性格式（NHWC）时返回161002
TEST_F(l2_dequant_situ_quant_test, yout_format_nhwc_invalid)
{
    auto x_desc = TensorDesc({2, 2, 2, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({2, 2, 2, 32}, ACL_INT8, ACL_FORMAT_NHWC);
    auto y_scale_desc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// INT32 x维度不等于2时返回161002
TEST_F(l2_dequant_situ_quant_test, int32_x_rank_not_two)
{
    auto x_desc = TensorDesc({2, 16, 64}, ACL_INT32, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto activation_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("dynamic");

    auto ut = OP_API_UT(
        aclnnDequantSituQuant,
        INPUT(x_desc, weight_scale_desc, activation_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// yOut维度与x不匹配时返回161002
TEST_F(l2_dequant_situ_quant_test, yout_rank_mismatch)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// yScaleOut维度与x不匹配时返回161002
TEST_F(l2_dequant_situ_quant_test, yscale_rank_mismatch)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 合法入参（INT8 static，含bias/quantOffset）：应通过第一段入参校验
// inner接口在UT环境下的返回值依赖平台桩实现，此处只断言未命中入参错误码
TEST_F(l2_dequant_situ_quant_test, valid_int8_static_params)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_offset_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, bias_desc, quant_scale_desc,
                              quant_offset_desc, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 合法入参（INT8 static，未传quantOffset）：quantOffsetOptional在static下为可选，缺省按0处理，应通过入参校验
TEST_F(l2_dequant_situ_quant_test, valid_int8_static_without_quant_offset)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 合法入参（INT32 dynamic，含activationScale/groupIndex）：应通过第一段入参校验
TEST_F(l2_dequant_situ_quant_test, valid_int32_dynamic_params)
{
    auto x_desc = TensorDesc({16, 64}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto activation_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto group_index_desc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("dynamic");

    auto ut = OP_API_UT(
        aclnnDequantSituQuant,
        INPUT(x_desc, weight_scale_desc, activation_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, group_index_desc, beta, linear_beta, activate_left, quant_type),
        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 161002: x数据类型不在支持范围内
TEST_F(l2_dequant_situ_quant_test, x_dtype_invalid)
{
    auto x_desc = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto quant_scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    auto y_scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    const float beta = 4.0f;
    const float linear_beta = 25.0f;
    const bool activate_left = true;
    char* quant_type = const_cast<char*>("static");

    auto ut = OP_API_UT(aclnnDequantSituQuant,
                        INPUT(x_desc, weight_scale_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, quant_scale_desc,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, beta, linear_beta, activate_left, quant_type),
                        OUTPUT(y_desc, y_scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
