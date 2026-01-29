/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <float.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_gelu_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_gelu_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_gelu_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_gelu_quant_test TearDown" << endl;
    }
};

TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_bf16_E4M3_tanh_dynamic_rint_scale)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "tanh";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_fp16_hi8_none_static_round_scale)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "round";
    int64_t dstType = static_cast<int64_t>(ACL_HIFLOAT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_fp32_uint8_tanh_static_round_scale)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "tanh";
    const char* quantMode = "static";
    const char* roundMode = "round";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_fp32_E5M2_none_dynamic_rint_scale)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 空tensor
// 静态输入scale为空
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_scale_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "tanh";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 动态态输出outscale为空
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_dynamic_outscale_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// self 输入为空
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_self_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(
            (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// y 输入为空
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_y_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT((aclTensor*)nullptr, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// approximate 输入为空
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_approximate_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = nullptr;
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// approximate 非法
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_approximate_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "erf";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// quantMode 输入为空
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_quantMode_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_UINT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = nullptr;
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_UINT8);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// quantMode 是否合法
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_quantMode_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dy";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// roundMode 是否合法
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_roundMode_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "round";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// self 维度为9维
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_self_Dim_fail)
{
    TensorDesc x_desc = TensorDesc({64, 1, 1, 1, 1, 1, 1, 1, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 1, 1, 1, 1, 1, 1, 1, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 动态 self 维度为1维
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_dynamic_self_Dim_fail)
{
    TensorDesc x_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 动态outscale 为 8维
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_dynamic_outScale_Dim8_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({64, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 scale 维度不为1
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_scale_Dim_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 offset 维度不为1
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_offset_Dim_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, offset_desc, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// self y shape 不一致
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_self_y_shape_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 10}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 动态 outscale shape
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_dynamic_outScale_shape_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            32,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 self scale dtype
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_self_scale_dtype_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 self dtype
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_self_dtype_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_UINT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 scale dtype
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_scale_dtype_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_UINT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 offset dtype
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_offset_dtype_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc(
        {
            1,
        },
        ACL_UINT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, offset_desc, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 scale offset dtype
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_scale_offset_dtype_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, offset_desc, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 静态 y dtype
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_y_dtype_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT16);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// y dstTpe diff
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_static_y_dsttype_diff_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "static";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT16);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 动态 outscale dtype
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_dynamic_outscale_dtype_fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant,
        INPUT(x_desc, (aclTensor*)nullptr, (aclTensor*)nullptr, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// dyn scale nullptr offset
TEST_F(l2_gelu_quant_test, ascend950_gelu_quant_dynamic_scale_offset__fail)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT16, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, (aclTensor*)nullptr, offset_desc, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// Soc 错误
TEST_F(l2_gelu_quant_test, ascend910B2_gelu_quant_fp32_E5M2_none_dynamic_rint_scale)
{
    TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc(
        {
            1,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    TensorDesc outScale_desc = TensorDesc(
        {
            64,
        },
        ACL_FLOAT, ACL_FORMAT_ND);
    const char* approximate = "none";
    const char* quantMode = "dynamic";
    const char* roundMode = "rint";
    int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
    auto ut = OP_API_UT(
        aclnnGeluQuant, INPUT(x_desc, scale_desc, offset_desc, approximate, quantMode, roundMode, dstType),
        OUTPUT(y_desc, outScale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}