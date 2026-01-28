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
#include "../../../op_host/op_api/aclnn_dynamic_quant_v3.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_dynamic_quant_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_dynamic_quant_v3_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_dynamic_quant_v3_test TearDown" << endl;
    }
};

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_fail_01)
{
    // isSymmetrical + offset not nullptr
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({2, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT8);
    const char* quantMode = "pertoken";
    bool isSymmetrical = true;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, smooth_desc, moe_desc, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_02)
{
    // group_index shape invalid
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({2, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT8);
    const char* quantMode = "pertoken";
    bool isSymmetrical = false;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, smooth_desc, moe_desc, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_03)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT8);
    const char* quantMode = "pertensor";
    bool isSymmetrical = false;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, smooth_desc, nullptr, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_fail_04)
{
    // isSymmetrical is true and offset is not nullptr
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT8);
    const char* quantMode = "perchannel";
    bool isSymmetrical = true;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, smooth_desc, nullptr, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_05)
{
    // isSymmetrical is true and offset is nullptr
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT8);
    const char* quantMode = "perchannel";
    bool isSymmetrical = true;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, smooth_desc, nullptr, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_shape_fail_06)
{
    // y shape is not equal to x shape
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({32, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT8);
    const char* quantMode = "perchannel";
    bool isSymmetrical = true;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, smooth_desc, nullptr, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_type_fail_07)
{
    // y type is not supported
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({16}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_BF16);
    const char* quantMode = "perchannel";
    bool isSymmetrical = true;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, smooth_desc, nullptr, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}

TEST_F(l2_dynamic_quant_v3_test, ascend910_9589_dynamic_quant_v3_bf16_int8_without_smooth_08)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT8);
    const char* quantMode = "perchannel";
    bool isSymmetrical = true;

    auto ut = OP_API_UT(
        aclnnDynamicQuantV3, INPUT(x_desc, nullptr, nullptr, dstType, isSymmetrical, quantMode),
        OUTPUT(y_desc, scale_desc, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_INNER);
}