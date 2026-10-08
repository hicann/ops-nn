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
#include "../../../op_host/op_api/aclnn_dynamic_quant_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_dynamic_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_dynamic_quant_v2_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_dynamic_quant_v2_test TearDown" << endl; }
};

TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_bf16)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({8, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = 2;

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_bf16_int4)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({8, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = 29;

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_bf16_int32)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({8, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT32);

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_bf16_int32_1)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({2, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT32);

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_dynamic_quant_v2_test, ascend950PR_9589_dynamic_quant_v2_bf16_int32_01_fail_groupindex)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({2, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = static_cast<int32_t>(ACL_INT32);

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue 6271: x 非空时, smooth_scales({0, 8}) 与 group_index({0,}) 的专家数为 0,
// 第一段接口应返回 161002(ACLNN_ERR_PARAM_INVALID), 而非 561103(ACLNN_ERR_INNER_NULLPTR)
TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_empty_expert_num_fail)
{
    TensorDesc x_desc = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({0, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({4, 8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = 2;

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x 为空 Tensor 时, 专家数为 0 的 MoE 输入合法, 走空 Tensor 提前返回, 返回成功
TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_empty_expert_num_empty_x)
{
    TensorDesc x_desc = TensorDesc({0, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({0, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({0, 8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = 2;

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// group_index 提供时, smooth_scales 必须为 2 维 (专家数, x 最后一维)
TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_moe_smooth_dim_fail)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = 2;

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// group_index 提供时, smooth_scales 的第一维(专家数)必须与 group_index 的维度大小一致
TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_moe_expert_num_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({2, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = 2;

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, smooth_desc, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// group_index 提供时, smooth_scales 必须非空
TEST_F(l2_dynamic_quant_v2_test, ascend910B2_dynamic_quant_v2_group_without_smooth_fail)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc moe_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dstType = 2;

    auto ut = OP_API_UT(aclnnDynamicQuantV2, INPUT(x_desc, nullptr, moe_desc, dstType),
                        OUTPUT(y_desc, scale_desc, offset_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
