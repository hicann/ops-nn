/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "quant/quant_max_v2/op_api/aclnn_quant_max_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2QuantMaxV2Test : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

// 空指针校验：x / y / amaxOut / roundMode 任一为空即拒绝。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_null_x)
{
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(nullptr, scaleDesc.ToAclTypeRawPtr(), nullptr, "rint", 35, 448.0f,
                                                      yDesc.ToAclTypeRawPtr(), amaxDesc.ToAclTypeRawPtr(), nullptr,
                                                      &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_null_y)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "rint", 35, 448.0f,
                                                      nullptr, amaxDesc.ToAclTypeRawPtr(), nullptr, &workspaceSize,
                                                      &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_null_amax)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "rint", 35, 448.0f,
                                                      yDesc.ToAclTypeRawPtr(), nullptr, nullptr, &workspaceSize,
                                                      &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 动态量化必须输出 yScale：不传 xScale 但 yScale 为 nullptr 时拒绝。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_dynamic_requires_yscale)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "rint", 35, 448.0f,
                                                      yDesc.ToAclTypeRawPtr(), amaxDesc.ToAclTypeRawPtr(), nullptr,
                                                      &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 静态量化不输出 yScale：传 xScale 的同时传非空 yScale 一律拒绝（与 tiling/InferShape 契约统一）。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_static_rejects_yscale)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(
        xDesc.ToAclTypeRawPtr(), scaleDesc.ToAclTypeRawPtr(), nullptr, "rint", 35, 448.0f, yDesc.ToAclTypeRawPtr(),
        amaxDesc.ToAclTypeRawPtr(), yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 动态量化 yScale 形状/dtype 校验：非 FLOAT 或 [num_groups] 不符即拒绝。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_dynamic_yscale_wrong_shape)
{
    auto xDesc = TensorDesc({8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({8, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND); // num_groups=2，传 3 应拒绝

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(
        xDesc.ToAclTypeRawPtr(), nullptr, groupListDesc.ToAclTypeRawPtr(), "rint", 35, 448.0f, yDesc.ToAclTypeRawPtr(),
        amaxDesc.ToAclTypeRawPtr(), yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// dstType 合法性：34/35/36 之外拒绝。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_invalid_dst_type)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "rint", 33, 448.0f,
                                                      yDesc.ToAclTypeRawPtr(), amaxDesc.ToAclTypeRawPtr(),
                                                      yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// roundMode 与 dstType 匹配：FP8 仅支持 rint，HiF8 支持 round/hybrid。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_roundmode_mismatch)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // dstType=35 (E5M2) 只允许 rint；round 应被拒绝。
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "round", 35, 448.0f,
                                                      yDesc.ToAclTypeRawPtr(), amaxDesc.ToAclTypeRawPtr(),
                                                      yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// dstTypeMax 越界：[0, 32768] 之外拒绝。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_dst_type_max_out_of_range)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "round", 34, 40000.0f,
                                                      yDesc.ToAclTypeRawPtr(), amaxDesc.ToAclTypeRawPtr(),
                                                      yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// x 维度范围 [2, 8]：1 维输入拒绝。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_x_rank_below_2)
{
    auto xDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({16}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "rint", 35, 448.0f,
                                                      yDesc.ToAclTypeRawPtr(), amaxDesc.ToAclTypeRawPtr(),
                                                      yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 分组数上限 384：groupList 长度 385 拒绝。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_group_count_over_limit)
{
    auto xDesc = TensorDesc({8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({8, 4}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({385}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({385}, ACL_INT64, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({385}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(
        xDesc.ToAclTypeRawPtr(), nullptr, groupListDesc.ToAclTypeRawPtr(), "rint", 35, 448.0f, yDesc.ToAclTypeRawPtr(),
        amaxDesc.ToAclTypeRawPtr(), yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径：动态单张量（无 scale / 无 groupList），参数齐全应通过第一段校验。
TEST_F(l2QuantMaxV2Test, l2_quant_max_v2_dynamic_tensor_ok)
{
    auto xDesc = TensorDesc({64, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({64, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnQuantMaxV2GetWorkspaceSize(xDesc.ToAclTypeRawPtr(), nullptr, nullptr, "rint", 35, 448.0f,
                                                      yDesc.ToAclTypeRawPtr(), amaxDesc.ToAclTypeRawPtr(),
                                                      yScaleDesc.ToAclTypeRawPtr(), &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}
