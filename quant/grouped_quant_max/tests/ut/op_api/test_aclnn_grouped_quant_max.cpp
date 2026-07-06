/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "quant/grouped_quant_max/op_api/aclnn_grouped_quant_max.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2GroupedQuantMaxTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2GroupedQuantMaxTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2GroupedQuantMaxTest TearDown" << std::endl; }
};

// Test null input x
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_null_x)
{
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = nullptr;
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// Test null scale
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_null_scale)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = nullptr;
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// Test null groupList
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_null_group_list)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = nullptr;
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// Test null y
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_null_y)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = nullptr;
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// Test null amax
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_null_amax)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// Test unsupported x dtype (INT32)
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_unsupported_dtype)
{
    auto xDesc = TensorDesc({6, 128}, ACL_INT32, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test invalid dstType
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_invalid_dst_type)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = -1;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test dstType out of valid range
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_dst_type_out_of_range)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 37;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test invalid scale dtype (should be FLOAT)
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_invalid_scale_dtype)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test invalid groupList dtype (should be INT64)
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_invalid_group_list_dtype)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test y shape != x shape
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_y_shape_mismatch)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({128, 6}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test amax dtype != x dtype
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_amax_dtype_mismatch)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test roundMode mismatch: FLOAT8_E5M2 + "round"
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_roundmode_mismatch_e5m2)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "round";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test roundMode mismatch: HIFLOAT8 + "rint"
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_roundmode_mismatch_hifloat8)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 34;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test roundMode nullptr
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_roundmode_nullptr)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char* roundMode = nullptr;
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// Test y dtype mismatch with dstType
TEST_F(l2GroupedQuantMaxTest, l2_grouped_quant_max_y_dtype_mismatch_dsttype)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35; // E5M2, but y dtype is HIFLOAT8
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// DISABLED: Requires kernel execution
TEST_F(l2GroupedQuantMaxTest, DISABLED_l2_grouped_quant_max_fp32_to_fp8_e5m2)
{
    auto xDesc = TensorDesc({6, 128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({6, 128}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto amaxDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 35;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// DISABLED: Requires kernel execution
TEST_F(l2GroupedQuantMaxTest, DISABLED_l2_grouped_quant_max_fp16_to_fp8_e4m3fn)
{
    auto xDesc = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto amaxDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 36;
    char roundMode[] = "rint";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// DISABLED: Requires kernel execution
TEST_F(l2GroupedQuantMaxTest, DISABLED_l2_grouped_quant_max_bf16_to_hifloat8_hybrid)
{
    auto xDesc = TensorDesc({12, 32}, ACL_BF16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto groupListDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({12, 32}, ACL_HIFLOAT8, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto amaxDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND);

    aclTensor* x = xDesc.ToAclTypeRawPtr();
    aclTensor* scale = scaleDesc.ToAclTypeRawPtr();
    aclTensor* groupList = groupListDesc.ToAclTypeRawPtr();
    int64_t dstType = 34;
    char roundMode[] = "hybrid";
    aclTensor* y = yDesc.ToAclTypeRawPtr();
    aclTensor* amax = amaxDesc.ToAclTypeRawPtr();
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus ret = aclnnGroupedQuantMaxGetWorkspaceSize(x, scale, groupList, roundMode, dstType, y, amax,
                                                           &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}
