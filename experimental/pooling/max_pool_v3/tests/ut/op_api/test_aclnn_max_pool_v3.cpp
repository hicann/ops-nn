/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_max_pool_v3.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class max_pool_v3_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "max_pool_v3_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "max_pool_v3_test TearDown" << endl; }
};

// ---------------------------------------------------------------------------
// case_001: 基础 FP32 max pool, kernel=2x2, stride=2x2, 无padding
// 输入 [1,1,4,4] → 输出 [1,1,2,2]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_001_float)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 15);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_002: FP16 max pool, kernel=3x3, stride=1x1, 无padding
// 输入 [1,2,6,6] → 输出 [1,2,4,4]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_002_float16)
{
    auto xDesc = TensorDesc({1, 2, 6, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 2, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 1, 1});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_003: BF16 max pool, kernel=2x2, stride=2x2
// bf16 仅支持 910B 及以上平台
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_003_bfloat16)
{
    auto xDesc = TensorDesc({1, 3, 8, 8}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({1, 3, 4, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// ---------------------------------------------------------------------------
// case_004: FP32, kernel=2x2, stride=2x2, 带padding
// 输入 [1,1,4,4] + pad[0,1,0,1] → 输出 [1,1,3,3]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_004_with_padding)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 15);
    auto yDesc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 1, 0, 1});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_005: ceil_mode=true, 输入 [1,1,5,5], kernel=2x2, stride=2x2
// ceil_mode=false → [1,1,2,2], ceil_mode=true → [1,1,3,3]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_005_ceil_mode)
{
    auto xDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 24);
    auto yDesc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(1));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_006: 大批量 + 多通道, kernel=2x2, stride=2x2
// 输入 [4,16,32,32] → 输出 [4,16,16,16]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_006_large_batch)
{
    auto xDesc = TensorDesc({4, 16, 32, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({4, 16, 16, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_007: 大kernel=5x5, stride=1x1
// 输入 [1,1,10,10] → 输出 [1,1,6,6]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_007_large_kernel)
{
    auto xDesc = TensorDesc({1, 1, 10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto yDesc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 5, 5});
    auto stridesDesc = IntArrayDesc({1, 1, 1, 1});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_008: kernel=3x3, stride=2x2, 全padding模式
// 输入 [1,1,4,4] + pad[1,1,1,1] → 输出 [1,1,2,2]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_008_full_padding)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 15);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({1, 1, 1, 1});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_009: kernel=2x2, stride=3x3 (stride > kernel), 无padding
// 输入 [1,1,6,6] → 输出 [1,1,2,2]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_009_stride_larger_than_kernel)
{
    auto xDesc = TensorDesc({1, 1, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 35);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 3, 3});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_010: 不支持的类型 double
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_010_invalid_type_double)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ---------------------------------------------------------------------------
// case_011: 不支持的类型 int32
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_011_invalid_type_int32)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_INT32, ACL_FORMAT_ND);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ---------------------------------------------------------------------------
// case_012: 空tensor
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_012_empty_tensor)
{
    auto xDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({1, 1, 0, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_013: nullptr 测试 — x 为空
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_013_nullptr_x)
{
    auto tensorDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut1 = OP_API_UT(aclnnMaxPoolV3, INPUT(nullptr, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc),
                         OUTPUT(tensorDesc));
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ---------------------------------------------------------------------------
// case_014: nullptr 测试 — ksize 为空
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_014_nullptr_ksize)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;

    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut2 = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, nullptr, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));
    aclnnStatus aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ---------------------------------------------------------------------------
// case_015: nullptr 测试 — strides 为空
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_015_nullptr_strides)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut3 = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, nullptr, padsDesc, ceilModeDesc), OUTPUT(yDesc));
    aclnnStatus aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ---------------------------------------------------------------------------
// case_016: nullptr 测试 — 输出为空
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_016_nullptr_output)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut4 = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(nullptr));
    aclnnStatus aclRet = ut4.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// ---------------------------------------------------------------------------
// case_017: 池化算子InferShape自动修正输出shape，此测试验证框架正常处理
// 输入 [1,1,4,4], kernel=2x2, stride=1x1 → 正确输出 [1,1,3,3]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_017_kernel2_stride1)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 15);
    auto yDesc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 1, 1});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_018: 输入输出dtype不匹配
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_018_output_dtype_mismatch)
{
    auto xDesc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 15);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ---------------------------------------------------------------------------
// case_019: 非连续tensor
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_019_not_contiguous)
{
    auto xDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto yDesc = TensorDesc({1, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // 非连续tensor会自动做Contiguous处理，返回ACL_SUCCESS
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_020: ceil_mode=true + padding, 边界条件测试
// 输入 [1,1,5,5], kernel=3x3, stride=2x2, pad[1,0,1,0], ceil_mode=true
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_020_ceil_mode_with_padding)
{
    auto xDesc = TensorDesc({1, 1, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 24);
    auto yDesc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({1, 0, 1, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(1));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// =========================================================================
// 扩展测试用例 (case_022 ~ case_033): 满足精度评估 ≥30 例的要求
// 覆盖: 多dtype、多shape规模、边界条件、生产场景
// =========================================================================

// case_021: 1x1 kernel (等效于恒等映射，无降采样)
// 输入 [1,3,7,7], kernel=1x1, stride=1x1 → 输出 [1,3,7,7]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_021_1x1_kernel)
{
    auto xDesc = TensorDesc({1, 3, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto yDesc = TensorDesc({1, 3, 7, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 1, 1});
    auto stridesDesc = IntArrayDesc({1, 1, 1, 1});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_022: FP16 中等特征图, kernel=3x3, stride=2x2
// 输入 [2,8,14,14] → 输出 [2,8,6,6]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_022_fp16_medium_feature_map)
{
    auto xDesc = TensorDesc({2, 8, 14, 14}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 6, 6}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_023: BF16 大批量+大特征图, kernel=2x2, stride=2x2
// 输入 [8,32,64,64] → 输出 [8,32,32,32]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_023_bf16_large_feature_map)
{
    auto xDesc = TensorDesc({8, 32, 64, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({8, 32, 32, 32}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// ---------------------------------------------------------------------------
// case_024: FP32 ResNet 下采样风格, kernel=3x3, stride=2x2, pad=1
// 输入 [1,64,56,56] → 输出 [1,64,28,28]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_024_resnet_downsample)
{
    auto xDesc = TensorDesc({1, 64, 56, 56}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto yDesc = TensorDesc({1, 64, 28, 28}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({1, 1, 1, 1});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_025: FP16 非对称padding, kernel=3x3, stride=2x2
// 输入 [1,3,15,15], pad=[1,0,2,0] → 输出 [1,3,8,7]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_025_asymmetric_padding)
{
    auto xDesc = TensorDesc({1, 3, 15, 15}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 3, 8, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({1, 0, 2, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_026: FP32 多通道, kernel=4x4, stride=4x4
// 输入 [1,128,16,16] → 输出 [1,128,4,4]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_026_channel_heavy_pool)
{
    auto xDesc = TensorDesc({1, 128, 16, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-3, 3);
    auto yDesc = TensorDesc({1, 128, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 4, 4});
    auto stridesDesc = IntArrayDesc({1, 1, 4, 4});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_027: FP16 kernel > 输入一半, 极端降采样
// 输入 [1,1,8,8], kernel=6x6, stride=2x2
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_027_large_kernel_relative)
{
    auto xDesc = TensorDesc({1, 1, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 1, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ksizeDesc = IntArrayDesc({1, 1, 6, 6});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_028: BF16 ceil_mode + padding 组合
// 输入 [2,6,9,9], kernel=3x3, stride=2x2, pad=[1,1,1,1], ceil=true
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_028_bf16_ceil_with_pad)
{
    auto xDesc = TensorDesc({2, 6, 9, 9}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 6, 5, 5}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({1, 1, 1, 1});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(1));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// ---------------------------------------------------------------------------
// case_029: FP32 极端stride (stride=4, kernel=2)
// 输入 [1,1,12,12], kernel=2x2, stride=4x4
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_029_extreme_stride)
{
    auto xDesc = TensorDesc({1, 1, 12, 12}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 143);
    auto yDesc = TensorDesc({1, 1, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 4, 4});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_030: FP16 VGG风格 pooling, kernel=2x2, stride=2x2
// 输入 [1,128,112,112] → 输出 [1,128,56,56]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_030_vgg_style_pool)
{
    auto xDesc = TensorDesc({1, 128, 112, 112}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({1, 128, 56, 56}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_031: FP32 全正值, kernel=2x2, stride=2x2
// 输入 [2,4,10,10] 全正值 [1,100]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_031_all_positive_values)
{
    auto xDesc = TensorDesc({2, 4, 10, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 100);
    auto yDesc = TensorDesc({2, 4, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 2, 2});
    auto stridesDesc = IntArrayDesc({1, 1, 2, 2});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_032: FP32 全负值, kernel=3x3, stride=3x3
// 输入 [1,2,9,9] 全负值 [-100, -1]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_032_all_negative_values)
{
    auto xDesc = TensorDesc({1, 2, 9, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-100, -1);
    auto yDesc = TensorDesc({1, 2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ksizeDesc = IntArrayDesc({1, 1, 3, 3});
    auto stridesDesc = IntArrayDesc({1, 1, 3, 3});
    auto padsDesc = IntArrayDesc({0, 0, 0, 0});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

// ---------------------------------------------------------------------------
// case_033: FP16 1x1 kernel + padding (恒等映射+pad)
// 输入 [2,3,8,8], kernel=1x1, stride=1x1, pad=[1,1,1,1]
// ---------------------------------------------------------------------------
TEST_F(max_pool_v3_test, case_033_identity_with_padding)
{
    auto xDesc = TensorDesc({2, 3, 8, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-5, 5);
    auto yDesc = TensorDesc({2, 3, 10, 10}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ksizeDesc = IntArrayDesc({1, 1, 1, 1});
    auto stridesDesc = IntArrayDesc({1, 1, 1, 1});
    auto padsDesc = IntArrayDesc({1, 1, 1, 1});
    auto ceilModeDesc = ScalarDesc(static_cast<int64_t>(0));

    auto ut = OP_API_UT(aclnnMaxPoolV3, INPUT(xDesc, ksizeDesc, stridesDesc, padsDesc, ceilModeDesc), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}
