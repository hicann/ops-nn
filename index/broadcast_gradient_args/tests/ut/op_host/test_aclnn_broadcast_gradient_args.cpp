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
#include <vector>

#include "../../../op_api/aclnn_broadcast_gradient_args.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

// 根据真实架构设置期望值：950上期望archExpect，非950架构因IsRegbase检查在参数校验之后，
// 正常用例(archExpect=ACL_SUCCESS)期望ACLNN_ERR_PARAM_INVALID（架构检查），错误用例期望各自错误码
static void ExpectArchCond(aclnnStatus aclRet, aclnnStatus archExpect)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        EXPECT_EQ(aclRet, archExpect);
    } else {
        EXPECT_EQ(aclRet, archExpect == ACL_SUCCESS ? ACLNN_ERR_PARAM_INVALID : archExpect);
    }
}

class l2_broadcast_gradient_args_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "broadcast_gradient_args_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "broadcast_gradient_args_test TearDown" << std::endl; }
};

// 正常用例：int32，标准广播场景
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_01_int32_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// int64数据类型
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_02_int64)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// x1/x2长度不同
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_03_different_len)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 空shape
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_04_empty_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto y1Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 完全相同的shape（输出为空）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_05_all_equal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// dtype不一致（应报错）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_06_dtype_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不支持的dtype（应报错）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_07_unsupport_dtype)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto y1Desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y2Desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法输入shape（非1D，应报错）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_08_not_1d)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6, 0});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空指针（应报错）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_09_nullptr)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    // x1为空
    auto ut_x1 = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(nullptr, x2Desc), OUTPUT(y1Desc, y2Desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_x1.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // y1为空
    auto ut_y1 = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(nullptr, y2Desc));
    aclRet = ut_y1.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输出dtype与输入不一致（应报错）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_10_output_dtype_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 泛化用例：广播模式变化 ====================

// 单维广播：x1={1}, x2={5}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_11_single_dim_broadcast)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto x2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{5});
    auto y1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// x1全为1：x1={1,1,1}, x2={2,3,4}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_12_x1_all_ones)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 1, 1});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 4});
    auto y1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// x2全为1：x1={2,3,4}, x2={1,1,1}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_13_x2_all_ones)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 4});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 1, 1});
    auto y1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 交替模式：x1={1,2,1,2}, x2={2,1,2,1}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_14_alternating_pattern)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 2, 1, 2});
    auto x2Desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 2, 1});
    auto y1Desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 相同位置都有1：x1={1,2,1}, x2={1,1,1}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_15_both_ones_same_pos)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 2, 1});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 1, 1});
    auto y1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 大rank(8维)：x1={2,1,3,1,4,1,5,1}, x2={2,9,3,9,4,9,5,9}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_16_large_rank_8)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 3, 1, 4, 1, 5, 1});
    auto x2Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 9, 3, 9, 4, 9, 5, 9});
    auto y1Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// ==================== 泛化用例：长度差异 ====================

// x1远短于x2：x1={1}, x2={2,3,4,5,6,7,8}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_17_x1_much_shorter)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto x2Desc = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 4, 5, 6, 7, 8});
    auto y1Desc = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// x2远短于x1：x1={2,3,4,5,6,7,8}, x2={1}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_18_x2_much_shorter)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 4, 5, 6, 7, 8});
    auto x2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto y1Desc = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({7}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// x1长于x2：x1={2,3,4,1}, x2={1,2}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_19_x1_longer_than_x2)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 4, 1});
    auto x2Desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 2});
    auto y1Desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 两个都为单元素且相同：x1={1}, x2={1}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_20_both_single_one)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto x2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1});
    auto y1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// ==================== 泛化用例：int64覆盖 ====================

// int64单维广播
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_21_int64_single_broadcast)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1});
    auto x2Desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{5});
    auto y1Desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// int64交替模式
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_22_int64_alternating)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1, 2, 1, 2});
    auto x2Desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 1, 2, 1});
    auto y1Desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// int64大维度值
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_23_int64_large_dims)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1000000, 1, 2000000, 1, 3000000});
    auto x2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{1000000, 5000000, 1, 6000000, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// int64大rank
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_24_int64_large_rank)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 1, 3, 1, 4, 1, 5, 1});
    auto x2Desc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{2, 9, 3, 9, 4, 9, 5, 9});
    auto y1Desc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// ==================== 泛化用例：空tensor边缘场景 ====================

// x1空、x2非空：x1={0}, x2={5}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_25_x1_empty_x2_not)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{5});
    auto y1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// x1非空、x2空：x1={5}, x2={0}
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_26_x1_not_x2_empty)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{5});
    auto x2Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto y1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// ==================== 泛化用例：输出shape变化 ====================

// 输出shape恰好为max(x1,x2)
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_27_output_exact_size)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1});
    auto y1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 输出shape大于max(x1,x2)
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_28_output_oversized)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1});
    auto y1Desc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 输出shape为1（容量不足，y1/y2长度小于max(x1_len, x2_len)）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_29_output_shape_one)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1});
    auto y1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 泛化用例：非连续输入 ====================

// x1非连续（stride=2）
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_30_non_contiguous_x1)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND, {2}, 0, {6}).Value(vector<int32_t>{2, 1, 4});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1});
    auto y1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// x1和x2均非连续
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_31_non_contiguous_both)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND, {2}, 0, {6}).Value(vector<int32_t>{2, 1, 4});
    auto x2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND, {2}, 0, {6}).Value(vector<int32_t>{2, 3, 1});
    auto y1Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// ==================== 泛化用例：平台检查 ====================

// 非Regbase平台（Ascend910B），应返回ACLNN_ERR_PARAM_INVALID
TEST_F(l2_broadcast_gradient_args_test, Ascend910B_aclnnBroadcastGradientArgs_32_non_regbase_platform)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 泛化用例：额外错误场景 ====================

// 两个输入均为2D
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_33_both_2d)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6, 0});
    auto x2Desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1, 0});
    auto y1Desc = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x2为2D
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_34_x2_2d)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1, 0});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 3D输入
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_35_3d_input)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({2, 2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6, 0, 3, 2});
    auto x2Desc = TensorDesc({2, 2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1, 0, 4, 2});
    auto y1Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 0D（标量）输入
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_36_scalar_0d_input)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);
    auto y1Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x2为空指针
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_37_x2_nullptr)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, nullptr), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// y2为空指针
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_38_y2_nullptr)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, nullptr));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 全部为空指针
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_39_all_nullptr)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});

    auto ut_x1 = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(nullptr, x1Desc), OUTPUT(nullptr, nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_x1.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 不支持的dtype: int16
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_40_unsupport_int16)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto x2Desc = TensorDesc({5}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto y1Desc = TensorDesc({5}, ACL_INT16, ACL_FORMAT_ND);
    auto y2Desc = TensorDesc({5}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不支持的dtype: uint8
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_41_unsupport_uint8)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x2Desc = TensorDesc({5}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 4);
    auto y1Desc = TensorDesc({5}, ACL_UINT8, ACL_FORMAT_ND);
    auto y2Desc = TensorDesc({5}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y1单独dtype不匹配
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_42_y1_only_dtype_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y2单独dtype不匹配
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_43_y2_only_dtype_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ==================== 泛化用例：workspace验证 ====================

TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_44_workspace_nonzero)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 1, 4, 1, 6});
    auto x2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{2, 3, 1, 5, 1});
    auto y1Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
}

// 双空输入时workspace应为0
TEST_F(l2_broadcast_gradient_args_test, Ascend950_aclnnBroadcastGradientArgs_45_workspace_zero_empty)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto x1Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto x2Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    auto y1Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto y2Desc = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnBroadcastGradientArgs, INPUT(x1Desc, x2Desc), OUTPUT(y1Desc, y2Desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ExpectArchCond(aclRet, ACL_SUCCESS);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        EXPECT_EQ(workspace_size, 0UL);
    }
}
