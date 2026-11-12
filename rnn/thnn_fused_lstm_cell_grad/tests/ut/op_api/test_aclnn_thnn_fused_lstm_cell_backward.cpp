/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_thnn_fused_lstm_cell_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_thnn_fused_lstm_cell_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_thnn_fused_lstm_cell_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_thnn_fused_lstm_cell_backward_test TearDown" << std::endl; }
};

// 正常input场景
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_normal_float)
{
    vector<int64_t> dhy_shape = {1, 8};
    vector<int64_t> b_shape = {32};
    vector<int64_t> gates_shape = {1, 32};

    auto dhy = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(dhy, dc, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 防御性负向用例:dhy 维度不足(1维)应被参数校验拦截
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_dhy_dim_invalid)
{
    vector<int64_t> dhy_shape = {8}; // 期望2维,构造1维触发维度校验
    vector<int64_t> b_shape = {32};
    vector<int64_t> gates_shape = {1, 32};

    auto dhy = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(dhy, dc, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 可选输入gradHyOptional传nullptr(等价零梯度),不应返回161001
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_grad_hy_nullptr)
{
    auto dc = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(nullptr, dc, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 可选输入gradCOptional传nullptr(等价零梯度),不应返回161001
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_grad_c_nullptr)
{
    auto dhy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(dhy, nullptr, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 两个可选梯度均传nullptr(等价全零梯度),不应返回161001
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_both_grad_nullptr)
{
    auto cx = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(nullptr, nullptr, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gradHyOptional为空且hasBias=false: 可选梯度补零路径与bias可选路径叠加
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_grad_hy_nullptr_no_bias)
{
    auto dc = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(nullptr, dc, cx, cy, storage, false),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 两个可选梯度均为空且hasBias=false: 直出零路径不写bias输出
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_both_grad_nullptr_no_bias)
{
    auto cx = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(nullptr, nullptr, cx, cy, storage, false),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gradHyOptional为空时以cx为基准继续做shape校验: dc shape与cx不一致应报161002
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_grad_hy_nullptr_dc_shape_invalid)
{
    auto dc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND); // 与cx的[1,8]不一致
    auto cx = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc({1, 32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(nullptr, dc, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 两个可选梯度均传nullptr的FLOAT16形态: 置零快路径不受dtype影响
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_both_grad_nullptr_float16)
{
    auto cx = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto cy = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto storage = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto db_out = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(nullptr, nullptr, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 两个可选梯度均传nullptr且输出为非连续张量: 置零快路径需经ViewCopy写回带stride的出参
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_both_grad_nullptr_non_contiguous_out)
{
    auto cx = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    // view [2,8]/[2,2], stride 跨步, storage 相应放大
    auto dgates_out = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND, {16, 1}, 0, {2, 16});
    auto dc_prev_out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {4, 1}, 0, {2, 4});
    auto db_out = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnThnnFusedLstmCellBackward, INPUT(nullptr, nullptr, cx, cy, storage, true),
                        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
