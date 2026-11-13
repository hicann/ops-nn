/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_thnn_fused_lstm_cell.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_thnn_fused_lstm_cell_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_thnn_fused_lstm_cell_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_thnn_fused_lstm_cell_test TearDown" << std::endl; }
};

struct WorkspaceSizeResult {
    aclnnStatus status = ACLNN_SUCCESS;
    bool workspaceMatched = true;
};

static aclnnStatus GetSerialWorkspaceSize(int64_t B, int64_t H, uint64_t* workspaceSize)
{
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));
    return ut.TestGetWorkspaceSize(workspaceSize);
}

static WorkspaceSizeResult RunConcurrentGetWorkspaceSize(int64_t B, int64_t H, uint64_t expectedWorkspaceSize,
                                                         std::atomic<int>& ready, std::atomic<bool>& start)
{
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    ready.fetch_add(1);
    while (!start.load()) {
        std::this_thread::yield();
    }

    WorkspaceSizeResult result;
    constexpr size_t call_num = 100;
    for (size_t idx = 0; idx < call_num; ++idx) {
        uint64_t workspace_size = 0;
        result.status = ut.TestGetWorkspaceSize(&workspace_size);
        if (result.status != ACL_SUCCESS) {
            return result;
        }
        result.workspaceMatched = result.workspaceMatched && workspace_size == expectedWorkspaceSize;
    }
    return result;
}

// 正常input场景
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_normal_float)
{
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空inputGates+其余张量一致: 合法空batch(B=0),首段校验通过后空早退直接成功
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_empty_input_gates_consistent)
{
    int64_t B = 0;
    int64_t H = 5;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空inputGates+cx非空且shape关系不满足: 空早退不得旁路shape校验,须报161002
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_empty_input_gates_shape_mismatch)
{
    int64_t H = 5;
    vector<int64_t> gates_shape = {0, 4 * H}; // inputGates空
    vector<int64_t> common_shape = {3, H};    // cx非空,与空gates的B不一致
    vector<int64_t> bias_shape = {4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空inputGates+hiddenGates维度错误: 空早退不得旁路dim校验,须报161002
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_empty_input_gates_dim_mismatch)
{
    int64_t H = 5;
    vector<int64_t> gates_shape = {0, 4 * H}; // inputGates空
    vector<int64_t> wrong_shape = {4 * H};    // hiddenGates一维,rank!=2
    vector<int64_t> common_shape = {0, H};
    vector<int64_t> bias_shape = {4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(wrong_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空inputGates dtype与其余张量不一致: 空早退不得旁路dtype校验,须报161002
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_empty_input_gates_dtype_mismatch)
{
    int64_t H = 5;
    vector<int64_t> gates_shape = {0, 4 * H};
    vector<int64_t> common_shape = {0, H};
    vector<int64_t> bias_shape = {4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT16, ACL_FORMAT_ND); // 空且fp16
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 全量不支持dtype(int32): CheckDtypes NOT_SUPPORT(161002)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_all_int32_dtype_not_support)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_INT32, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_INT32, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_INT32, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_INT32, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_INT32, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_INT32, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_INT32, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非空cx dtype与基准inputGates不一致: CheckDtypes NOT_MATCH(161002)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_cx_dtype_mismatch)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非空输出hyOut dtype与基准inputGates不一致: CheckDtypes NOT_MATCH(161002)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_output_dtype_mismatch)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输出hyOut shape与期望(B,H)不匹配: CheckShapes(161002)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_output_shape_mismatch)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> wrong_shape = {999, 999};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(wrong_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非空hiddenGates shape与期望(B,4H)不匹配: CheckShapes(161002)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_hidden_gates_shape_mismatch)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> wrong_gates_shape = {2, 8};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(wrong_gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非空inputBias shape与期望(4H,)不匹配: CheckShapes(161002)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_input_bias_shape_mismatch)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> wrong_bias_shape = {8};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(wrong_bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 输出空指针(hyOut/storageOut为nullptr): CheckParamsNullptr(161001)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_output_nullptr)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspace_size = 0;
    auto ut_hy = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                           OUTPUT(nullptr, cy, storage));
    aclnnStatus aclRet = ut_hy.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_storage = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                                OUTPUT(hy, cy, nullptr));
    aclRet = ut_storage.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// bias不成对(inputBias有效+hiddenBias为nullptr,或反之): CheckParamsNullptr(161001)
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_bias_unpaired)
{
    int64_t B = 2;
    int64_t H = 4;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspace_size = 0;
    auto ut_l = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, nullptr),
                          OUTPUT(hy, cy, storage));
    aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_r = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, nullptr, hiddenBias),
                          OUTPUT(hy, cy, storage));
    aclRet = ut_r.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_concurrent_get_workspace_size)
{
    constexpr size_t thread_num = 8;
    array<WorkspaceSizeResult, thread_num> results{};
    array<uint64_t, thread_num> expectedWorkspaceSizes{};
    atomic<int> ready{0};
    atomic<bool> start{false};
    array<thread, thread_num> threads;

    for (size_t idx = 0; idx < thread_num; ++idx) {
        EXPECT_EQ(GetSerialWorkspaceSize(3 + idx, 5 + 2 * idx, &expectedWorkspaceSizes[idx]), ACL_SUCCESS);
    }
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = thread([&results, &expectedWorkspaceSizes, &ready, &start, idx]() {
            results[idx] = RunConcurrentGetWorkspaceSize(3 + idx, 5 + 2 * idx, expectedWorkspaceSizes[idx], ready,
                                                         start);
        });
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (ready.load() != static_cast<int>(thread_num) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    bool allThreadsReady = ready.load() == static_cast<int>(thread_num);
    start.store(true);
    for (auto& thread : threads) {
        thread.join();
    }
    ASSERT_TRUE(allThreadsReady);
    for (const auto& result : results) {
        EXPECT_EQ(result.status, ACL_SUCCESS);
        EXPECT_TRUE(result.workspaceMatched);
    }
}
