/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_cla_gate_quant.cpp
 * \brief ClaGateQuant aclnn (l2) UT.
 *
 * Exercises aclnnClaGateQuantGetWorkspaceSize end to end on the host:
 *   - happy paths: dual/single axis x FP8(E5M2/E4M3FN) x FP4(E2M1/E1M2),
 *     both scale_alg values, all round modes, nullptr optional strings.
 *   - every validation branch: null pointers, dtype mismatch, rank/N/D
 *     constraints, gate shape mismatch, round_mode / layout / scale_alg
 *     constraints, FP8-only-rint, FP4-only-alg0 and output shape mismatch.
 */

#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_cla_gate_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

namespace {
// T=4, N=16, D=128  ->  K = N*D = 2048
constexpr int64_t T = 4;
constexpr int64_t N = 16;
constexpr int64_t D = 128;
constexpr int64_t K = N * D;          // 2048
constexpr int64_t ROW_SCALE_NUM = 32; // ceil(K / 64)
constexpr int64_t COL_SCALE_NUM = 1;  // ceil(T / 64)
constexpr int64_t SCALE_LAST_DIM = 2;

constexpr char ROUND_RINT[] = "rint";
constexpr char ROUND_FLOOR[] = "floor";
constexpr char ROUND_ROUND[] = "round";
constexpr char ROUND_INVALID[] = "xxx";
constexpr char LAYOUT_TND[] = "TND";
constexpr char LAYOUT_INVALID[] = "BSND";
} // namespace

class l2_cla_gate_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_cla_gate_quant_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_cla_gate_quant_test TearDown" << endl; }
};

// --------------------------------------------------------------------------- //
// happy paths
// --------------------------------------------------------------------------- //
TEST_F(l2_cla_gate_quant_test, ascend950_dual_fp16_to_e4m3fn_alg1_rint)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_cla_gate_quant_test, ascend950_dual_bf16_to_e5m2_alg0_rint)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_BF16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_BF16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(0),
                              static_cast<int64_t>(35), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_cla_gate_quant_test, ascend950_dual_fp16_to_e2m1_alg0_rint)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(0),
                              static_cast<int64_t>(40), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_cla_gate_quant_test, ascend950_dual_bf16_to_e1m2_alg0_floor)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_BF16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_BF16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT4_E1M2, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_FLOOR,
                              static_cast<int64_t>(0), static_cast<int64_t>(41), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_cla_gate_quant_test, ascend950_single_fp16_to_e4m3fn_alg1_rint_nullptr_col)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, false),
                        OUTPUT(rowDataDesc, rowScaleDesc, (aclTensor*)nullptr, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_cla_gate_quant_test, ascend950_single_fp16_to_e2m1_alg0_round_nullptr_col)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_ROUND,
                              static_cast<int64_t>(0), static_cast<int64_t>(40), LAYOUT_TND, false),
                        OUTPUT(rowDataDesc, rowScaleDesc, (aclTensor*)nullptr, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// nullptr round_mode / inputAttnLayout are explicitly allowed by the checks and
// defaulted to "rint" / "TND" inside the l0 op.
TEST_F(l2_cla_gate_quant_test, ascend950_dual_nullptr_optional_strings)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnClaGateQuant,
        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, static_cast<const char*>(nullptr),
              static_cast<int64_t>(0), static_cast<int64_t>(36), static_cast<const char*>(nullptr), true),
        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// T=1 and N=1 boundary: col_data/col_scale still resolve to [1, K] / [1, K, 2].
TEST_F(l2_cla_gate_quant_test, ascend950_single_t1_n1_boundary)
{
    auto globalDesc = TensorDesc({1, 1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({1, 1, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({1, 256}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({1, 4, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, false),
                        OUTPUT(rowDataDesc, rowScaleDesc, (aclTensor*)nullptr, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

// --------------------------------------------------------------------------- //
// null pointer branches (CheckNotNull)
// --------------------------------------------------------------------------- //
TEST_F(l2_cla_gate_quant_test, ascend950_error_null_global_attn)
{
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT((aclTensor*)nullptr, localDesc, globalGateDesc, localGateDesc, ROUND_RINT,
                              static_cast<int64_t>(1), static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_null_local_attn)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, (aclTensor*)nullptr, globalGateDesc, localGateDesc, ROUND_RINT,
                              static_cast<int64_t>(1), static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_null_global_gate)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, (aclTensor*)nullptr, localGateDesc, ROUND_RINT,
                              static_cast<int64_t>(1), static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_null_local_gate)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, (aclTensor*)nullptr, ROUND_RINT,
                              static_cast<int64_t>(1), static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_null_row_data)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT((aclTensor*)nullptr, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_null_row_scale)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, (aclTensor*)nullptr, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// dual_axis_flag == true requires col_data / col_scale.
TEST_F(l2_cla_gate_quant_test, ascend950_error_dual_missing_col_data)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, (aclTensor*)nullptr, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_dual_missing_col_scale)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// single-axis must NOT receive col_data / col_scale.
TEST_F(l2_cla_gate_quant_test, ascend950_error_single_with_col_data)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({1, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, false),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// --------------------------------------------------------------------------- //
// dtype branches (CheckDtypeValid)
// --------------------------------------------------------------------------- //
TEST_F(l2_cla_gate_quant_test, ascend950_error_input_fp32)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_local_attn_dtype_mismatch)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_BF16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_global_gate_dtype_mismatch)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_BF16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_local_gate_dtype_mismatch)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_BF16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_invalid_dst_type)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(99), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// --------------------------------------------------------------------------- //
// shape branches (CheckShape / CheckOutputShape)
// --------------------------------------------------------------------------- //
TEST_F(l2_cla_gate_quant_test, ascend950_error_input_rank_not_3)
{
    auto globalDesc = TensorDesc({T, K}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, K}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_n_out_of_range)
{
    auto globalDesc = TensorDesc({T, 129, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, 129, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, 129}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, 129}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_invalid_head_dim)
{
    auto globalDesc = TensorDesc({T, N, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_local_attn_shape_mismatch)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N + 1, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_gate_rank_not_2)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_gate_shape_mismatch)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N + 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N + 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_local_gate_shape_mismatch)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T + 1, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// --------------------------------------------------------------------------- //
// attribute branches
// --------------------------------------------------------------------------- //
TEST_F(l2_cla_gate_quant_test, ascend950_error_invalid_round_mode)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_INVALID,
                              static_cast<int64_t>(1), static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// FP8 output only supports round_mode == "rint".
TEST_F(l2_cla_gate_quant_test, ascend950_error_fp8_with_floor)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_FLOOR,
                              static_cast<int64_t>(0), static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_invalid_input_attn_layout)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_INVALID, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_invalid_scale_alg)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(2),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// FP4 output only supports scale_alg == 0.
TEST_F(l2_cla_gate_quant_test, ascend950_error_fp4_with_scale_alg_1)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(40), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// --------------------------------------------------------------------------- //
// output shape branches (CheckOutputShape)
// --------------------------------------------------------------------------- //
TEST_F(l2_cla_gate_quant_test, ascend950_error_row_data_shape)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K + 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_row_scale_shape)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM + 1, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_col_data_shape)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K + 1}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_col_scale_shape)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K + 1, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_null_workspace_size)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    EXPECT_EQ(ut.TestGetWorkspaceSize(nullptr), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_non_positive_t)
{
    auto globalDesc = TensorDesc({0, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({0, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({0, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({0, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({0, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({0, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({0, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({0, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_row_data_dtype)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_row_scale_dtype)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_col_data_dtype)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_quant_test, ascend950_error_col_scale_dtype)
{
    auto globalDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localDesc = TensorDesc({T, N, D}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto globalGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto localGateDesc = TensorDesc({T, N}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rowDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto rowScaleDesc = TensorDesc({T, ROW_SCALE_NUM, SCALE_LAST_DIM}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
    auto colDataDesc = TensorDesc({T, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto colScaleDesc = TensorDesc({COL_SCALE_NUM, K, SCALE_LAST_DIM}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnClaGateQuant,
                        INPUT(globalDesc, localDesc, globalGateDesc, localGateDesc, ROUND_RINT, static_cast<int64_t>(1),
                              static_cast<int64_t>(36), LAYOUT_TND, true),
                        OUTPUT(rowDataDesc, rowScaleDesc, colDataDesc, colScaleDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
