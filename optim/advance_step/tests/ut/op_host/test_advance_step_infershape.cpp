/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/advance_step_proto.h"

namespace {
constexpr int64_t NUM_SEQS = 8;
constexpr int64_t NUM_QUERIES = 4;
constexpr int64_t BLOCK_SIZE = 2;

// 按原型声明的真实输入名逐个设置输入描述，可选输入一并给出，
// 确保推导回调拿到的是 AdvanceStep 注册输入而非空描述
void SetAdvanceStepInputs(ge::op::AdvanceStep& op, std::initializer_list<int64_t> stateShape,
                          std::initializer_list<int64_t> sampledShape, std::initializer_list<int64_t> blockTablesShape)
{
    op.UpdateInputDesc("input_tokens", create_desc(stateShape, ge::DT_INT64));
    op.UpdateInputDesc("sampled_token_ids", create_desc(sampledShape, ge::DT_INT64));
    op.UpdateInputDesc("input_positions", create_desc(stateShape, ge::DT_INT64));
    op.UpdateInputDesc("seq_lens", create_desc(stateShape, ge::DT_INT64));
    op.UpdateInputDesc("slot_mapping", create_desc(stateShape, ge::DT_INT64));
    op.UpdateInputDesc("block_tables", create_desc(blockTablesShape, ge::DT_INT64));
    op.UpdateInputDesc("spec_token", create_desc({NUM_SEQS, 2}, ge::DT_INT64));
    op.UpdateInputDesc("accepted_num", create_desc({NUM_SEQS}, ge::DT_INT64));
}

void ExpectOutputsShapeEquals(const ge::op::AdvanceStep& op, const std::vector<int64_t>& expected)
{
    for (size_t i = 0; i < 4; i++) {
        const auto dims = op.GetOutputDesc(i).GetShape().GetDims();
        if (dims.size() != expected.size()) {
            EXPECT_EQ(dims.size(), expected.size());
            return;
        }
        for (size_t d = 0; d < expected.size(); d++) {
            EXPECT_EQ(dims[d], expected[d]);
        }
    }
}
} // namespace

class AdvanceStep : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdvanceStep SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdvanceStep TearDown" << std::endl; }
};

// 4 个 inplace 状态张量输出与 input_tokens 同 shape：1-D 用例
// （InferDataType 已拆分至 op_graph，dtype 推导见 test_advance_step_infer_pass.cpp）
TEST_F(AdvanceStep, AdvanceStep_infershape_case_0)
{
    ge::op::AdvanceStep op;
    SetAdvanceStepInputs(op, {NUM_SEQS}, {NUM_QUERIES, 1}, {NUM_SEQS, 4});
    op.SetAttr("num_seqs", NUM_SEQS);
    op.SetAttr("num_queries", NUM_QUERIES);
    op.SetAttr("block_size", BLOCK_SIZE);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    ExpectOutputsShapeEquals(op, {NUM_SEQS});
}

// 4 个 inplace 状态张量输出与 input_tokens 同 shape：不同长度的 1-D 用例
TEST_F(AdvanceStep, AdvanceStep_infershape_case_1)
{
    ge::op::AdvanceStep op;
    SetAdvanceStepInputs(op, {16}, {4, 1}, {16, 4});
    op.SetAttr("num_seqs", 16);
    op.SetAttr("num_queries", 4);
    op.SetAttr("block_size", 2);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    ExpectOutputsShapeEquals(op, {16});
}

// Host 形状门负例：0-D（标量）状态张量按 rank 0 拒绝
TEST_F(AdvanceStep, AdvanceStep_infershape_reject_rank0)
{
    ge::op::AdvanceStep op;
    SetAdvanceStepInputs(op, {}, {NUM_QUERIES, 1}, {NUM_SEQS, 4});
    op.SetAttr("num_seqs", NUM_SEQS);
    op.SetAttr("num_queries", NUM_QUERIES);
    op.SetAttr("block_size", BLOCK_SIZE);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

// Host 形状门负例：状态张量第一维长度不一致
TEST_F(AdvanceStep, AdvanceStep_infershape_reject_length_mismatch)
{
    ge::op::AdvanceStep op;
    op.UpdateInputDesc("input_tokens", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("sampled_token_ids", create_desc({NUM_QUERIES, 1}, ge::DT_INT64));
    op.UpdateInputDesc("input_positions", create_desc({NUM_SEQS + 1}, ge::DT_INT64));
    op.UpdateInputDesc("seq_lens", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("slot_mapping", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("block_tables", create_desc({NUM_SEQS, 4}, ge::DT_INT64));
    op.SetAttr("num_seqs", NUM_SEQS);
    op.SetAttr("num_queries", NUM_QUERIES);
    op.SetAttr("block_size", BLOCK_SIZE);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

// Host 形状门负例：blockTables 秩不为 2
TEST_F(AdvanceStep, AdvanceStep_infershape_reject_bt_rank)
{
    ge::op::AdvanceStep op;
    SetAdvanceStepInputs(op, {NUM_SEQS}, {NUM_QUERIES, 1}, {NUM_SEQS});
    op.SetAttr("num_seqs", NUM_SEQS);
    op.SetAttr("num_queries", NUM_QUERIES);
    op.SetAttr("block_size", BLOCK_SIZE);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

// Host 形状门与 tiling 宽容度对齐：空可选张量（numel==0，任意 rank）视为未传入，
// 跳过 rank 检查回落 V1（tiling CheckOptionalShapeExisting 同口径），门必须放行
TEST_F(AdvanceStep, AdvanceStep_infershape_allows_empty_optional_any_rank)
{
    ge::op::AdvanceStep op;
    op.UpdateInputDesc("input_tokens", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("sampled_token_ids", create_desc({NUM_QUERIES, 1}, ge::DT_INT64));
    op.UpdateInputDesc("input_positions", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("seq_lens", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("slot_mapping", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("block_tables", create_desc({NUM_SEQS, 4}, ge::DT_INT64));
    // spec_token 传 1-D 空张量 (0,)：numel==0 → 视为未传入，rank 1 不应被拒
    op.UpdateInputDesc("spec_token", create_desc({0}, ge::DT_INT64));
    op.UpdateInputDesc("accepted_num", create_desc({0}, ge::DT_INT64));
    op.SetAttr("num_seqs", NUM_SEQS);
    op.SetAttr("num_queries", NUM_QUERIES);
    op.SetAttr("block_size", BLOCK_SIZE);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

// Host 形状门负例：已传入（非空）的可选 spec_token 秩不为 2
TEST_F(AdvanceStep, AdvanceStep_infershape_reject_spec_rank)
{
    ge::op::AdvanceStep op;
    op.UpdateInputDesc("input_tokens", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("sampled_token_ids", create_desc({NUM_QUERIES, 1}, ge::DT_INT64));
    op.UpdateInputDesc("input_positions", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("seq_lens", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("slot_mapping", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.UpdateInputDesc("block_tables", create_desc({NUM_SEQS, 4}, ge::DT_INT64));
    // spec_token 非空 (8,2,1) rank 3：已传入，tiling CheckOptionalInputs 会拒，门同样拒
    op.UpdateInputDesc("spec_token", create_desc({NUM_SEQS, 2, 1}, ge::DT_INT64));
    op.UpdateInputDesc("accepted_num", create_desc({NUM_SEQS}, ge::DT_INT64));
    op.SetAttr("num_seqs", NUM_SEQS);
    op.SetAttr("num_queries", NUM_QUERIES);
    op.SetAttr("block_size", BLOCK_SIZE);

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}
