/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "base/registry/op_impl_space_registry_v2.h"
#include "../../../op_graph/dynamic_augru_proto.h"
#include "../../../op_graph/dynamic_augru_proto.h"

namespace ops {
ge::graphStatus InferShapeAndType4DynamicAUGRU(ge::Operator& op);
}
namespace {
using DynamicAUGRUGraphOp = ge::op::DynamicAUGRU;

void SetGraphInput(ge::Operator& op, const char* name, std::vector<int64_t> dims, ge::DataType type = ge::DT_FLOAT16)
{
    ASSERT_EQ(op.UpdateInputDesc(name, ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, type)), ge::GRAPH_SUCCESS);
}

} // namespace

TEST(DynamicAUGRUGraphInfershape, unknown_rank_with_missing_optional_inputs)
{
    DynamicAUGRUGraphOp op;
    for (const char* name : {"x", "weight_input", "weight_hidden", "weight_att"}) {
        SetGraphInput(op, name, {-2});
    }
    ASSERT_EQ(ops::InferShapeAndType4DynamicAUGRU(op), ge::GRAPH_SUCCESS);
    for (uint32_t i = 0; i < 7; ++i) {
        EXPECT_EQ(op.GetOutputDesc(i).GetShape().GetDims(), (std::vector<int64_t>{-1, -1, -1}));
        EXPECT_EQ(op.GetOutputDesc(i).GetDataType(), ge::DT_FLOAT16);
    }
}

TEST(DynamicAUGRUGraphInfershape, merges_optional_dimensions_and_state_dtype)
{
    DynamicAUGRUGraphOp op;
    for (const char* name : {"x", "weight_input", "weight_hidden", "weight_att"}) {
        SetGraphInput(op, name, {-2});
    }
    SetGraphInput(op, "bias_hidden", {21}, ge::DT_FLOAT);
    SetGraphInput(op, "seq_length", {9, 4, 7});
    ASSERT_EQ(ops::InferShapeAndType4DynamicAUGRU(op), ge::GRAPH_SUCCESS);
    for (uint32_t i = 0; i < 7; ++i) {
        EXPECT_EQ(op.GetOutputDesc(i).GetShape().GetDims(), (std::vector<int64_t>{9, 4, 7}));
        EXPECT_EQ(op.GetOutputDesc(i).GetDataType(), ge::DT_FLOAT);
    }
    SetGraphInput(op, "x", {-1, 5, -1});
    EXPECT_EQ(ops::InferShapeAndType4DynamicAUGRU(op), ge::GRAPH_FAILED);
}

TEST(DynamicAUGRUGraphInfershape, rejects_invalid_known_rank)
{
    DynamicAUGRUGraphOp op;
    SetGraphInput(op, "x", {2, 5});
    SetGraphInput(op, "weight_input", {5, 21});
    SetGraphInput(op, "weight_hidden", {7, 21});
    SetGraphInput(op, "weight_att", {-1, 2});
    EXPECT_EQ(ops::InferShapeAndType4DynamicAUGRU(op), ge::GRAPH_FAILED);
}

TEST(DynamicAUGRUGraphInfershape, registers_graph_datatype_callback)
{
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    gert::OppSoDesc graphSo({ge::AscendString(DYNAMIC_AUGRU_GRAPH_REGISTRY_SO)}, "dynamic_augru_graph_ut");
    ASSERT_EQ(registry->AddSoToRegistry(graphSo), ge::GRAPH_SUCCESS);
    const auto* impl = registry->GetOpImpl("DynamicAUGRU");
    ASSERT_NE(impl, nullptr);
    EXPECT_NE(impl->infer_datatype, nullptr);
}
