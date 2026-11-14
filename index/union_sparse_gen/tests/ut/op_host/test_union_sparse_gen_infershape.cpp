/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <array>
#include <tuple>
#include <vector>
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"

namespace {
using AnyValue = Ops::NN::AnyValue;
constexpr size_t kOutputCount = 3;

struct ShapeCase {
    bool qPrefix = true;
    bool kvPrefix = true;
    int64_t maskType = 0;
    int64_t totalQ = 32;
    int64_t batch = 2;
    int64_t maxQ = 16;
    int64_t topK = 16;
    int64_t groups = 3;
    int64_t logical = 8;
};

void SetFixtureAttributes(gert::InferShapeContextFaker& faker, const ShapeCase& c)
{
    faker.NodeAttrs({{"block_shape", AnyValue::CreateFrom<std::vector<int64_t>>({1, 128})},
                     {"union_size", AnyValue::CreateFrom<int64_t>(8)},
                     {"mask_type", AnyValue::CreateFrom<int64_t>(c.maskType)},

                     {"totalq_union_num", AnyValue::CreateFrom<int64_t>(c.groups)},
                     {"max_logical_blocks", AnyValue::CreateFrom<int64_t>(c.logical)},
                     {"max_q_seqlen", AnyValue::CreateFrom<int64_t>(c.maxQ)},
                     {"max_kv_seqlen", AnyValue::CreateFrom<int64_t>(-1)},
                     {"layout_sparse_pattern", AnyValue::CreateFrom<int64_t>(4)},
                     {"is_input_idx_sorted", AnyValue::CreateFrom<bool>(true)},
                     {"residual_block_mode", AnyValue::CreateFrom<int64_t>(0)}});
}

ge::graphStatus Infer(const ShapeCase& c, std::array<gert::Shape, kOutputCount>& outputs)
{
    gert::Shape indices{2, c.totalQ, c.topK};
    gert::Shape counts{2, c.totalQ};
    gert::Shape prefix{c.batch + 1};
    gert::Shape used{c.batch};
    std::vector<gert::Shape*> inputs{&indices, &counts};
    std::vector<ge::DataType> types{ge::DT_INT32, ge::DT_INT32};
    for (bool present : {c.qPrefix, c.kvPrefix}) {
        if (present) {
            inputs.push_back(&prefix);
            types.push_back(ge::DT_INT64);
        }
    }
    inputs.insert(inputs.end(), {&used, &used});
    types.insert(types.end(), {ge::DT_INT32, ge::DT_INT32});
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(inputs.size(), kOutputCount)
        .IrInstanceNum({1, 1, uint32_t(c.qPrefix), uint32_t(c.kvPrefix), 1, 1}, {1, 1, 1})
        .InputShapes(inputs)
        .OutputShapes({&outputs[0], &outputs[1], &outputs[2]});
    SetFixtureAttributes(faker, c);
    for (size_t i = 0; i < types.size(); ++i) {
        faker.NodeInputTd(i, types[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < kOutputCount; ++i) {
        faker.NodeOutputTd(i, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnionSparseGen");
    if (impl == nullptr || impl->infer_shape == nullptr) {
        ADD_FAILURE() << "UnionSparseGen shape inference is not registered";
        return ge::GRAPH_FAILED;
    }
    auto* context = holder.GetContext<gert::InferShapeContext>();
    const auto status = impl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] = *context->GetOutputShape(i);
        }
    }
    return status;
}

class UnionSparseGenLengthShape : public testing::TestWithParam<std::tuple<bool, bool, int64_t>> {};

TEST_P(UnionSparseGenLengthShape, IndependentPrefixModes)
{
    ShapeCase c;
    std::tie(c.qPrefix, c.kvPrefix, c.maskType) = GetParam();
    std::array<gert::Shape, kOutputCount> outputs;
    ASSERT_EQ(Infer(c, outputs), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputs[0], (gert::Shape{2, 3, 8}));
    EXPECT_EQ(outputs[1], (gert::Shape{2, 3}));
    EXPECT_EQ(outputs[2], (gert::Shape{2, 3, 1024}));
}

INSTANTIATE_TEST_SUITE_P(UnionSparseGen, UnionSparseGenLengthShape,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Values(int64_t{0}, int64_t{1})));

TEST(UnionSparseGenInferShape, FixedSlotsRequirePositiveWidth)
{
    ShapeCase c;
    c.qPrefix = false;
    c.maxQ = -1;
    std::array<gert::Shape, kOutputCount> outputs;
    EXPECT_EQ(Infer(c, outputs), ge::GRAPH_FAILED);
}

TEST(UnionSparseGenInferShape, DerivesCapacityFromInputAndLogicalDomain)
{
    for (int64_t logical : {1, 8, 127, 128, 129, 2048}) {
        ShapeCase c;
        c.logical = logical;
        std::array<gert::Shape, kOutputCount> outputs;
        ASSERT_EQ(Infer(c, outputs), ge::GRAPH_SUCCESS);
        const int64_t capacity = logical < 128 ? logical : 128;
        EXPECT_EQ(outputs[0], (gert::Shape{2, 3, capacity}));
        EXPECT_EQ(outputs[2], (gert::Shape{2, 3, capacity * 128}));
    }
}

TEST(UnionSparseGenInferShape, RejectsInvalidLogicalDomain)
{
    ShapeCase c;
    c.logical = 0;
    std::array<gert::Shape, kOutputCount> outputs;
    EXPECT_EQ(Infer(c, outputs), ge::GRAPH_FAILED);
}

TEST(UnionSparseGenInferShape, EmptyFixedBatch)
{
    ShapeCase c;
    c.qPrefix = false;
    c.kvPrefix = false;
    c.totalQ = 0;
    c.batch = 0;
    c.groups = 0;
    c.logical = 1;
    std::array<gert::Shape, kOutputCount> outputs;
    ASSERT_EQ(Infer(c, outputs), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outputs[0], (gert::Shape{2, 0, 1}));
    EXPECT_EQ(outputs[2], (gert::Shape{2, 0, 128}));
}

TEST(UnionSparseGenInferShape, Int64PrefixesKeepInt32Outputs)
{
    std::array<ge::DataType, 6> inputs{ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
                                       ge::DT_INT64, ge::DT_INT32, ge::DT_INT32};
    std::array<ge::DataType, 3> outputs{ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED};
    gert::InferDataTypeContextFaker faker;
    faker.NodeIoNum(6, 3)
        .InputDataTypes({&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5]})
        .OutputDataTypes({&outputs[0], &outputs[1], &outputs[2]});
    for (int32_t i = 0; i < 6; ++i) {
        faker.NodeInputTd(i, inputs[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < kOutputCount; ++i) {
        faker.NodeOutputTd(i, outputs[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("UnionSparseGen");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    for (size_t i = 0; i < outputs.size(); ++i) {
        EXPECT_EQ(context->GetOutputDataType(i), ge::DT_INT32);
    }
}
} // namespace
