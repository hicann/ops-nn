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
#include <initializer_list>
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "register/op_impl_registry.h"
#include "../../../op_host/union_sparse_gen_geometry.h"

namespace {
namespace Geometry = UnionSparseGenGeometry;
namespace Contract = UnionSparseGenContract;
struct LengthContext {
    std::array<const gert::Tensor*, Contract::kInputCount> tensors{};
    mutable size_t reads = 0;
    const gert::Shape* GetOptionalInputShape(size_t i) const
    {
        return tensors[i] == nullptr ? nullptr : &tensors[i]->GetStorageShape();
    }
    const gert::Tensor* GetOptionalInputTensor(size_t i) const
    {
        ++reads;
        return tensors[i];
    }
};

template <typename T>
gert::Tensor LengthTensor(T* values, int64_t size, ge::DataType type)
{
    return gert::Tensor(gert::StorageShape({size}, {size}), gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
                        gert::kOnHost, type, values);
}

TEST(UnionSparseGenGeometry, ExplicitHintsDoNotReadValues)
{
    LengthContext context;
    Geometry::Result result{3, 512};
    EXPECT_TRUE(Geometry::Resolve(&context, 2, 32, 16, 8, 128, result));
    EXPECT_EQ(context.reads, 0);
    EXPECT_EQ(result.groups, 3);
    EXPECT_EQ(result.logical, 512);
}

TEST(UnionSparseGenGeometry, IndependentPrefixesAndPartialHints)
{
    int64_t q[] = {0, 16, 32}, kv[] = {0, 384, 768};
    int32_t uq[] = {3, 9}, ukv[] = {259, 384};
    auto qt = LengthTensor(q, 3, ge::DT_INT64), kvt = LengthTensor(kv, 3, ge::DT_INT64);
    auto uqt = LengthTensor(uq, 2, ge::DT_INT32), ukvt = LengthTensor(ukv, 2, ge::DT_INT32);
    for (unsigned flags = 0; flags < 4; ++flags) {
        LengthContext context;
        context.tensors[Contract::kCuQInput] = flags & 1U ? &qt : nullptr;
        context.tensors[Contract::kCuKvInput] = flags & 2U ? &kvt : nullptr;
        context.tensors[Contract::kUsedQInput] = &uqt;
        context.tensors[Contract::kUsedKvInput] = &ukvt;
        for (int64_t groups : {-1, 3}) {
            for (int64_t logical : {-1, 3}) {
                Geometry::Result result{groups, logical};
                ASSERT_TRUE(Geometry::Resolve(&context, 2, 32, 16, 8, 128, result));
                EXPECT_EQ(result.groups, 3);
                EXPECT_EQ(result.logical, 3);
            }
        }
    }
}

TEST(UnionSparseGenGeometry, SameShapeValueUpdateRecomputes)
{
    int32_t q[] = {3, 9}, kv[] = {259, 384};
    auto qt = LengthTensor(q, 2, ge::DT_INT32), kvt = LengthTensor(kv, 2, ge::DT_INT32);
    LengthContext context;
    context.tensors[Contract::kUsedQInput] = &qt;
    context.tensors[Contract::kUsedKvInput] = &kvt;
    Geometry::Result result;
    ASSERT_TRUE(Geometry::Resolve(&context, 2, 32, 16, 8, 128, result));
    q[0] = 16;
    q[1] = 16;
    kv[0] = 262144;
    result = {};
    ASSERT_TRUE(Geometry::Resolve(&context, 2, 32, 16, 8, 128, result));
    EXPECT_EQ(result.groups, 4);
    EXPECT_EQ(result.logical, 2048);
}

TEST(UnionSparseGenGeometry, UnavailableUsedLengthDoesNotFallBack)
{
    int64_t prefix[] = {0, 16, 32};
    auto qt = LengthTensor(prefix, 3, ge::DT_INT64);
    auto used = LengthTensor<int32_t>(nullptr, 2, ge::DT_INT32);
    LengthContext context;
    context.tensors[Contract::kCuQInput] = &qt;
    context.tensors[Contract::kUsedQInput] = &used;
    Geometry::Result result{-1, 1};
    ASSERT_TRUE(Geometry::Resolve(&context, 2, 32, -1, 8, 128, result));
    EXPECT_EQ(result.groups, -1);
}

TEST(UnionSparseGenGeometry, RejectsInvalidLengthsAndLogicalOverflow)
{
    int64_t prefix[] = {0, 16, 32};
    int32_t used[] = {17, 9};
    auto qt = LengthTensor(prefix, 3, ge::DT_INT64), uqt = LengthTensor(used, 2, ge::DT_INT32);
    LengthContext context;
    context.tensors[Contract::kCuQInput] = &qt;
    context.tensors[Contract::kUsedQInput] = &uqt;
    Geometry::Result result{-1, 1};
    EXPECT_FALSE(Geometry::Resolve(&context, 2, 32, -1, 8, 128, result));
    used[0] = 3;
    prefix[1] = -1;
    EXPECT_FALSE(Geometry::Resolve(&context, 2, 32, -1, 8, 128, result));
    int64_t kv[] = {0, std::numeric_limits<int64_t>::max()};
    auto kvt = LengthTensor(kv, 2, ge::DT_INT64);
    context.tensors[Contract::kCuKvInput] = &kvt;
    result = {1, -1};
    EXPECT_FALSE(Geometry::Resolve(&context, 1, 8, -1, 8, 128, result));
}

TEST(UnionSparseGenGeometry, EmptyBatchAndInvalidHints)
{
    LengthContext context;
    Geometry::Result result;
    ASSERT_TRUE(Geometry::Resolve(&context, 0, 0, 16, 8, 128, result));
    EXPECT_EQ(result.groups, 0);
    EXPECT_EQ(result.logical, 1);
    EXPECT_EQ(context.reads, 0);
    for (auto hints : {Geometry::Result{-2, 1}, Geometry::Result{0, 0}, Geometry::Result{0, -2}}) {
        EXPECT_FALSE(Geometry::Resolve(&context, 0, 0, 16, 8, 128, hints));
    }
}

struct AutoShapeFixture {
    static constexpr int64_t kBatchSize = 2, kHeads = 2, kTopK = 2;
    static constexpr int64_t kQSlotRows = 16, kStoredQ = kBatchSize * kQSlotRows, kKvSlotTokens = 384;
    static constexpr int64_t kExpectedGroups = 3, kExpectedLogicalBlocks = 3;
    static constexpr unsigned kQPrefix = 1U, kKvPrefix = kQPrefix << 1, kPrefixCombinations = kKvPrefix << 1;
    static constexpr unsigned kGroupsHint = 1U, kLogicalHint = kGroupsHint << 1, kHintCombinations = kLogicalHint << 1;
    static constexpr std::array<int64_t, kBatchSize + 1> kQueryPrefix{0, kQSlotRows, kStoredQ};
    static constexpr std::array<int64_t, kBatchSize + 1> kKvPrefixValues{0, kKvSlotTokens, kBatchSize* kKvSlotTokens};
    static constexpr std::array<int32_t, kBatchSize> kUsedQ{3, 9}, kUsedKv{259, kKvSlotTokens};
};

ge::graphStatus InferAuto(unsigned prefixes, unsigned hints, bool valuesAvailable,
                          std::array<gert::Shape, Contract::kOutputCount>& outputs)
{
    using Fixture = AutoShapeFixture;
    auto q = Fixture::kQueryPrefix, kv = Fixture::kKvPrefixValues;
    auto uq = Fixture::kUsedQ, ukv = Fixture::kUsedKv;
    const bool hasQPrefix = (prefixes & Fixture::kQPrefix) != 0;
    const bool hasKvPrefix = (prefixes & Fixture::kKvPrefix) != 0;
    const gert::StorageFormat format(ge::FORMAT_ND, ge::FORMAT_ND, {});
    const std::initializer_list<int64_t> idxShape{Fixture::kHeads, Fixture::kStoredQ, Fixture::kTopK};
    const std::initializer_list<int64_t> countShape{Fixture::kHeads, Fixture::kStoredQ};
    gert::Tensor idx(gert::StorageShape(idxShape, idxShape), format, ge::DT_INT32);
    gert::Tensor count(gert::StorageShape(countShape, countShape), format, ge::DT_INT32);
    auto qt = LengthTensor(valuesAvailable ? q.data() : nullptr, q.size(), ge::DT_INT64);
    auto kvt = LengthTensor(valuesAvailable ? kv.data() : nullptr, kv.size(), ge::DT_INT64);
    auto uqt = LengthTensor(valuesAvailable ? uq.data() : nullptr, uq.size(), ge::DT_INT32);
    auto ukvt = LengthTensor(valuesAvailable ? ukv.data() : nullptr, ukv.size(), ge::DT_INT32);
    std::vector<gert::Tensor*> inputs{&idx, &count};
    if (hasQPrefix)
        inputs.push_back(&qt);
    if (hasKvPrefix)
        inputs.push_back(&kvt);
    inputs.push_back(&uqt);
    inputs.push_back(&ukvt);
    gert::OpInferShapeContextBuilder builder;
    builder.OpType("UnionSparseGen")
        .OpName("UnionSparseGen")
        .IOInstanceNum({1, 1, uint32_t(hasQPrefix), uint32_t(hasKvPrefix), 1, 1}, {1, 1, 1})
        .InputTensors(inputs)
        .AppendAttr(std::vector<int64_t>{Contract::kSupportedBlockX, Contract::kSupportedBlockY})
        .AppendAttr(Contract::kSupportedUnionSize)
        .AppendAttr(Contract::kDefaultMaskType)
        .AppendAttr(hints & Fixture::kGroupsHint ? Fixture::kExpectedGroups : Contract::kUnspecifiedGeometry)
        .AppendAttr(hints & Fixture::kLogicalHint ? Fixture::kExpectedLogicalBlocks : Contract::kUnspecifiedGeometry)
        .AppendAttr(Fixture::kQSlotRows)
        .AppendAttr(Contract::kUnspecifiedMaxSeqLen)
        .AppendAttr(Contract::kSparsePatternLayout)
        .AppendAttr(true)
        .AppendAttr(Contract::kExplicitBlocks);
    for (size_t i = 0; i < outputs.size(); ++i) {
        builder.OutputTensorDesc(i, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    const auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto* impl = spaceRegistry->GetOpImpl("UnionSparseGen");
    if (impl == nullptr || impl->infer_shape == nullptr)
        return ge::GRAPH_FAILED;
    auto* context = holder.GetContext();
    const auto status = impl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        for (size_t i = 0; i < outputs.size(); ++i)
            outputs[i] = *context->GetOutputShape(i);
    }
    return status;
}

TEST(UnionSparseGenGeometry, ActualInferShapeResolvesAllHintAndPrefixCombinations)
{
    using Fixture = AutoShapeFixture;
    for (unsigned prefixes = 0; prefixes < Fixture::kPrefixCombinations; ++prefixes) {
        for (unsigned hints = 0; hints < Fixture::kHintCombinations; ++hints) {
            std::array<gert::Shape, Contract::kOutputCount> outputs;
            ASSERT_EQ(InferAuto(prefixes, hints, true, outputs), ge::GRAPH_SUCCESS);
            EXPECT_EQ(outputs[Contract::kUnionIdxOutput],
                      (gert::Shape{Fixture::kHeads, Fixture::kExpectedGroups, Fixture::kExpectedLogicalBlocks}));
            EXPECT_EQ(outputs[Contract::kUnionCountOutput], (gert::Shape{Fixture::kHeads, Fixture::kExpectedGroups}));
            EXPECT_EQ(outputs[Contract::kExactMaskOutput],
                      (gert::Shape{Fixture::kHeads, Fixture::kExpectedGroups,
                                   Fixture::kExpectedLogicalBlocks * Contract::kSupportedBlockY}));
        }
    }
}

TEST(UnionSparseGenGeometry, UnknownValuesPreserveRankAndOnlyUnknownDimensions)
{
    using Fixture = AutoShapeFixture;
    for (unsigned hints = 0; hints < Fixture::kHintCombinations; ++hints) {
        std::array<gert::Shape, Contract::kOutputCount> outputs;
        ASSERT_EQ(InferAuto(Fixture::kQPrefix | Fixture::kKvPrefix, hints, false, outputs), ge::GRAPH_SUCCESS);
        const int64_t groups = hints & Fixture::kGroupsHint ? Fixture::kExpectedGroups : Contract::kUnspecifiedGeometry;
        const int64_t capacity = hints & Fixture::kLogicalHint ? Fixture::kExpectedLogicalBlocks :
                                                                 Contract::kUnspecifiedGeometry;
        EXPECT_EQ(outputs[Contract::kUnionIdxOutput], (gert::Shape{Fixture::kHeads, groups, capacity}));
        EXPECT_EQ(outputs[Contract::kUnionCountOutput], (gert::Shape{Fixture::kHeads, groups}));
        EXPECT_EQ(outputs[Contract::kExactMaskOutput],
                  (gert::Shape{Fixture::kHeads, groups,
                               capacity < 0 ? Contract::kUnspecifiedGeometry : capacity * Contract::kSupportedBlockY}));
    }
}
} // namespace
