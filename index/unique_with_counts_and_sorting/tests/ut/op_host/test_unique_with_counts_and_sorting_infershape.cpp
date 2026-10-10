/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shaperange_context_faker.h"
#include "op_impl_registry.h"

namespace {
constexpr size_t INPUT_COUNT = 1;
constexpr size_t OUTPUT_COUNT = 3;
constexpr size_t VALUES_OUTPUT = 0;
constexpr size_t INDICES_OUTPUT = 1;
constexpr size_t COUNTS_OUTPUT = 2;
constexpr int64_t ROWS = 2;
constexpr int64_t COLUMNS = 3;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_RANK = -2;
constexpr int64_t MIN_UNIQUE_COUNT = 1;
constexpr int64_t SHAPE_SIZE_OVERFLOW = std::numeric_limits<int64_t>::min();
constexpr const char* OP_TYPE = "UniqueWithCountsAndSorting";
using Attrs = std::vector<std::pair<std::string, Ops::NN::AnyValue>>;

Attrs MakeAttrs(bool inverse, bool counts)
{
    return {{"return_inverse", Ops::NN::AnyValue::CreateFrom<bool>(inverse)},
            {"return_counts", Ops::NN::AnyValue::CreateFrom<bool>(counts)},
            {"sorted", Ops::NN::AnyValue::CreateFrom<bool>(true)}};
}

class UniqueWithCountsAndSortingInferTest : public testing::Test {
protected:
    void SetUp() override
    {
        impl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
        ASSERT_NE(impl, nullptr);
        ASSERT_NE(impl->infer_shape, nullptr);
        ASSERT_NE(impl->infer_datatype, nullptr);
        ASSERT_NE(impl->infer_shape_range, nullptr);
    }
    const gert::OpImplKernelRegistry::OpImplFunctions* impl = nullptr;
};

TEST_F(UniqueWithCountsAndSortingInferTest, RejectScalarAndNullContexts)
{
    gert::Shape input;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(INPUT_COUNT, OUTPUT_COUNT)
                      .NodeInputTd(VALUES_OUTPUT, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&input})
                      .NodeAttrs(MakeAttrs(false, false))
                      .Build();
    EXPECT_EQ(impl->infer_shape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    EXPECT_EQ(impl->infer_shape(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(impl->infer_datatype(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(impl->infer_shape_range(nullptr), ge::GRAPH_FAILED);
}

TEST_F(UniqueWithCountsAndSortingInferTest, PreserveCanndevShapeBranches)
{
    const std::vector<gert::Shape> shapes{{ROWS, COLUMNS}, {0}, {UNKNOWN_DIM, COLUMNS}, {UNKNOWN_RANK}};
    for (const auto& shape : shapes) {
        for (bool inverse : {false, true}) {
            for (bool counts : {false, true}) {
                gert::Shape input = shape;
                gert::Shape values;
                gert::Shape indices;
                gert::Shape count;
                auto holder = gert::InferShapeContextFaker()
                                  .SetOpType(OP_TYPE)
                                  .NodeIoNum(INPUT_COUNT, OUTPUT_COUNT)
                                  .NodeInputTd(VALUES_OUTPUT, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputShapes({&input})
                                  .OutputShapes({&values, &indices, &count})
                                  .NodeAttrs(MakeAttrs(inverse, counts))
                                  .Build();
                auto* context = holder.GetContext<gert::InferShapeContext>();
                ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);
                EXPECT_EQ(*context->GetOutputShape(VALUES_OUTPUT), gert::Shape({UNKNOWN_DIM}));
                EXPECT_EQ(*context->GetOutputShape(INDICES_OUTPUT), inverse || counts ? input : gert::Shape({0}));
                const gert::Shape expectedCounts = counts  ? gert::Shape({UNKNOWN_DIM}) :
                                                   inverse ? gert::Shape() :
                                                             gert::Shape({0});
                EXPECT_EQ(*context->GetOutputShape(COUNTS_OUTPUT), expectedCounts);
            }
        }
    }
}

TEST_F(UniqueWithCountsAndSortingInferTest, PreserveDtypeAndLegacyDefault)
{
    for (auto inputType :
         {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8,
          ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64}) {
        for (auto indexType : {ge::DT_UNDEFINED, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT}) {
            auto attrs = MakeAttrs(false, false);
            if (indexType != ge::DT_UNDEFINED) {
                attrs.emplace_back("out_idx", Ops::NN::AnyValue::CreateFrom<int64_t>(indexType));
            }
            auto holder = gert::InferDataTypeContextFaker()
                              .SetOpType(OP_TYPE)
                              .NodeIoNum(INPUT_COUNT, OUTPUT_COUNT)
                              .NodeInputTd(VALUES_OUTPUT, inputType, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(VALUES_OUTPUT, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(INDICES_OUTPUT, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(COUNTS_OUTPUT, ge::DT_UNDEFINED, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs(attrs)
                              .Build();
            auto* context = holder.GetContext<gert::InferDataTypeContext>();
            ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
            EXPECT_EQ(context->GetOutputDataType(VALUES_OUTPUT), inputType);
            EXPECT_EQ(context->GetOutputDataType(INDICES_OUTPUT), ge::DT_INT64);
            EXPECT_EQ(context->GetOutputDataType(COUNTS_OUTPUT), ge::DT_INT64);
        }
    }
}

TEST_F(UniqueWithCountsAndSortingInferTest, PreserveCanndevRangeBoundaries)
{
    // Preserve legacy results even when a boundary range is unusual (for example, min=1 and max=0).
    const std::vector<std::pair<gert::Shape, int64_t>> cases{
        {{ROWS, COLUMNS}, ROWS * COLUMNS},
        {{0}, 0},
        {{UNKNOWN_DIM, COLUMNS}, UNKNOWN_DIM * COLUMNS},
        {{UNKNOWN_RANK}, UNKNOWN_RANK},
        {{std::numeric_limits<int64_t>::max(), ROWS}, SHAPE_SIZE_OVERFLOW}};
    for (const auto& testCase : cases) {
        const auto& shape = testCase.first;
        for (bool inverse : {false, true}) {
            for (bool counts : {false, true}) {
                gert::Shape min = shape;
                gert::Shape max = shape;
                gert::Range<gert::Shape> input(&min, &max);
                gert::Shape empty;
                gert::Range<gert::Shape> output(&empty, &empty);
                auto holder = gert::InferShapeRangeContextFaker()
                                  .SetOpType(OP_TYPE)
                                  .NodeIoNum(INPUT_COUNT, OUTPUT_COUNT)
                                  .NodeInputTd(VALUES_OUTPUT, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputShapeRanges({&input})
                                  .OutputShapeRanges({&output, &output, &output})
                                  .NodeAttrs(MakeAttrs(inverse, counts))
                                  .Build();
                auto* context = holder.GetContext<gert::InferShapeRangeContext>();
                ASSERT_EQ(impl->infer_shape_range(context), ge::GRAPH_SUCCESS);
                const int64_t minElements = MIN_UNIQUE_COUNT;
                const int64_t maxElements = testCase.second;
                EXPECT_EQ(*context->GetOutputShapeRange(VALUES_OUTPUT)->GetMin(), gert::Shape({minElements}));
                EXPECT_EQ(*context->GetOutputShapeRange(VALUES_OUTPUT)->GetMax(), gert::Shape({maxElements}));
                EXPECT_EQ(*context->GetOutputShapeRange(INDICES_OUTPUT)->GetMin(),
                          inverse || counts ? min : gert::Shape({0}));
                EXPECT_EQ(*context->GetOutputShapeRange(INDICES_OUTPUT)->GetMax(),
                          inverse || counts ? max : gert::Shape({0}));
                EXPECT_EQ(*context->GetOutputShapeRange(COUNTS_OUTPUT)->GetMin(),
                          gert::Shape({counts ? minElements : 0}));
                EXPECT_EQ(*context->GetOutputShapeRange(COUNTS_OUTPUT)->GetMax(),
                          gert::Shape({counts ? maxElements : 0}));
            }
        }
    }
}
} // namespace
