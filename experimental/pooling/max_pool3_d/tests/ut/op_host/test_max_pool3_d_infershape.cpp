/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_graph/max_pool3_d_proto.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
using Attr = std::pair<std::string, Ops::NN::AnyValue>;

std::vector<Attr> MakeAttrs(const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides,
                            const std::string& padding, const std::vector<int64_t>& pads,
                            const std::vector<int64_t>& dilation, int64_t ceilMode, const std::string& dataFormat)
{
    return {{"ksize", Ops::NN::AnyValue::CreateFrom(ksize)},
            {"strides", Ops::NN::AnyValue::CreateFrom(strides)},
            {"padding", Ops::NN::AnyValue::CreateFrom(padding)},
            {"pads", Ops::NN::AnyValue::CreateFrom(pads)},
            {"dilation", Ops::NN::AnyValue::CreateFrom(dilation)},
            {"ceil_mode", Ops::NN::AnyValue::CreateFrom(ceilMode)},
            {"data_format", Ops::NN::AnyValue::CreateFrom(dataFormat)}};
}

void ExpectShape(const gert::Shape& actual, std::initializer_list<int64_t> expected)
{
    ASSERT_EQ(actual.GetDimNum(), expected.size());
    size_t index = 0;
    for (const int64_t dim : expected) {
        EXPECT_EQ(actual.GetDim(index), dim);
        ++index;
    }
}

struct InferResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    gert::Shape shape;
};

InferResult InferShape(const gert::StorageShape& input, ge::Format originFormat, const std::vector<Attr>& attrs)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3D");
    if (opImpl == nullptr) {
        return {};
    }
    if (opImpl->infer_shape == nullptr) {
        return {};
    }
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({const_cast<gert::StorageShape*>(&input)})
                      .NodeInputTd(0, ge::DT_FLOAT, originFormat, ge::FORMAT_ND)
                      .NodeAttrs(attrs)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    InferResult result;
    if (context == nullptr) {
        return result;
    }
    result.status = opImpl->infer_shape(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        auto* outputShape = context->GetOutputShape(0);
        if (outputShape != nullptr) {
            result.shape = *outputShape;
        }
    }
    return result;
}

TEST(MaxPool3DInferShape, ValidNcdhw)
{
    gert::StorageShape input = {{2, 3, 8, 10, 12}, {2, 3, 8, 10, 12}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 3}, {1, 1, 2, 2, 3}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    const auto result = InferShape(input, ge::FORMAT_NCDHW, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {2, 3, 4, 5, 4});
}

TEST(MaxPool3DInferShape, SameNdhwc)
{
    gert::StorageShape input = {{2, 8, 10, 12, 3}, {2, 8, 10, 12, 3}};
    const auto attrs = MakeAttrs({1, 2, 2, 3, 1}, {1, 2, 2, 3, 1}, "SAME", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NDHWC");
    const auto result = InferShape(input, ge::FORMAT_NDHWC, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {2, 4, 5, 4, 3});
}

TEST(MaxPool3DInferShape, CalculatedNcdhw)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({1, 1, 3, 3, 3}, {1, 1, 2, 2, 2}, "CALCULATED", {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    const auto result = InferShape(input, ge::FORMAT_NCDHW, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {1, 2, 3, 3, 4});
}

TEST(MaxPool3DInferShape, CalculatedCeilDilatedNdhwc)
{
    gert::StorageShape input = {{1, 5, 6, 7, 3}, {1, 5, 6, 7, 3}};
    const auto attrs = MakeAttrs({1, 3, 3, 3, 1}, {1, 2, 2, 2, 1}, "CALCULATED", {1, 1, 1, 1, 1, 1}, {1, 1, 2, 1, 1}, 1,
                                 "NDHWC");
    const auto result = InferShape(input, ge::FORMAT_NDHWC, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {1, 3, 3, 4, 3});
}

TEST(MaxPool3DInferShape, ValidThreeElementDilatedAttrs)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({3, 3, 3}, {1, 1, 1}, "VALID", {0, 0, 0, 0, 0, 0}, {2, 2, 2}, 0, "NCDHW");
    const auto result = InferShape(input, ge::FORMAT_NCDHW, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {1, 2, 1, 2, 3});
}

TEST(MaxPool3DInferShape, SameOneElementStride)
{
    gert::StorageShape input = {{1, 5, 6, 7, 3}, {1, 5, 6, 7, 3}};
    const auto attrs = MakeAttrs({2}, {2}, "SAME", {0, 0, 0, 0, 0, 0}, {1}, 0, "NDHWC");
    const auto result = InferShape(input, ge::FORMAT_NDHWC, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {1, 3, 3, 4, 3});
}

TEST(MaxPool3DInferShape, PreservesUnknownRank)
{
    gert::StorageShape input = {{-2}, {-2}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    const auto result = InferShape(input, ge::FORMAT_NCDHW, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {-2});
}

TEST(MaxPool3DInferShape, PreservesUnknownShape)
{
    gert::StorageShape input = {{1, 2, -1, 6, 7}, {1, 2, -1, 6, 7}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    const auto result = InferShape(input, ge::FORMAT_NCDHW, attrs);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.shape, {-1, -1, -1, -1, -1});
}

TEST(MaxPool3DInferShape, RejectsInvalidPadding)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "INVALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, RejectsInvalidKsizeLength)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({2, 2}, {1, 1, 2, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0, "NCDHW");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, RejectsNonUnitOuterKsize)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({2, 1, 2, 2, 2}, {1, 1, 1, 1, 1}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, RejectsZeroSpatialStride)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 0, 2, 2}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, RejectsInvalidPadsLength)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "CALCULATED", {0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, RejectsNegativeCalculatedPad)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 2, 2, 2}, "CALCULATED", {-1, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1},
                                 0, "NCDHW");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, RejectsInvalidDilation)
{
    gert::StorageShape input = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 1, 1, 1}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 0, 1}, 0, "NCDHW");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, SameRejectsInvalidWindowAttrs)
{
    gert::StorageShape input = {{1, 5, 6, 7, 3}, {1, 5, 6, 7, 3}};
    const auto invalidKsize = MakeAttrs({0}, {1}, "SAME", {0, 0, 0, 0, 0, 0}, {1}, 0, "NDHWC");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NDHWC, invalidKsize).status, ge::GRAPH_FAILED);

    const auto invalidDilation = MakeAttrs({2}, {1}, "SAME", {0, 0, 0, 0, 0, 0}, {0}, 0, "NDHWC");
    EXPECT_EQ(InferShape(input, ge::FORMAT_NDHWC, invalidDilation).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, RejectsInvalidRankAndFormat)
{
    const auto attrs = MakeAttrs({1, 1, 2, 2, 2}, {1, 1, 1, 1, 1}, "VALID", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 0,
                                 "NCDHW");
    gert::StorageShape rankFourInput = {{1, 2, 5, 6}, {1, 2, 5, 6}};
    EXPECT_EQ(InferShape(rankFourInput, ge::FORMAT_NCDHW, attrs).status, ge::GRAPH_FAILED);

    gert::StorageShape rankFiveInput = {{1, 2, 5, 6, 7}, {1, 2, 5, 6, 7}};
    EXPECT_EQ(InferShape(rankFiveInput, ge::FORMAT_ND, attrs).status, ge::GRAPH_FAILED);
}

TEST(MaxPool3DInferShape, PropagatesSupportedDtypes)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3D");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    for (const ge::DataType dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        ge::DataType inputDtype = dtype;
        ge::DataType outputDtype = ge::DT_UNDEFINED;
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(1, 1)
                          .IrInstanceNum({1})
                          .InputDataTypes({&inputDtype})
                          .OutputDataTypes({&outputDtype})
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
    }
}
} // namespace
