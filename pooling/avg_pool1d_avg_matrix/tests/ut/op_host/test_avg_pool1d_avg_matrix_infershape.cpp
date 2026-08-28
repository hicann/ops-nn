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
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
constexpr size_t kInputX = 0;
constexpr size_t kOutputY = 0;

gert::KernelRunContextHolder BuildShapeContext(gert::StorageShape& inputShape, gert::StorageShape& outputShape,
                                               ge::Format format, int64_t ksize, int64_t strides,
                                               const std::vector<int64_t>& pads, bool ceilMode,
                                               bool countIncludePad = false)
{
    return gert::InferShapeContextFaker()
        .NodeIoNum(kInputNum, kOutputNum)
        .IrInstanceNum({1})
        .NodeInputTd(kInputX, ge::DT_FLOAT, format, format)
        .NodeOutputTd(kOutputY, ge::DT_FLOAT, format, format)
        .InputShapes({&inputShape})
        .OutputShapes({&outputShape})
        .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<int64_t>(ksize)},
                    {"strides", Ops::NN::AnyValue::CreateFrom<int64_t>(strides)},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                    {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceilMode)},
                    {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(countIncludePad)}})
        .Build();
}

void ExpectShape(const gert::Shape& actual, const std::vector<int64_t>& expected)
{
    ASSERT_EQ(actual.GetDimNum(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual.GetDim(i), expected[i]);
    }
}
} // namespace

TEST(AvgPool1DAvgMatrixInferShape, NchwFloorMode)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{2, 3, 4, 8}, {2, 3, 4, 8}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = BuildShapeContext(inputShape, outputShape, ge::FORMAT_NCHW, 3, 2, {1, 1}, false);

    ASSERT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ExpectShape(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(kOutputY), {1, 16, 1, 4});
}

TEST(AvgPool1DAvgMatrixInferShape, MatchesLegacyRt1Case)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{1, 1, 1, 4}, {1, 1, 1, 4}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = BuildShapeContext(inputShape, outputShape, ge::FORMAT_NCHW, 2, 2, {1, 2}, false, true);

    ASSERT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ExpectShape(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(kOutputY), {1, 16, 1, 3});
}

TEST(AvgPool1DAvgMatrixInferShape, NhwcCeilMode)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{2, 4, 8, 3}, {2, 4, 8, 3}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = BuildShapeContext(inputShape, outputShape, ge::FORMAT_NHWC, 3, 2, {1, 1}, true);

    ASSERT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ExpectShape(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(kOutputY), {1, 1, 5, 16});
}

TEST(AvgPool1DAvgMatrixInferShape, DynamicWidthSkipsAttrs)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{2, 3, 4, -1}, {2, 3, 4, -1}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1})
                      .NodeInputTd(kInputX, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(kOutputY, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .InputShapes({&inputShape})
                      .OutputShapes({&outputShape})
                      .Build();

    ASSERT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ExpectShape(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(kOutputY), {1, 16, 1, -1});
}

TEST(AvgPool1DAvgMatrixInferShape, ShortRankUsesLegacyZeroWidth)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{1, 1}, {1, 1}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = BuildShapeContext(inputShape, outputShape, ge::FORMAT_NCHW, 1, 1, {0, 0}, false);

    ASSERT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    ExpectShape(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(kOutputY), {1, 16, 1, 0});
}

TEST(AvgPool1DAvgMatrixInferShape, RejectsUnsupportedFormat)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{1, 1, 1, 8}, {1, 1, 1, 8}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = BuildShapeContext(inputShape, outputShape, ge::FORMAT_ND, 3, 1, {0, 0}, false);

    EXPECT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(AvgPool1DAvgMatrixInferShape, RejectsZeroStride)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{1, 1, 1, 8}, {1, 1, 1, 8}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = BuildShapeContext(inputShape, outputShape, ge::FORMAT_NCHW, 3, 0, {0, 0}, false);

    EXPECT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(AvgPool1DAvgMatrixInferShape, RejectsShortPads)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    gert::StorageShape inputShape = {{1, 1, 1, 8}, {1, 1, 1, 8}};
    gert::StorageShape outputShape = {{}, {}};
    auto holder = BuildShapeContext(inputShape, outputShape, ge::FORMAT_NCHW, 3, 1, {0}, false);

    EXPECT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST(AvgPool1DAvgMatrixInferShape, DynamicWidthRange)
{
    auto inferShapeRange = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape_range;
    ASSERT_NE(inferShapeRange, nullptr);
    gert::Shape inputMin = {1, 1, 1, 1};
    gert::Shape inputMax = {1, 1, 1, -1};
    gert::Range<gert::Shape> inputRange(&inputMin, &inputMax);
    gert::Shape outputMin;
    gert::Shape outputMax;
    gert::Range<gert::Shape> outputRange(&outputMin, &outputMax);
    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1})
                      .NodeInputTd(kInputX, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(kOutputY, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .InputShapeRanges({&inputRange})
                      .OutputShapeRanges({&outputRange})
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_EQ(inferShapeRange(context), ge::GRAPH_SUCCESS);
    ExpectShape(*context->GetOutputShapeRange(kOutputY)->GetMin(), {1, 16, 1, 1});
    ExpectShape(*context->GetOutputShapeRange(kOutputY)->GetMax(), {1, 16, 1, -1});
}

TEST(AvgPool1DAvgMatrixInferShape, StaticWidthRange)
{
    auto inferShapeRange = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool1DAvgMatrix")->infer_shape_range;
    ASSERT_NE(inferShapeRange, nullptr);
    gert::Shape inputMin = {1, 1, 1, 8};
    gert::Shape inputMax = {1, 1, 1, 8};
    gert::Range<gert::Shape> inputRange(&inputMin, &inputMax);
    gert::Shape outputMin;
    gert::Shape outputMax;
    gert::Range<gert::Shape> outputRange(&outputMin, &outputMax);
    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(kInputNum, kOutputNum)
                      .IrInstanceNum({1})
                      .NodeInputTd(kInputX, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(kOutputY, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .InputShapeRanges({&inputRange})
                      .OutputShapeRanges({&outputRange})
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_EQ(inferShapeRange(context), ge::GRAPH_SUCCESS);
    ExpectShape(*context->GetOutputShapeRange(kOutputY)->GetMin(), {1, 16, 1, 4});
    ExpectShape(*context->GetOutputShapeRange(kOutputY)->GetMax(), {1, 16, 1, 4});
}
