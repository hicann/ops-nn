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
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
using Attr = std::pair<std::string, Ops::NN::AnyValue>;

std::vector<Attr> MakeAttrs(const std::vector<int64_t>& kernelSize, const std::vector<int64_t>& stride,
                            const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, bool ceilMode)
{
    return {{"kernelSize", Ops::NN::AnyValue::CreateFrom(kernelSize)},
            {"stride", Ops::NN::AnyValue::CreateFrom(stride)},
            {"padding", Ops::NN::AnyValue::CreateFrom(padding)},
            {"dilation", Ops::NN::AnyValue::CreateFrom(dilation)},
            {"ceilMode", Ops::NN::AnyValue::CreateFrom(ceilMode)}};
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
    gert::Shape out;
    gert::Shape indices;
};

InferResult InferShape(const gert::StorageShape& input, const std::vector<Attr>& attrs)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool2dWithMask");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return {};
    }
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({const_cast<gert::StorageShape*>(&input)})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(attrs)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    InferResult result;
    if (context == nullptr) {
        return result;
    }
    result.status = opImpl->infer_shape(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.out = *context->GetOutputShape(0);
        result.indices = *context->GetOutputShape(1);
    }
    return result;
}

TEST(MaxPool2dWithMaskInferShape, RankFourTaskCase)
{
    gert::StorageShape input = {{2, 3, 8, 9}, {2, 3, 8, 9}};
    const auto result = InferShape(input, MakeAttrs({3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.out, {2, 3, 4, 5});
    ExpectShape(result.indices, {2, 3, 9, 96});
}

TEST(MaxPool2dWithMaskInferShape, RankThreeCeilModeAndEmptyStride)
{
    gert::StorageShape input = {{4, 6, 7}, {4, 6, 7}};
    const auto result = InferShape(input, MakeAttrs({3}, {}, {1}, {1}, true));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.out, {4, 3, 3});
    ExpectShape(result.indices, {4, 9, 64});
}

TEST(MaxPool2dWithMaskInferShape, PreservesUnknownRank)
{
    gert::StorageShape input = {{-2}, {-2}};
    const auto result = InferShape(input, MakeAttrs({3, 3}, {2, 2}, {1, 1}, {1, 1}, false));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.out, {-2});
    ExpectShape(result.indices, {-2});
}

TEST(MaxPool2dWithMaskInferShape, CeilModeAllowsPartialFirstWindow)
{
    gert::StorageShape input = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    const auto result = InferShape(input, MakeAttrs({3}, {2}, {0}, {1}, true));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.out, {1, 1, 1, 1});
    ExpectShape(result.indices, {1, 1, 9, 64});
    EXPECT_EQ(InferShape(input, MakeAttrs({3}, {2}, {0}, {1}, false)).status, ge::GRAPH_FAILED);
}

TEST(MaxPool2dWithMaskInferShape, LargeStrideDoesNotOverflowCeilDivision)
{
    gert::StorageShape input = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    const auto result = InferShape(input, MakeAttrs({2}, {std::numeric_limits<int64_t>::max()}, {0}, {1}, true));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.out, {1, 1, 1, 1});
}

TEST(MaxPool2dWithMaskInferShape, RejectsInvalidParameters)
{
    gert::StorageShape input = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    EXPECT_EQ(InferShape(input, MakeAttrs({3, 3}, {2, 2}, {2, 1}, {1, 1}, false)).status, ge::GRAPH_FAILED);
    EXPECT_EQ(InferShape(input, MakeAttrs({3, 3}, {2, 2}, {1, 1}, {2, 1}, false)).status, ge::GRAPH_FAILED);
    EXPECT_EQ(InferShape(input, MakeAttrs({3, 3, 3}, {2, 2}, {1, 1}, {1, 1}, false)).status, ge::GRAPH_FAILED);
}

TEST(MaxPool2dWithMaskInferShape, OneByOneHasSpaceForEveryInt32Index)
{
    for (const auto& boundary :
         {std::pair<int64_t, int64_t>{16, 64}, {17, 96}, {24, 96}, {25, 128}, {32, 128}, {33, 160}}) {
        const int64_t elements = boundary.first;
        const int64_t width = boundary.second;
        gert::StorageShape nchw = {{2, 3, 1, elements}, {2, 3, 1, elements}};
        const auto rankFour = InferShape(nchw, MakeAttrs({1}, {1}, {0}, {1}, false));
        ASSERT_EQ(rankFour.status, ge::GRAPH_SUCCESS);
        ExpectShape(rankFour.out, {2, 3, 1, elements});
        ExpectShape(rankFour.indices, {2, 3, 1, width});
        EXPECT_GE(width, elements * static_cast<int64_t>(sizeof(int32_t)));
        gert::StorageShape chw = {{3, 1, elements}, {3, 1, elements}};
        const auto rankThree = InferShape(chw, MakeAttrs({1}, {}, {0}, {1}, false));
        ASSERT_EQ(rankThree.status, ge::GRAPH_SUCCESS);
        ExpectShape(rankThree.out, {3, 1, elements});
        ExpectShape(rankThree.indices, {3, 1, width});
    }
    gert::StorageShape input = {{1, 1, 5, 5}, {1, 1, 5, 5}};
    const auto result = InferShape(input, MakeAttrs({1}, {1}, {0}, {1}, false));
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ExpectShape(result.indices, {1, 1, 1, 128});
}

} // namespace
