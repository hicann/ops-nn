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
 * \file test_depthwise_conv2d_backprop_input_infershape.cpp
 * \brief 覆盖depthwisedx校验规格：
 *        input_size为非const时，filter_shape不支持-1/-2，out_backprop shape支持-1/-2；
 *        input_size为const时，filter_shape不支持-1/-2，out_backprop shape支持-1，不支持-2
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <vector>

namespace {
constexpr size_t kInputSizeEleNum = 4;
} // namespace

class DepthwiseConv2DBackpropInputRuntimeInferShape : public testing::Test {};

static gert::KernelRunContextHolder BuildDepthwiseConv2DBackpropInputContext(
    std::unique_ptr<uint8_t[]>& tensor_holder, bool with_const_input_size, const std::vector<int64_t>& input_size,
    gert::StorageShape& filter_shape, ge::Format filter_format, gert::StorageShape& dedy_shape)
{
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, filter_format, filter_format)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    if (with_const_input_size) {
        size_t total_size = 0;
        tensor_holder = gert::Tensor::CreateFollowing(kInputSizeEleNum, ge::DT_INT64, total_size);
        auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
        tensor->MutableStorageShape().AppendDim(kInputSizeEleNum);
        tensor->MutableOriginShape().AppendDim(kInputSizeEleNum);
        tensor->SetOriginFormat(ge::FORMAT_ND);
        tensor->SetStorageFormat(ge::FORMAT_ND);
        std::vector<int64_t> inputSizeValues(input_size);
        (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), inputSizeValues.data(),
                       inputSizeValues.size() * sizeof(int64_t));
        return faker.InputShapes({tensor, &filter_shape, &dedy_shape}).OutputShapes({&output_shape}).Build();
    }
    gert::StorageShape input_size_shape = {{kInputSizeEleNum}, {kInputSizeEleNum}};
    return faker.InputShapes({&input_size_shape, &filter_shape, &dedy_shape}).OutputShapes({&output_shape}).Build();
}

static ge::graphStatus RunDepthwiseConv2DBackpropInputInferShape(
    std::unique_ptr<uint8_t[]>& tensor_holder, bool with_const_input_size, const std::vector<int64_t>& input_size,
    gert::StorageShape& filter_shape, ge::Format filter_format, gert::StorageShape& dedy_shape)
{
    auto holder = BuildDepthwiseConv2DBackpropInputContext(tensor_holder, with_const_input_size, input_size,
                                                           filter_shape, filter_format, dedy_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropInput")->infer_shape;
    return infer_shape_func(holder.GetContext<gert::InferShapeContext>());
}

TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterNegativeHFail)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropInput"), nullptr);
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_NCHW, dedy_shape),
              ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterNegativeWHwcnFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{3, -1, 64, 32}, {3, -1, 64, 32}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_HWCN, dedy_shape),
              ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterUnknownRankFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{-2}, {-2}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_NCHW, dedy_shape),
              ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterPositiveSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{3, 3, 64, 32}, {3, 3, 64, 32}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    auto holder = BuildDepthwiseConv2DBackpropInputContext(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                           ge::FORMAT_HWCN, dedy_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 32, 32, 32]");
}

// 校验规格：input_size为const时，filter_shape不支持-1/-2（out_backprop含-1也不放行）
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeDynamicDedyFilterNegativeFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_NCHW, dedy_shape),
              ge::GRAPH_FAILED);
}

// 校验规格：input_size为const时，filter_shape不支持-1/-2（filter含-1即报错，无论out_backprop状态）
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeUnknownRankDedyFilterNegativeFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{-2}, {-2}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_NCHW, dedy_shape),
              ge::GRAPH_FAILED);
}

// 校验规格：input_size为非const时，filter_shape仍不支持-1/-2
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, dynamicInputSizeStaticDedyFilterNegativeFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, false, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_NCHW, dedy_shape),
              ge::GRAPH_FAILED);
}

// 校验规格：input_size为非const时，filter_shape不支持[-2]
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, dynamicInputSizeStaticDedyFilterUnknownRankFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{-2}, {-2}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, false, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_NCHW, dedy_shape),
              ge::GRAPH_FAILED);
}

// 校验规格：filter_shape任一维度不支持-1（含C/N维度，非仅H/W）
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterNegativeCinFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{-1, 1, 3, 3}, {-1, 1, 3, 3}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_NCHW, dedy_shape),
              ge::GRAPH_FAILED);
}

// 校验规格：input_size为非const时，out_backprop shape支持-1
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, dynamicInputSizeDynamicDedyStaticFilterSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{3, 3, 64, 32}, {3, 3, 64, 32}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    auto holder = BuildDepthwiseConv2DBackpropInputContext(tensor_holder, false, {2, 32, 32, 32}, filter_shape,
                                                           ge::FORMAT_HWCN, dedy_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[-1, -1, -1, -1]");
}

// 校验规格：input_size为非const时，out_backprop shape支持-2
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, dynamicInputSizeUnknownRankDedyStaticFilterSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{3, 3, 64, 32}, {3, 3, 64, 32}};
    gert::StorageShape dedy_shape = {{-2}, {-2}};
    auto holder = BuildDepthwiseConv2DBackpropInputContext(tensor_holder, false, {2, 32, 32, 32}, filter_shape,
                                                           ge::FORMAT_HWCN, dedy_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[-1, -1, -1, -1]");
}

// 校验规格：input_size为const时，out_backprop shape不支持-2（支持-1）
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeUnknownRankDedyStaticFilterFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{3, 3, 64, 32}, {3, 3, 64, 32}};
    gert::StorageShape dedy_shape = {{-2}, {-2}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_HWCN, dedy_shape),
              ge::GRAPH_FAILED);
}

// 校验规格：input_size为const时，out_backprop shape支持-1
TEST_F(DepthwiseConv2DBackpropInputRuntimeInferShape, constInputSizeDynamicDedyStaticFilterSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{3, 3, 64, 32}, {3, 3, 64, 32}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape,
                                                        ge::FORMAT_HWCN, dedy_shape),
              ge::GRAPH_SUCCESS);
}
