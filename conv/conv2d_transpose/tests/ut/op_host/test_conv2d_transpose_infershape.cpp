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
 * \file test_conv2d_transpose_infershape.cpp
 * \brief 覆盖迁移自1.0(nn_calculation_ops.cc Conv2DTransposeInfer)的显式校验：
 *        input_size为const且x为静态shape时，filter的H/W维度不支持-1/-2；
 *        以及1.0占位符修正(SetDeDxAttrForConv2DBackpropInput的CheckShapeAllZero分支)：
 *        onnx插件合成的input_size为{0,0,0,0}全0占位符时，按ONNX语义重算输出shape
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

class Conv2DTransposeRuntimeInferShape : public testing::Test {};

static gert::KernelRunContextHolder BuildConv2DTransposeContext(
    std::unique_ptr<uint8_t[]>& tensor_holder, bool with_const_input_size, const std::vector<int64_t>& input_size,
    gert::StorageShape& x_shape, gert::StorageShape& filter_shape, ge::Format filter_format)
{
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(2, ge::DT_FLOAT16, filter_format, filter_format)
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
        return faker.InputShapes({tensor, &x_shape, &filter_shape}).OutputShapes({&output_shape}).Build();
    }
    gert::StorageShape input_size_shape = {{kInputSizeEleNum}, {kInputSizeEleNum}};
    return faker.InputShapes({&input_size_shape, &x_shape, &filter_shape}).OutputShapes({&output_shape}).Build();
}

static ge::graphStatus RunConv2DTransposeInferShape(std::unique_ptr<uint8_t[]>& tensor_holder,
                                                    bool with_const_input_size, const std::vector<int64_t>& input_size,
                                                    gert::StorageShape& x_shape, gert::StorageShape& filter_shape,
                                                    ge::Format filter_format)
{
    auto holder = BuildConv2DTransposeContext(tensor_holder, with_const_input_size, input_size, x_shape, filter_shape,
                                              filter_format);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose")->infer_shape;
    return infer_shape_func(holder.GetContext<gert::InferShapeContext>());
}

TEST_F(Conv2DTransposeRuntimeInferShape, constInputSizeStaticXFilterNegativeHFail)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose"), nullptr);
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{32, 64, 3, -1}, {32, 64, 3, -1}};
    EXPECT_EQ(
        RunConv2DTransposeInferShape(tensor_holder, true, {2, 32, 32, 32}, x_shape, filter_shape, ge::FORMAT_NCHW),
        ge::GRAPH_FAILED);
}

TEST_F(Conv2DTransposeRuntimeInferShape, constInputSizeStaticXFilterNegativeHFhwcFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{-1, 3, 64, 32}, {-1, 3, 64, 32}};
    EXPECT_EQ(
        RunConv2DTransposeInferShape(tensor_holder, true, {2, 32, 32, 32}, x_shape, filter_shape, ge::FORMAT_HWCN),
        ge::GRAPH_FAILED);
}

TEST_F(Conv2DTransposeRuntimeInferShape, constInputSizeStaticXFilterUnknownRankFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{-2}, {-2}};
    EXPECT_EQ(
        RunConv2DTransposeInferShape(tensor_holder, true, {2, 32, 32, 32}, x_shape, filter_shape, ge::FORMAT_NCHW),
        ge::GRAPH_FAILED);
}

TEST_F(Conv2DTransposeRuntimeInferShape, constInputSizeStaticXFilterPositiveSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{32, 64, 3, 3}, {32, 64, 3, 3}};
    auto holder = BuildConv2DTransposeContext(tensor_holder, true, {2, 32, 32, 32}, x_shape, filter_shape,
                                              ge::FORMAT_NCHW);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 32, 32, 32]");
}

TEST_F(Conv2DTransposeRuntimeInferShape, constInputSizeDynamicXFilterNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, -1, -1}, {2, 64, -1, -1}};
    gert::StorageShape filter_shape = {{32, 64, 3, -1}, {32, 64, 3, -1}};
    EXPECT_EQ(
        RunConv2DTransposeInferShape(tensor_holder, true, {2, 32, 32, 32}, x_shape, filter_shape, ge::FORMAT_NCHW),
        ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DTransposeRuntimeInferShape, dynamicInputSizeStaticXFilterNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{32, 64, 3, -1}, {32, 64, 3, -1}};
    EXPECT_EQ(
        RunConv2DTransposeInferShape(tensor_holder, false, {2, 32, 32, 32}, x_shape, filter_shape, ge::FORMAT_NCHW),
        ge::GRAPH_SUCCESS);
}

// onnx插件合成全0占位符input_size，faker属性按下标追加，需按IR顺序(strides/pads/dilations/groups/
// data_format/output_padding/offset_x/padding/auto_pad/output_shape/_op_impl_mode_enum)传全11个
static gert::KernelRunContextHolder BuildConv2DTransposePlaceholderContext(
    std::unique_ptr<uint8_t[]>& tensor_holder, gert::StorageShape& x_shape, gert::StorageShape& filter_shape,
    ge::Format filter_format, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations, const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& output_shape)
{
    gert::StorageShape output_shape_desc = {{}, {}};
    size_t total_size = 0;
    tensor_holder = gert::Tensor::CreateFollowing(kInputSizeEleNum, ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(kInputSizeEleNum);
    tensor->MutableOriginShape().AppendDim(kInputSizeEleNum);
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    const std::vector<int64_t> placeholder = {0, 0, 0, 0};
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), placeholder.data(),
                   placeholder.size() * sizeof(int64_t));

    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(2, ge::DT_FLOAT16, filter_format, filter_format)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                    {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                    {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)},
                    {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                    {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                    {"auto_pad", Ops::NN::AnyValue::CreateFrom<std::string>("NOTSET")},
                    {"output_shape", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_shape)},
                    {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(-1L)}});
    return faker.InputShapes({tensor, &x_shape, &filter_shape}).OutputShapes({&output_shape_desc}).Build();
}

static ge::graphStatus RunConv2DTransposePlaceholderInferShape(
    std::unique_ptr<uint8_t[]>& tensor_holder, gert::StorageShape& x_shape, gert::StorageShape& filter_shape,
    ge::Format filter_format, const std::vector<int64_t>& strides, const std::vector<int64_t>& pads,
    const std::vector<int64_t>& dilations, const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& output_shape)
{
    auto holder = BuildConv2DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, filter_format, strides,
                                                         pads, dilations, output_padding, output_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose")->infer_shape;
    return infer_shape_func(holder.GetContext<gert::InferShapeContext>());
}

// 占位符修正：output_shape属性为{0,0,0,0}时按公式重算
// outH = 2*(16-1) + (0 + (3-1)*1+1) - (1+1) = 31，outW同，N取x=2，C取filter的C维64
TEST_F(Conv2DTransposeRuntimeInferShape, zeroPlaceholderInputSizeFormulaSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{32, 64, 3, 3}, {32, 64, 3, 3}};
    auto holder = BuildConv2DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCHW,
                                                         {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0},
                                                         {0, 0, 0, 0});
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 64, 31, 31]");
}

// 占位符修正：模型带output_shape属性(统一为HW布局)时优先采信
TEST_F(Conv2DTransposeRuntimeInferShape, zeroPlaceholderWithOutputShapeAttrSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{32, 64, 3, 3}, {32, 64, 3, 3}};
    auto holder = BuildConv2DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCHW,
                                                         {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0},
                                                         {28, 28});
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 64, 28, 28]");
}

// 占位符修正：dilation参与公式 outH = 2*15 + ((3-1)*2+1) - 2 = 33
TEST_F(Conv2DTransposeRuntimeInferShape, zeroPlaceholderDilationFormulaSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{32, 64, 3, 3}, {32, 64, 3, 3}};
    auto holder = BuildConv2DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCHW,
                                                         {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 2, 2}, {0, 0, 0, 0},
                                                         {0, 0, 0, 0});
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 64, 33, 33]");
}

// 占位符修正：HWCN filter按格式位置取C/H/W(dim2/dim0/dim1)，结果与NCHW一致
TEST_F(Conv2DTransposeRuntimeInferShape, zeroPlaceholderHwcnFilterFormulaSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    gert::StorageShape filter_shape = {{3, 3, 64, 32}, {3, 3, 64, 32}};
    auto holder = BuildConv2DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_HWCN,
                                                         {1, 1, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0},
                                                         {0, 0, 0, 0});
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 64, 31, 31]");
}

// 占位符修正：pads过大导致推导H/W非正时报错 outH = 0 + 5 - 8 = -3
TEST_F(Conv2DTransposeRuntimeInferShape, zeroPlaceholderNegativeOutputFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 64, 1, 1}, {2, 64, 1, 1}};
    gert::StorageShape filter_shape = {{32, 64, 5, 5}, {32, 64, 5, 5}};
    EXPECT_EQ(
        RunConv2DTransposePlaceholderInferShape(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCHW, {1, 1, 1, 1},
                                                {4, 4, 4, 4}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}),
        ge::GRAPH_FAILED);
}
