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
 * \file test_conv2d_backprop_filter_infershape.cpp
 * \brief 覆盖迁移自1.0(nn_calculation_ops.cc Conv2DBackpropFilterInfer)的显式校验：
 *        filter_size为const且x为静态shape时，out_backprop的H/W维度不支持-1
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <string>
#include <vector>

namespace {
constexpr size_t kFilterSizeEleNum = 4;
} // namespace

class Conv2DBackpropFilterRuntimeInferShape : public testing::Test {};

static gert::KernelRunContextHolder BuildConv2DBackpropFilterContext(
    std::unique_ptr<uint8_t[]>& tensor_holder, bool with_const_filter_size, const std::vector<int64_t>& filter_size,
    gert::StorageShape& x_shape, gert::StorageShape& dedy_shape, ge::Format dedy_format)
{
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, dedy_format, dedy_format)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                    {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                    {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                    {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}});
    if (with_const_filter_size) {
        size_t total_size = 0;
        tensor_holder = gert::Tensor::CreateFollowing(kFilterSizeEleNum, ge::DT_INT64, total_size);
        auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
        tensor->MutableStorageShape().AppendDim(kFilterSizeEleNum);
        tensor->MutableOriginShape().AppendDim(kFilterSizeEleNum);
        tensor->SetOriginFormat(ge::FORMAT_ND);
        tensor->SetStorageFormat(ge::FORMAT_ND);
        std::vector<int64_t> filterSizeValues(filter_size);
        (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), filterSizeValues.data(),
                       filterSizeValues.size() * sizeof(int64_t));
        return faker.InputShapes({&x_shape, tensor, &dedy_shape}).OutputShapes({&output_shape}).Build();
    }
    gert::StorageShape filter_size_shape = {{kFilterSizeEleNum}, {kFilterSizeEleNum}};
    return faker.InputShapes({&x_shape, &filter_size_shape, &dedy_shape}).OutputShapes({&output_shape}).Build();
}

static ge::graphStatus RunConv2DBackpropFilterInferShape(std::unique_ptr<uint8_t[]>& tensor_holder,
                                                         bool with_const_filter_size,
                                                         const std::vector<int64_t>& filter_size,
                                                         gert::StorageShape& x_shape, gert::StorageShape& dedy_shape,
                                                         ge::Format dedy_format)
{
    auto holder = BuildConv2DBackpropFilterContext(tensor_holder, with_const_filter_size, filter_size, x_shape,
                                                   dedy_shape, dedy_format);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropFilter")->infer_shape;
    return infer_shape_func(holder.GetContext<gert::InferShapeContext>());
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropNegativeHFail)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropFilter"), nullptr);
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(
        RunConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape, ge::FORMAT_NCHW),
        ge::GRAPH_FAILED);
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropAllNegativeFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    EXPECT_EQ(
        RunConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape, ge::FORMAT_NCHW),
        ge::GRAPH_FAILED);
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropNegativeWNhwcFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 16, -1, 64}, {2, 16, -1, 64}};
    EXPECT_EQ(
        RunConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape, ge::FORMAT_NHWC),
        ge::GRAPH_FAILED);
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropUnknownRankSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{-2}, {-2}};
    EXPECT_EQ(
        RunConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape, ge::FORMAT_NCHW),
        ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropNegativeNSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{-1, 64, 16, 16}, {-1, 64, 16, 16}};
    EXPECT_EQ(
        RunConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape, ge::FORMAT_NCHW),
        ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropStaticSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    auto holder = BuildConv2DBackpropFilterContext(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                   ge::FORMAT_NCHW);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[64, 32, 3, 3]");
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, constFilterSizeDynamicXOutBackpropNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{-1, 32, 16, 16}, {-1, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(
        RunConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape, ge::FORMAT_NCHW),
        ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DBackpropFilterRuntimeInferShape, dynamicFilterSizeStaticXOutBackpropNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(
        RunConv2DBackpropFilterInferShape(tensor_holder, false, {64, 32, 3, 3}, x_shape, dedy_shape, ge::FORMAT_NCHW),
        ge::GRAPH_SUCCESS);
}

// filter_size const不可见时，y的C维=x的C维/groups、N维=out_backprop的C维，其余保持-1
// 现场用例形态：x=[1,2,2,2]静态、dedy全-1、groups=1 → 期望y=[-1,2,-1,-1]
TEST_F(Conv2DBackpropFilterRuntimeInferShape, dynamicFilterSizeStaticXPartialInferSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{1, 2, 2, 2}, {1, 2, 2, 2}};
    gert::StorageShape dedy_shape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    auto holder = BuildConv2DBackpropFilterContext(tensor_holder, false, {}, x_shape, dedy_shape, ge::FORMAT_NCHW);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[-1, 2, -1, -1]");
}

// 部分推导：dedy的C维已知时N维同步补全；groups=2时C维=x_C/groups
TEST_F(Conv2DBackpropFilterRuntimeInferShape, dynamicFilterSizePartialInferWithDedyCGroups2Success)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{1, 4, 2, 2}, {1, 4, 2, 2}};
    gert::StorageShape dedy_shape = {{-1, 8, -1, -1}, {-1, 8, -1, -1}};
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                    {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                    {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                    {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}});
    gert::StorageShape filter_size_shape = {{kFilterSizeEleNum}, {kFilterSizeEleNum}};
    auto holder = faker.InputShapes({&x_shape, &filter_size_shape, &dedy_shape}).OutputShapes({&output_shape}).Build();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[8, 2, -1, -1]");
}

// 部分推导守卫(与1.0一致)：x的C维无法整除groups时报错
TEST_F(Conv2DBackpropFilterRuntimeInferShape, dynamicFilterSizeXChannelNotDivisibleByGroupsFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{1, 3, 2, 2}, {1, 3, 2, 2}};
    gert::StorageShape dedy_shape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                    {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                    {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                    {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}});
    gert::StorageShape filter_size_shape = {{kFilterSizeEleNum}, {kFilterSizeEleNum}};
    auto holder = faker.InputShapes({&x_shape, &filter_size_shape, &dedy_shape}).OutputShapes({&output_shape}).Build();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
