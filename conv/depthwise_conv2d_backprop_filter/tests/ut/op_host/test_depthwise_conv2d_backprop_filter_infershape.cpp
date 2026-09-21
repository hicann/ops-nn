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
 * \file test_depthwise_conv2d_backprop_filter_infershape.cpp
 * \brief 覆盖迁移自1.0(nn_calculation_ops.cc DepthwiseConv2DBackpropFilterInferShape)的显式校验：
 *        filter_size为const且input为静态shape时，out_backprop的H/W维度不支持-1
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <vector>

namespace {
constexpr size_t kFilterSizeEleNum = 4;
} // namespace

class DepthwiseConv2DBackpropFilterRuntimeInferShape : public testing::Test {};

static gert::KernelRunContextHolder BuildDepthwiseConv2DBackpropFilterContext(
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
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
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

static ge::graphStatus RunDepthwiseConv2DBackpropFilterInferShape(
    std::unique_ptr<uint8_t[]>& tensor_holder, bool with_const_filter_size, const std::vector<int64_t>& filter_size,
    gert::StorageShape& x_shape, gert::StorageShape& dedy_shape, ge::Format dedy_format)
{
    auto holder = BuildDepthwiseConv2DBackpropFilterContext(tensor_holder, with_const_filter_size, filter_size, x_shape,
                                                            dedy_shape, dedy_format);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropFilter")->infer_shape;
    return infer_shape_func(holder.GetContext<gert::InferShapeContext>());
}

TEST_F(DepthwiseConv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropNegativeHFail)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropFilter"), nullptr);
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                         ge::FORMAT_NCHW),
              ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropNegativeWNhwcFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 16, -1, 64}, {2, 16, -1, 64}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                         ge::FORMAT_NHWC),
              ge::GRAPH_FAILED);
}

TEST_F(DepthwiseConv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropUnknownRankSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{-2}, {-2}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                         ge::FORMAT_NCHW),
              ge::GRAPH_SUCCESS);
}

TEST_F(DepthwiseConv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropNegativeNSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{-1, 64, 16, 16}, {-1, 64, 16, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                         ge::FORMAT_NCHW),
              ge::GRAPH_SUCCESS);
}

TEST_F(DepthwiseConv2DBackpropFilterRuntimeInferShape, constFilterSizeStaticXOutBackpropStaticSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    auto holder = BuildDepthwiseConv2DBackpropFilterContext(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                            ge::FORMAT_NCHW);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DepthwiseConv2DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[64, 32, 3, 3]");
}

TEST_F(DepthwiseConv2DBackpropFilterRuntimeInferShape, constFilterSizeDynamicXOutBackpropNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{-1, 32, 16, 16}, {-1, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropFilterInferShape(tensor_holder, true, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                         ge::FORMAT_NCHW),
              ge::GRAPH_SUCCESS);
}

TEST_F(DepthwiseConv2DBackpropFilterRuntimeInferShape, dynamicFilterSizeStaticXOutBackpropNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape dedy_shape = {{2, 64, -1, 16}, {2, 64, -1, 16}};
    EXPECT_EQ(RunDepthwiseConv2DBackpropFilterInferShape(tensor_holder, false, {64, 32, 3, 3}, x_shape, dedy_shape,
                                                         ge::FORMAT_NCHW),
              ge::GRAPH_SUCCESS);
}
