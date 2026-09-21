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
 * \file test_conv3d_transpose_infershape.cpp
 * \brief 覆盖迁移自1.0(nn_calculation_ops.cc SetInputsizeListConv3dtranspose的CheckVectorAllZero分支)
 *        的占位符修正：onnx/torch适配层合成的input_size为全0占位符时，按反卷积公式重算输出shape
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <vector>

namespace {
constexpr size_t kInputSizeEleNum = 5;
} // namespace

class Conv3DTransposeRuntimeInferShape : public testing::Test {};

// onnx/torch适配层合成全0占位符input_size，faker属性按下标追加，需按IR顺序(strides/pads/dilations/
// groups/data_format/output_padding/offset_x/padding/_op_impl_mode_enum)传全9个
static gert::KernelRunContextHolder BuildConv3DTransposePlaceholderContext(
    std::unique_ptr<uint8_t[]>& tensor_holder, gert::StorageShape& x_shape, gert::StorageShape& filter_shape,
    ge::Format x_format, ge::Format filter_format, const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads, const std::vector<int64_t>& dilations, const std::vector<int64_t>& output_padding,
    const char* padding = "", const int64_t groups = 1)
{
    gert::StorageShape output_shape = {{}, {}};
    size_t total_size = 0;
    tensor_holder = gert::Tensor::CreateFollowing(kInputSizeEleNum, ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(kInputSizeEleNum);
    tensor->MutableOriginShape().AppendDim(kInputSizeEleNum);
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    const std::vector<int64_t> placeholder = {0, 0, 0, 0, 0};
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), placeholder.data(),
                   placeholder.size() * sizeof(int64_t));

    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, x_format, x_format)
        .NodeInputTd(2, ge::DT_FLOAT16, filter_format, filter_format)
        .NodeOutputTd(0, ge::DT_FLOAT16, x_format, x_format)
        .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                    {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                    {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)},
                    {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                    {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                    {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(0L)}});
    return faker.InputShapes({tensor, &x_shape, &filter_shape}).OutputShapes({&output_shape}).Build();
}

static ge::graphStatus RunConv3DTransposePlaceholderInferShape(
    std::unique_ptr<uint8_t[]>& tensor_holder, gert::StorageShape& x_shape, gert::StorageShape& filter_shape,
    ge::Format x_format, ge::Format filter_format, const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads, const std::vector<int64_t>& dilations, const std::vector<int64_t>& output_padding,
    const char* padding = "")
{
    auto holder = BuildConv3DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, x_format, filter_format,
                                                         strides, pads, dilations, output_padding, padding);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose")->infer_shape;
    return infer_shape_func(holder.GetContext<gert::InferShapeContext>());
}

// 占位符修正：5D全0 input_size按公式重算
// outD = 2*(4-1) + 0 + (3-1)*1+1 - (1+1) = 7，outH/outW = 2*7+3-2 = 15，N取x=2，C取filter的C维16(NCDHW dim1)
TEST_F(Conv3DTransposeRuntimeInferShape, zeroPlaceholderInputSizeNcdhwFormulaSuccess)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose"), nullptr);
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 16, 4, 8, 8}, {2, 16, 4, 8, 8}};
    gert::StorageShape filter_shape = {{8, 16, 3, 3, 3}, {8, 16, 3, 3, 3}};
    auto holder = BuildConv3DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCDHW,
                                                         ge::FORMAT_NCDHW, {1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1},
                                                         {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0});
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[2, 16, 7, 15, 15]");
}

// 占位符修正：NDHWC布局 x/filter/output，C维在末位
// outD = 1*(4-1) + 0 + (3-1)*1+1 - 2 = 4，outH/outW = 7+3-2 = 8，C取filter NDHWC dim4=8
TEST_F(Conv3DTransposeRuntimeInferShape, zeroPlaceholderInputSizeNdhwcFormulaSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 4, 8, 8, 16}, {2, 4, 8, 8, 16}};
    gert::StorageShape filter_shape = {{16, 3, 3, 3, 8}, {16, 3, 3, 3, 8}};
    auto holder = BuildConv3DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NDHWC,
                                                         ge::FORMAT_NDHWC, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1},
                                                         {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0});
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 4, 8, 8, 8]");
}

// 占位符修正：dilation参与公式 outD = 2*3 + ((3-1)*2+1) - 2 = 9，outH/outW = 14+5-2 = 17
TEST_F(Conv3DTransposeRuntimeInferShape, zeroPlaceholderDilationFormulaSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 16, 4, 8, 8}, {2, 16, 4, 8, 8}};
    gert::StorageShape filter_shape = {{8, 16, 3, 3, 3}, {8, 16, 3, 3, 3}};
    auto holder = BuildConv3DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCDHW,
                                                         ge::FORMAT_NCDHW, {1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1},
                                                         {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0});
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[2, 16, 9, 17, 17]");
}

// 占位符修正：padding=SAME时按x/kernel重算pad，outH = 2*(8-1) + (3-1)*1+1 - 0 = 16
TEST_F(Conv3DTransposeRuntimeInferShape, zeroPlaceholderPaddingSameSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 16, 4, 8, 8}, {2, 16, 4, 8, 8}};
    gert::StorageShape filter_shape = {{8, 16, 3, 3, 3}, {8, 16, 3, 3, 3}};
    auto holder = BuildConv3DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCDHW,
                                                         ge::FORMAT_NCDHW, {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0, 0},
                                                         {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}, "SAME");
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[2, 16, 8, 16, 16]");
}

// 占位符修正：groups>1时输出C=filter的C维*groups(对齐V2口径)，精确复刻现场group=2用例：
// filter为torch布局[cin,cout/g,kH,kW]=[2,13,1,3](升维后[2,13,1,1,3], cin=2, cout/g=13)，
// gradinput(op输入x)=[1,2,9,29]且x.C==filter的N维cin，gradoutput(期望输出y)=[1,26,9,60]：
// y.C=13*2=26，outH=1*(9-1)+0+1-0=9，outW=2*(29-1)+1+3-0=60
TEST_F(Conv3DTransposeRuntimeInferShape, zeroPlaceholderGroupsFormulaSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{1, 2, 1, 9, 29}, {1, 2, 1, 9, 29}};
    gert::StorageShape filter_shape = {{2, 13, 1, 1, 3}, {2, 13, 1, 1, 3}};
    auto holder = BuildConv3DTransposePlaceholderContext(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCDHW,
                                                         ge::FORMAT_NCDHW, {1, 1, 1, 1, 2}, {0, 0, 0, 0, 0, 0},
                                                         {1, 1, 1, 1, 1}, {0, 0, 0, 0, 1}, "", 2);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[1, 26, 1, 9, 60]");
}

// 占位符修正：pads过大导致推导D非正时报错 outD = 1*(2-1) + 5 - (3+3) = 0
TEST_F(Conv3DTransposeRuntimeInferShape, zeroPlaceholderNonPositiveOutputFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 16, 2, 8, 8}, {2, 16, 2, 8, 8}};
    gert::StorageShape filter_shape = {{8, 16, 5, 3, 3}, {8, 16, 5, 3, 3}};
    EXPECT_EQ(RunConv3DTransposePlaceholderInferShape(tensor_holder, x_shape, filter_shape, ge::FORMAT_NCDHW,
                                                      ge::FORMAT_NCDHW, {1, 1, 1, 1, 1}, {3, 3, 1, 1, 1, 1},
                                                      {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}),
              ge::GRAPH_FAILED);
}

// 占位符修正：真实非全0的const input_size不受影响，直接采信const值
TEST_F(Conv3DTransposeRuntimeInferShape, realConstInputSizeNotAffectedSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 16, 4, 8, 8}, {2, 16, 4, 8, 8}};
    gert::StorageShape filter_shape = {{8, 16, 3, 3, 3}, {8, 16, 3, 3, 3}};
    gert::StorageShape output_shape = {{}, {}};
    size_t total_size = 0;
    tensor_holder = gert::Tensor::CreateFollowing(kInputSizeEleNum, ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(kInputSizeEleNum);
    tensor->MutableOriginShape().AppendDim(kInputSizeEleNum);
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    const std::vector<int64_t> realSize = {2, 16, 7, 17, 17};
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), realSize.data(),
                   realSize.size() * sizeof(int64_t));
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .Build();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTranspose")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[2, 16, 7, 17, 17]");
}
