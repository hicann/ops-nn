/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_extend_conv_transpose_infershape.cpp
 * \brief
 */
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

class ExtendConvTransposeProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExtendConvTranspose Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ExtendConvTranspose Proto Test TearDown" << std::endl; }
};

TEST_F(ExtendConvTransposeProtoTest, basic)
{
    vector<int64_t> strides({1, 1, 1, 1, 1});
    vector<int64_t> pads({0, 0, 0, 0, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCDHW");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    vector<int64_t> input_size = {4, 256, 1, 40, 32};
    gert::StorageShape input_size_shape = {{4, 256, 1, 40, 32}, {4, 256, 1, 40, 32}};
    gert::StorageShape x_shape = {{4, 512, 1, 20, 16}, {4, 512, 1, 20, 16}};
    gert::StorageShape filter_shape = {{512, 256, 1, 2, 2}, {512, 256, 1, 2, 2}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCDHW);
    tensor->SetStorageFormat(ge::FORMAT_NCDHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[4, 256, 1, 40, 32]");
}

TEST_F(ExtendConvTransposeProtoTest, dynamic)
{
    vector<int64_t> strides({1, 1, 1, 1, 1});
    vector<int64_t> pads({0, 0, 0, 0, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCDHW");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    vector<int64_t> input_size = {4, 256, 1, 40, 32};
    gert::StorageShape input_size_shape = {{4, 256, 1, 40, 32}, {4, 256, 1, 40, 32}};
    gert::StorageShape x_shape = {{-1, 512, 1, 20, 16}, {4, 512, 1, 20, 16}};
    gert::StorageShape filter_shape = {{512, 256, 1, 2, 2}, {512, 256, 1, 2, 2}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCDHW);
    tensor->SetStorageFormat(ge::FORMAT_NCDHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, -1, -1, -1, -1]");
}

TEST_F(ExtendConvTransposeProtoTest, base_dtype)
{
    vector<int64_t> strides({1, 1, 1, 1, 1});
    vector<int64_t> pads({0, 0, 0, 0, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCDHW");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    ge::DataType input_sizeDtype = ge::DT_INT32;
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType filterDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .InputDataTypes({&input_sizeDtype, &xDtype, &filterDtype})
                      .OutputDataTypes({&yDtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(ExtendConvTransposeProtoTest, 2d_extend_3d)
{
    vector<int64_t> strides({1, 1, 1, 1, 1});
    vector<int64_t> pads({0, 0, 0, 0, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCDHW");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    vector<int32_t> input_size = {8, 8, 24, 32};
    gert::StorageShape input_size_shape = {{8, 8, 24, 32}, {8, 8, 24, 32}};
    gert::StorageShape x_shape = {{8, 8, 24, 32}, {8, 8, 24, 32}};
    gert::StorageShape filter_shape = {{32, 8, 1, 1, 1}, {32, 8, 1, 1, 1}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT32,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCDHW);
    tensor->SetStorageFormat(ge::FORMAT_NCDHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[8, 8, 1, 24, 32]");
}

// input_size all zero -> triggers CheckOutputAllZero -> compute from scratch via conv3d transpose formula
// NCDHW format, pads size 6
TEST_F(ExtendConvTransposeProtoTest, compute_from_scratch_ncdhw)
{
    vector<int64_t> strides({1, 1, 2, 2, 2});
    vector<int64_t> pads({0, 0, 0, 0, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCDHW");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    vector<int64_t> input_size = {0, 0, 0, 0, 0};
    gert::StorageShape input_size_shape = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    gert::StorageShape x_shape = {{1, 3, 2, 2, 2}, {1, 3, 2, 2, 2}};
    gert::StorageShape filter_shape = {{6, 3, 3, 3, 3}, {6, 3, 3, 3, 3}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCDHW);
    tensor->SetStorageFormat(ge::FORMAT_NCDHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 3, 5, 5, 5]");
}

// compute from scratch with NDHWC format, pads size 1
TEST_F(ExtendConvTransposeProtoTest, compute_from_scratch_ndhwc)
{
    vector<int64_t> strides({1, 2, 2, 2, 1});
    vector<int64_t> pads({0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NDHWC");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    vector<int64_t> input_size = {0, 0, 0, 0, 0};
    gert::StorageShape input_size_shape = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    gert::StorageShape x_shape = {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
    gert::StorageShape filter_shape = {{6, 2, 2, 2, 5}, {6, 2, 2, 2, 5}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NDHWC);
    tensor->SetStorageFormat(ge::FORMAT_NDHWC);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 4, 6, 8, 5]");
}

// compute from scratch with pads list size 3
TEST_F(ExtendConvTransposeProtoTest, compute_from_scratch_pads_3)
{
    vector<int64_t> strides({1, 1, 1, 1, 1});
    vector<int64_t> pads({0, 1, 2});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCDHW");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    vector<int64_t> input_size = {0, 0, 0, 0, 0};
    gert::StorageShape input_size_shape = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    gert::StorageShape x_shape = {{1, 3, 4, 4, 4}, {1, 3, 4, 4, 4}};
    gert::StorageShape filter_shape = {{6, 3, 3, 3, 3}, {6, 3, 3, 3, 3}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCDHW);
    tensor->SetStorageFormat(ge::FORMAT_NCDHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 3, 6, 4, 2]");
}

// compute from scratch with groups = 2
TEST_F(ExtendConvTransposeProtoTest, compute_from_scratch_groups_2)
{
    vector<int64_t> strides({1, 1, 1, 1, 1});
    vector<int64_t> pads({0, 0, 0, 0, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 2;
    string data_format("NCDHW");
    vector<int64_t> output_padding({0, 0, 0, 0, 0});

    vector<int64_t> input_size = {0, 0, 0, 0, 0};
    gert::StorageShape input_size_shape = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    gert::StorageShape x_shape = {{1, 6, 2, 2, 2}, {1, 6, 2, 2, 2}};
    gert::StorageShape filter_shape = {{4, 3, 3, 3, 3}, {4, 3, 3, 3, 3}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCDHW);
    tensor->SetStorageFormat(ge::FORMAT_NCDHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                           {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ExtendConvTranspose")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 6, 4, 4, 4]");
}
