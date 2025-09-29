/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv3d_backprop_input_v2_infershape.cpp
 * \brief
 */
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

class Conv3DBackpropInputV2ProtoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "Conv3DBackpropInputV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Conv3DBackpropInputV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(Conv3DBackpropInputV2ProtoTest, basic)
{
    vector<int64_t> strides({1, 3, 3, 1, 1});
    vector<int64_t> pads({2, 2, 2, 3, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NDHWC");

    vector<int64_t> input_size = {32, 16, 8, 8, 24};
    gert::StorageShape input_size_shape = {{32, 16, 8, 8, 24}, {32, 16, 8, 8, 24}};
    gert::StorageShape filter_shape = {{5, 7, 1, 24, 224}, {5, 7, 1, 24, 224}};
    gert::StorageShape out_backprop_shape = {{32, 6, 3, 8, 224}, {32, 6, 3, 8, 224}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder =
        gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NDHWC);
    tensor->SetStorageFormat(ge::FORMAT_NDHWC);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
        input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &filter_shape, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropInputV2")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[32, 16, 8, 8, 24]");
}

TEST_F(Conv3DBackpropInputV2ProtoTest, dynamic)
{
    vector<int64_t> strides({1, 3, 3, 1, 1});
    vector<int64_t> pads({2, 2, 2, 3, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NDHWC");

    vector<int64_t> input_size = {32, 16, 8, 8, 24};
    gert::StorageShape input_size_shape = {{32, 16, 8, 8, 24}, {32, 16, 8, 8, 24}};
    gert::StorageShape filter_shape = {{-1, 7, 1, 24, 224}, {5, 7, 1, 24, 224}};
    gert::StorageShape out_backprop_shape = {{-1, 6, 3, 8, 224}, {32, 6, 3, 8, 224}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder =
        gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NDHWC);
    tensor->SetStorageFormat(ge::FORMAT_NDHWC);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
        input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &filter_shape, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropInputV2")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, -1, -1, -1, -1]");
}

TEST_F(Conv3DBackpropInputV2ProtoTest, base_dtype)
{
    vector<int64_t> strides({1, 3, 3, 1, 1});
    vector<int64_t> pads({2, 2, 2, 3, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NDHWC");

    ge::DataType input_sizeDtype = ge::DT_INT64;
    ge::DataType filterDtype = ge::DT_FLOAT16;
    ge::DataType out_backpropDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .InputDataTypes({&input_sizeDtype, &filterDtype, &out_backpropDtype})
                      .OutputDataTypes({&yDtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropInputV2")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(Conv3DBackpropInputV2ProtoTest, 2d_extend_3d)
{
    vector<int64_t> strides({1, 3, 3, 1, 1});
    vector<int64_t> pads({2, 2, 2, 3, 0, 0});
    vector<int64_t> dilations({1, 1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("DHWCN");

    vector<int32_t> input_size = {8, 8, 24, 32};
    gert::StorageShape input_size_shape = {{8, 8, 24, 32}, {8, 8, 24, 32}};
    gert::StorageShape filter_shape = {{5, 7, 1, 24, 224}, {5, 7, 1, 24, 224}};
    gert::StorageShape out_backprop_shape = {{32, 6, 3, 8, 224}, {32, 6, 3, 8, 224}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder =
        gert::Tensor::CreateFollowing(input_size_shape.GetStorageShape().GetDimNum(), ge::DT_INT32, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.GetStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.GetOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NDHWC);
    tensor->SetStorageFormat(ge::FORMAT_NDHWC);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
        input_size.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({tensor, &filter_shape, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs(
                          {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                           {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN)
                      .Build();

    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropInputV2")->infer_shape;
    ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 8, 8, 24, 32]");
}