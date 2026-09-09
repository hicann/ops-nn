/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv2d_backprop_filter_v2_infershape.cpp
 * \brief InferShapeForConv2DBackpropFilter / InferDataTypeForConv2DBackpropFilter (shared by
 *        Conv2DBackpropFilterV2 and Conv2DBackpropFilterV3) are not registered by IMPL_OP in this
 *        repo, so the functions are called directly with faker contexts.
 */
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../../../../common/op_host/conv_backprop_infershape.h"

class Conv2DBackpropFilterV2ProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Conv2DBackpropFilterV2 Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Conv2DBackpropFilterV2 Proto Test TearDown" << std::endl; }
};

// cover conv_backprop_infershape.cpp InferShapeForConv2DBackpropFilter basic path (L94-L107):
// INT64 filter_size, from_depthwise=false returns directly without resetting y shape
TEST_F(Conv2DBackpropFilterV2ProtoTest, basic)
{
    vector<int64_t> strides({1, 1, 1, 1});
    vector<int64_t> pads({1, 1, 1, 1});
    vector<int64_t> dilations({1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCHW");
    bool enable_hf32 = false;
    bool from_depthwise = false;

    vector<int64_t> filter_sizes = {3, 3, 128, 256};
    gert::StorageShape filter_sizes_shape = {{3, 3, 128, 256}, {3, 3, 128, 256}};
    gert::StorageShape x_shape = {{2, 128, 16, 16}, {2, 128, 16, 16}};
    gert::StorageShape out_backprop_shape = {{2, 256, 16, 16}, {2, 256, 16, 16}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(filter_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(filter_sizes_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(filter_sizes_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCHW);
    tensor->SetStorageFormat(ge::FORMAT_NCHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), filter_sizes.data(),
                   filter_sizes.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, tensor, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enable_hf32)},
                                  {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(from_depthwise)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();

    ASSERT_EQ(Ops::NN::Conv::InferShapeForConv2DBackpropFilter(holder.GetContext<gert::InferShapeContext>()),
              ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[3, 3, 128, 256]");
}

// cover conv_backprop_infershape.cpp InferShapeForConv2DBackpropFilter depthwise reset branch with
// NCHW output format (L109-L120): from_depthwise=true merges dim0*dim1 into dim0 and sets dim1 to 1
TEST_F(Conv2DBackpropFilterV2ProtoTest, depthwise_nchw)
{
    vector<int64_t> strides({1, 1, 1, 1});
    vector<int64_t> pads({1, 1, 1, 1});
    vector<int64_t> dilations({1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCHW");
    bool enable_hf32 = false;
    bool from_depthwise = true;

    vector<int64_t> filter_sizes = {3, 3, 128, 256};
    gert::StorageShape filter_sizes_shape = {{3, 3, 128, 256}, {3, 3, 128, 256}};
    gert::StorageShape x_shape = {{2, 128, 16, 16}, {2, 128, 16, 16}};
    gert::StorageShape out_backprop_shape = {{2, 256, 16, 16}, {2, 256, 16, 16}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(filter_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(filter_sizes_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(filter_sizes_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCHW);
    tensor->SetStorageFormat(ge::FORMAT_NCHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), filter_sizes.data(),
                   filter_sizes.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, tensor, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enable_hf32)},
                                  {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(from_depthwise)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();

    ASSERT_EQ(Ops::NN::Conv::InferShapeForConv2DBackpropFilter(holder.GetContext<gert::InferShapeContext>()),
              ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[9, 1, 128, 256]");
}

// cover conv_backprop_infershape.cpp InferShapeForConv2DBackpropFilter depthwise reset branch with
// HWCN output format (L121-L123): from_depthwise=true merges dim2*dim3 into dim3 and sets dim2 to 1
TEST_F(Conv2DBackpropFilterV2ProtoTest, depthwise_hwcn)
{
    vector<int64_t> strides({1, 1, 1, 1});
    vector<int64_t> pads({1, 1, 1, 1});
    vector<int64_t> dilations({1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("HWCN");
    bool enable_hf32 = false;
    bool from_depthwise = true;

    vector<int64_t> filter_sizes = {3, 3, 128, 256};
    gert::StorageShape filter_sizes_shape = {{3, 3, 128, 256}, {3, 3, 128, 256}};
    gert::StorageShape x_shape = {{16, 16, 128, 2}, {16, 16, 128, 2}};
    gert::StorageShape out_backprop_shape = {{16, 16, 256, 2}, {16, 16, 256, 2}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(filter_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT64,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(filter_sizes_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(filter_sizes_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_HWCN);
    tensor->SetStorageFormat(ge::FORMAT_HWCN);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), filter_sizes.data(),
                   filter_sizes.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, tensor, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enable_hf32)},
                                  {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(from_depthwise)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_HWCN)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_HWCN, ge::FORMAT_HWCN)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_HWCN)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_HWCN, ge::FORMAT_HWCN)
                      .Build();

    ASSERT_EQ(Ops::NN::Conv::InferShapeForConv2DBackpropFilter(holder.GetContext<gert::InferShapeContext>()),
              ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[3, 3, 1, 32768]");
}

// cover conv_backprop_infershape.cpp InferShapeForConvBackprop DT_INT32 branch (L71-L75):
// filter_size tensor with DT_INT32 data also fills y shape
TEST_F(Conv2DBackpropFilterV2ProtoTest, filter_size_int32)
{
    vector<int64_t> strides({1, 1, 1, 1});
    vector<int64_t> pads({1, 1, 1, 1});
    vector<int64_t> dilations({1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCHW");
    bool enable_hf32 = false;
    bool from_depthwise = false;

    vector<int32_t> filter_sizes = {3, 3, 128, 256};
    gert::StorageShape filter_sizes_shape = {{3, 3, 128, 256}, {3, 3, 128, 256}};
    gert::StorageShape x_shape = {{2, 128, 16, 16}, {2, 128, 16, 16}};
    gert::StorageShape out_backprop_shape = {{2, 256, 16, 16}, {2, 256, 16, 16}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(filter_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_INT32,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(filter_sizes_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(filter_sizes_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCHW);
    tensor->SetStorageFormat(ge::FORMAT_NCHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), filter_sizes.data(),
                   filter_sizes.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, tensor, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enable_hf32)},
                                  {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(from_depthwise)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();

    ASSERT_EQ(Ops::NN::Conv::InferShapeForConv2DBackpropFilter(holder.GetContext<gert::InferShapeContext>()),
              ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[3, 3, 128, 256]");
}

// cover conv_backprop_infershape.cpp InferShapeForConvBackprop invalid dtype branch (L82-L86):
// filter_size dtype DT_FLOAT is neither DT_INT32 nor DT_INT64, infer shape fails
TEST_F(Conv2DBackpropFilterV2ProtoTest, filter_size_invalid_dtype)
{
    vector<int64_t> strides({1, 1, 1, 1});
    vector<int64_t> pads({1, 1, 1, 1});
    vector<int64_t> dilations({1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCHW");
    bool enable_hf32 = false;
    bool from_depthwise = false;

    gert::StorageShape filter_sizes_shape = {{3, 3, 128, 256}, {3, 3, 128, 256}};
    gert::StorageShape x_shape = {{2, 128, 16, 16}, {2, 128, 16, 16}};
    gert::StorageShape out_backprop_shape = {{2, 256, 16, 16}, {2, 256, 16, 16}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(filter_sizes_shape.GetStorageShape().GetDimNum(), ge::DT_FLOAT,
                                                       total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(filter_sizes_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(filter_sizes_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCHW);
    tensor->SetStorageFormat(ge::FORMAT_NCHW);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, tensor, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enable_hf32)},
                                  {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(from_depthwise)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();

    ASSERT_EQ(Ops::NN::Conv::InferShapeForConv2DBackpropFilter(holder.GetContext<gert::InferShapeContext>()),
              ge::GRAPH_FAILED);
}

// cover conv_backprop_infershape.cpp IsConstTensor empty tensor branch (L22-L23) and
// InferShapeForConvBackprop unknown output branch (L63-L67): filter_size tensor without data
// (addr is null) is not a const tensor, all y dims are set to -1
TEST_F(Conv2DBackpropFilterV2ProtoTest, filter_size_empty_tensor)
{
    vector<int64_t> strides({1, 1, 1, 1});
    vector<int64_t> pads({1, 1, 1, 1});
    vector<int64_t> dilations({1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCHW");
    bool enable_hf32 = false;
    bool from_depthwise = false;

    gert::StorageShape filter_sizes_shape = {{3, 3, 128, 256}, {3, 3, 128, 256}};
    gert::StorageShape x_shape = {{2, 128, 16, 16}, {2, 128, 16, 16}};
    gert::StorageShape out_backprop_shape = {{2, 256, 16, 16}, {2, 256, 16, 16}};
    gert::StorageShape output_shape = {{}, {}};

    size_t total_size = 0;
    auto tensor_holder = gert::Tensor::CreateFollowing(0, ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(filter_sizes_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(filter_sizes_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(ge::FORMAT_NCHW);
    tensor->SetStorageFormat(ge::FORMAT_NCHW);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, tensor, &out_backprop_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enable_hf32)},
                                  {"from_depthwise", Ops::NN::AnyValue::CreateFrom<bool>(from_depthwise)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();

    ASSERT_EQ(Ops::NN::Conv::InferShapeForConv2DBackpropFilter(holder.GetContext<gert::InferShapeContext>()),
              ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, -1, -1, -1]");
}

// cover conv_backprop_infershape.cpp InferDataTypeForConv2DBackpropFilter (L130-L138):
// output dtype of Conv2DBackpropFilter is always DT_FLOAT
TEST_F(Conv2DBackpropFilterV2ProtoTest, base_dtype)
{
    vector<int64_t> strides({1, 1, 1, 1});
    vector<int64_t> pads({1, 1, 1, 1});
    vector<int64_t> dilations({1, 1, 1, 1});
    int64_t groups = 1;
    string data_format("NCHW");

    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType filter_sizeDtype = ge::DT_INT64;
    ge::DataType out_backpropDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .InputDataTypes({&xDtype, &filter_sizeDtype, &out_backpropDtype})
                      .OutputDataTypes({&yDtype})
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(Ops::NN::Conv::InferDataTypeForConv2DBackpropFilter(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
