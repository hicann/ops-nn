/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/max_pool_grad_with_argmax_v3_proto.h"

namespace {
template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

class MaxPoolGradWithArgmaxV3Infer : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolGradWithArgmaxV3InferTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "max_pool_grad_with_argmax_v3_infer_test TearDown" << std::endl;
    }
};

TEST_F(MaxPoolGradWithArgmaxV3Infer, max_pool_grad_with_argmax_v3_infershape_test_01)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape gradShape = {{}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[4, 512, 16, 16]");
}

TEST_F(MaxPoolGradWithArgmaxV3Infer, max_pool_grad_with_argmax_v3_inferdtype_test_01)
{
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmaxV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(3, 1)
                                  .IrInstanceNum({1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref1})
                                  .OutputDataTypes({&output_ref})
                                  .NodeAttrs(
                                      {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
                                       {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
                                       {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                                       {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                                       {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({9, 5})},
                                       {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                       {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MaxPoolGradWithArgmaxV3Infer, max_pool_grad_with_argmax_v3_infershape_test_02)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape gradShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{4, 512, -1, 16}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[-1, -1, -1, -1]");
}

TEST_F(MaxPoolGradWithArgmaxV3Infer, max_pool_grad_with_argmax_v3_infershape_test_03)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGradWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape gradShape = {{4, -1, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{4, 512, 16, 16}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape, &gradShape, &indicesShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[-1, -1, -1, -1]");
}

} // namespace