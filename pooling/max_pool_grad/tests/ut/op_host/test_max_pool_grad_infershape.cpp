/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <sstream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

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

class MaxPoolGradInfer : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolGradInferTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPoolGradInferTest TearDown" << std::endl;
    }
};

TEST_F(MaxPoolGradInfer, max_pool_grad_infershape_fp32_valid_nchw)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGrad")->infer_shape;

    gert::StorageShape x1Shape = {{1, 1, 4, 4}, {}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {}};
    gert::StorageShape gradShape = {{1, 1, 2, 2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&x1Shape, &x2Shape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 1, 4, 4]");
}

TEST_F(MaxPoolGradInfer, max_pool_grad_infershape_fp16_same_nchw)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGrad")->infer_shape;

    gert::StorageShape x1Shape = {{2, 3, 8, 8}, {}};
    gert::StorageShape x2Shape = {{2, 3, 4, 4}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&x1Shape, &x2Shape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[2, 3, 8, 8]");
}

TEST_F(MaxPoolGradInfer, max_pool_grad_infershape_bf16_valid_nchw)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGrad")->infer_shape;

    gert::StorageShape x1Shape = {{1, 8, 16, 16}, {}};
    gert::StorageShape x2Shape = {{1, 8, 8, 8}, {}};
    gert::StorageShape gradShape = {{1, 8, 8, 8}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&x1Shape, &x2Shape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 8, 16, 16]");
}

TEST_F(MaxPoolGradInfer, max_pool_grad_infershape_k3_valid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGrad")->infer_shape;

    gert::StorageShape x1Shape = {{1, 1, 8, 8}, {}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {}};
    gert::StorageShape gradShape = {{1, 1, 2, 2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3, 3, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 3, 3, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&x1Shape, &x2Shape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 1, 8, 8]");
}

TEST_F(MaxPoolGradInfer, max_pool_grad_inferdatatype_fp32)
{
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(3, 1)
                                  .IrInstanceNum({1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .NodeAttrs(
                                      {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                       {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                       {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                                       {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MaxPoolGradInfer, max_pool_grad_inferdatatype_fp16)
{
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(3, 1)
                                  .IrInstanceNum({1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .NodeAttrs(
                                      {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                       {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 1})},
                                       {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                       {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

} // namespace
