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
#include "platform/platform_info.h"

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

class MaxPool3DGradInfer : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPool3DGradInferTest SetUp" << std::endl;
        fe::PlatformInfo platformInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void TearDownTestCase() { std::cout << "MaxPool3DGradInferTest TearDown" << std::endl; }
};

// ======================== Success Cases ========================

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fp32_valid)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[2, 4, 8, 8, 3]");
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fp16_same)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{1, 16, 7, 7, 8}, {}};
    gert::StorageShape yInShape = {{1, 16, 4, 4, 8}, {}};
    gert::StorageShape gradShape = {{1, 16, 4, 4, 8}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 16, 7, 7, 8]");
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_bf16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{1, 32, 4, 4, 4}, {}};
    gert::StorageShape yInShape = {{1, 32, 2, 2, 4}, {}};
    gert::StorageShape gradShape = {{1, 32, 2, 2, 4}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[1, 32, 4, 4, 4]");
}

// ======================== Unknown Shape / Unknown Rank ========================

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, -1, -1, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[-1, -1, -1, -1, -1]");
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

// ======================== Failure Cases ========================

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_invalid_ksize_length)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_ksize_zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_invalid_strides_length)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2, 2})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_strides_zero)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 0, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_invalid_padding)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("INVALID")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ======================== InferDataType ========================

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_inferdatatype_fp16)
{
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(3, 1)
                .IrInstanceNum({1, 1, 1})
                .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .InputDataTypes({&input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&output_ref})
                .NodeAttrs(
                    {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                     {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                     {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                     {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                     {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_inferdatatype_bf16)
{
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(3, 1)
                .IrInstanceNum({1, 1, 1})
                .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                .InputDataTypes({&input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&output_ref})
                .NodeAttrs(
                    {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                     {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                     {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                     {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                     {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

// ======================== Non-Ascend950 Platform Tests ========================

class MaxPool3DGradInferNon950 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPool3DGradInferNon950Test SetUp" << std::endl;
        fe::PlatformInfo platformInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        optiCompilationInfo.soc_version = "Ascend910B";
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void TearDownTestCase() { std::cout << "MaxPool3DGradInferNon950Test TearDown" << std::endl; }
};

TEST_F(MaxPool3DGradInferNon950, max_pool3d_grad_infershape_calculated_success)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("CALCULATED")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 1, 0, 1, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MaxPool3DGradInferNon950, max_pool3d_grad_infershape_ncdhw_calculated_success)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 8, 8, 4}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 4}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 4}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("CALCULATED")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MaxPool3DGradInferNon950, max_pool3d_grad_infershape_fail_negative_pads)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("CALCULATED")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, -1, 0, 1, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_ncdhw_success)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 8, 8, 4}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 4}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 4}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2, 2})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("VALID")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[2, 3, 8, 8, 4]");
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_invalid_pads_length)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("CALCULATED")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInferNon950, max_pool3d_grad_infershape_fail_invalid_padding_non950)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("INVALID")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_negative_ksize)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, -1, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(MaxPool3DGradInfer, max_pool3d_grad_infershape_fail_negative_strides)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->infer_shape;

    gert::StorageShape xShape = {{2, 4, 8, 8, 3}, {}};
    gert::StorageShape yInShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape gradShape = {{2, 3, 4, 4, 3}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 2, 2, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, -2, 2, 2, 1})},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .InputShapes({&xShape, &yInShape, &gradShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

} // namespace
