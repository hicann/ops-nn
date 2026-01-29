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
#include "../../../op_graph/max_pool_with_argmax_v3_proto.h"

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

class MaxPoolWithArgmaxV3Infer : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolWithArgmaxV3InferTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPoolWithArgmaxV3InferTest TearDown" << std::endl;
    }
};

TEST_F(MaxPoolWithArgmaxV3Infer, maxpool_with_argmax_v3_infershape_test_1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{4, 512, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[4, 512, 8, 8]");
    ASSERT_EQ(Shape2String(*indices), "[4, 512, 8, 8]");
}

TEST_F(MaxPoolWithArgmaxV3Infer, maxpool_with_argmax_v3_infershape_test_2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{5, 256, 144, 589}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({9, 5})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[5, 125, 46, 589]");
    ASSERT_EQ(Shape2String(*indices), "[5, 125, 46, 589]");
}

TEST_F(MaxPoolWithArgmaxV3Infer, maxpool_with_argmax_v3_infershape_test_3)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{256, 144, 589}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 3})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({9, 5})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[125, 46, 589]");
    ASSERT_EQ(Shape2String(*indices), "[125, 46, 589]");
}

TEST_F(MaxPoolWithArgmaxV3Infer, maxpool_with_argmax_v3_infershape_test_4)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{1, 3, -1, -1}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[1, 3, -1, -1]");
    ASSERT_EQ(Shape2String(*indices), "[1, 3, -1, -1]");
}

TEST_F(MaxPoolWithArgmaxV3Infer, maxpool_with_argmax_v3_infershape_test_5)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolWithArgmaxV3")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape indicesShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NHWC, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({2, 2})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indicesShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* indices = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[-2]");
    ASSERT_EQ(Shape2String(*indices), "[-2]");
}

TEST_F(MaxPoolWithArgmaxV3Infer, max_pool_with_argmax_v3_inferdtype_success_01)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolWithArgmaxV3")->infer_datatype;

    ge::DataType x_dtype = ge::DT_FLOAT16;
    ge::DataType y_dtype = ge::DT_FLOAT16;
    ge::DataType argmax_dtype = ge::DT_INT32;
    ge::DataType expect_output_dtype = ge::DT_FLOAT16;

    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({
                          1,
                      })
                      .InputDataTypes({&x_dtype})
                      .OutputDataTypes({&y_dtype, &argmax_dtype})
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->GetOutputDataType(0), expect_output_dtype);
}
} // namespace
