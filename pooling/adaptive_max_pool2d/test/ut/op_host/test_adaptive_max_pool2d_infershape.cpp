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

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "../../../op_graph/adaptive_max_pool2d_proto.h"

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

class AdaptiveMaxPool2dInfer : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveMaxPool2dInferTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveMaxPool2dInferTest TearDown" << std::endl; }
};

// A5 platform (Ascend950): dtype inference uses the "argmax_dtype" attribute.
class AdaptiveMaxPool2dInferA5 : public testing::Test {
protected:
    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950PR_9589";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

// A2/A3 platform: dtype inference ignores "argmax_dtype" and hardcodes the argmax output dtype as INT64
// (infershape2.0 cannot read the graph-annotated output dtype, so it aligns with infershape1.0's default behavior).
class AdaptiveMaxPool2dInferA2A3 : public testing::Test {
protected:
    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

TEST_F(AdaptiveMaxPool2dInfer, infershape_null_context)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;
    ASSERT_EQ(inferShapeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_success_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* argmax = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[2, 3, 4, 4]");
    ASSERT_EQ(Shape2String(*argmax), "[2, 3, 4, 4]");
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_success_3d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{3, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* argmax = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Shape2String(*output), "[3, 4, 4]");
    ASSERT_EQ(Shape2String(*argmax), "[3, 4, 4]");
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_unknown_shape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{-1}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_invalid_dim_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{1, 2}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_invalid_dim_5d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{1, 2, 3, 4, 5}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_invalid_output_size_len)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4})}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(AdaptiveMaxPool2dInfer, infershape_missing_output_size)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_shape;

    gert::StorageShape xShape = {{2, 3, 16, 16}, {}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape argmaxShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1, 2})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(AdaptiveMaxPool2dInfer, inferdatatype_null_context)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(nullptr), ge::GRAPH_FAILED);
}

TEST_F(AdaptiveMaxPool2dInferA5, inferdatatype_a5_argmax_dtype_int32)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_datatype;

    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    ge::DataType argmaxDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputDataTypes({&xDtype})
                      .OutputDataTypes({&yDtype, &argmaxDtype})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})},
                                  {"argmax_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT32);
}

TEST_F(AdaptiveMaxPool2dInferA5, inferdatatype_a5_argmax_dtype_int64)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_datatype;

    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    ge::DataType argmaxDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputDataTypes({&xDtype})
                      .OutputDataTypes({&yDtype, &argmaxDtype})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})},
                                  {"argmax_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(9)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT64);
}

TEST_F(AdaptiveMaxPool2dInferA5, inferdatatype_a5_missing_argmax_dtype)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_datatype;

    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    ge::DataType argmaxDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputDataTypes({&xDtype})
                      .OutputDataTypes({&yDtype, &argmaxDtype})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_FAILED);
}

TEST_F(AdaptiveMaxPool2dInferA2A3, inferdatatype_a2a3_argmax_hardcoded_int64)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveMaxPool2d")->infer_datatype;

    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_UNDEFINED;
    ge::DataType argmaxDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputDataTypes({&xDtype})
                      .OutputDataTypes({&yDtype, &argmaxDtype})
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({4, 4})}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    // A2/A3: argmax dtype is hardcoded as INT64 regardless of the graph-annotated output desc dtype.
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_INT64);
}
} // namespace
