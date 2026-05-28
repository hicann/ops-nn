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
 * \file test_avg_pool_v2_grad_infershape.cpp
 * \brief
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

static constexpr size_t INPUT_NUM = 2;
static constexpr size_t OUTPUT_NUM = 1;

class AvgPoolV2GradInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AvgPoolV2GradInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AvgPoolV2GradInfershape TearDown" << std::endl;
    }
};

static void ExecuteAvgPoolV2GradInfershapeTest(
    const std::vector<int32_t>& inputShapeData,
    const gert::StorageShape& gradShape,
    ge::Format gradFormat,
    ge::DataType gradDtype,
    const std::string& dataFormat,
    const std::string& paddingMode,
    const std::vector<int64_t>& ksize,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads,
    bool globalPooling,
    bool ceilMode,
    bool exclusive,
    int64_t divisorOverride,
    ge::graphStatus expectResult,
    const std::string& expectOutputStr)
{
    int64_t input0Size = static_cast<int64_t>(inputShapeData.size());
    gert::StorageShape input0Shape = {{input0Size}, {input0Size}};
    gert::StorageShape outputShape = {{}, {}};

    size_t totalSize = 0;
    auto tensorHolder =
        gert::Tensor::CreateFollowing(input0Size, ge::DT_INT32, totalSize);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    tensor->MutableStorageShape().AppendDim(input0Size);
    tensor->MutableOriginShape().AppendDim(input0Size);
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(
        tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), inputShapeData.data(),
        inputShapeData.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(INPUT_NUM, OUTPUT_NUM)
                      .IrInstanceNum({1, 1})
                      .InputShapes({tensor, const_cast<gert::StorageShape*>(&gradShape)})
                      .OutputShapes({&outputShape})
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"padding_mode", Ops::NN::AnyValue::CreateFrom<std::string>(paddingMode)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)},
                           {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(globalPooling)},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceilMode)},
                           {"exclusive", Ops::NN::AnyValue::CreateFrom<bool>(exclusive)},
                           {"divisor_override", Ops::NN::AnyValue::CreateFrom<int64_t>(divisorOverride)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gradDtype, gradFormat, gradFormat)
                      .NodeOutputTd(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolV2Grad")->infer_shape;
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), expectResult);
    if (expectResult == ge::GRAPH_SUCCESS) {
        auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*output), expectOutputStr);
    }
}

// ======================== Success Cases ========================

TEST_F(AvgPoolV2GradInfershape, nchw_4d_basic)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_SUCCESS,
        "[2, 3, 8, 8]");
}

TEST_F(AvgPoolV2GradInfershape, nchw_4d_valid)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 128, 14, 14},
        {{2, 128, 7, 7}, {2, 128, 7, 7}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT,
        "NCHW",
        "VALID",
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_SUCCESS,
        "[2, 128, 14, 14]");
}

TEST_F(AvgPoolV2GradInfershape, nchw_4d_same)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {4, 64, 32, 32},
        {{4, 64, 16, 16}, {4, 64, 16, 16}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT,
        "NCHW",
        "SAME",
        {1, 1, 3, 3},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        true, false, false, 0,
        ge::GRAPH_SUCCESS,
        "[4, 64, 32, 32]");
}

TEST_F(AvgPoolV2GradInfershape, nd_format_4d)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_ND,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_SUCCESS,
        "[2, 3, 8, 8]");
}

TEST_F(AvgPoolV2GradInfershape, nhwc_4d_basic)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 8, 8, 3},
        {{2, 4, 4, 3}, {2, 4, 4, 3}},
        ge::FORMAT_NHWC,
        ge::DT_FLOAT,
        "NHWC",
        "CALCULATED",
        {1, 2, 2, 1},
        {1, 2, 2, 1},
        {0, 0, 0, 0},
        false, false, false, 0,
        ge::GRAPH_SUCCESS,
        "[2, 8, 8, 3]");
}

TEST_F(AvgPoolV2GradInfershape, nd_3d_input)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {3, 8, 8},
        {{3, 8, 8}, {3, 8, 8}},
        ge::FORMAT_ND,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_SUCCESS,
        "[3, 8, 8]");
}

// ======================== Failure Cases ========================

TEST_F(AvgPoolV2GradInfershape, fail_invalid_padding_mode)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "INVALID",
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_nchw_ksize0_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {2, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_nchw_ksize1_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 2, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_nchw_strides0_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {2, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_nchw_strides1_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {1, 3, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_nhwc_ksize3_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 8, 8, 3},
        {{2, 4, 4, 3}, {2, 4, 4, 3}},
        ge::FORMAT_NHWC,
        ge::DT_FLOAT,
        "NHWC",
        "CALCULATED",
        {1, 2, 2, 2},
        {1, 2, 2, 1},
        {0, 0, 0, 0},
        false, false, false, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_nhwc_strides3_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 8, 8, 3},
        {{2, 4, 4, 3}, {2, 4, 4, 3}},
        ge::FORMAT_NHWC,
        ge::DT_FLOAT,
        "NHWC",
        "CALCULATED",
        {1, 2, 2, 1},
        {1, 2, 2, 3},
        {0, 0, 0, 0},
        false, false, false, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_invalid_ksize_length)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_invalid_strides_length)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_NCHW,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_invalid_dim_count_5d)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8, 4},
        {{2, 3, 8, 8, 4}, {2, 3, 8, 8, 4}},
        ge::FORMAT_ND,
        ge::DT_FLOAT,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2, 2},
        {1, 1, 2, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_invalid_dim_count_2d)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {8, 8},
        {{8, 8}, {8, 8}},
        ge::FORMAT_ND,
        ge::DT_FLOAT,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_invalid_grad_format)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 3, 8, 8},
        {{2, 3, 4, 4}, {2, 3, 4, 4}},
        ge::FORMAT_FRACTAL_NZ,
        ge::DT_FLOAT16,
        "NCHW",
        "CALCULATED",
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {0, 0, 0, 0},
        false, false, true, 0,
        ge::GRAPH_FAILED,
        "");
}

// ======================== NHWC Additional Failure Cases ========================

TEST_F(AvgPoolV2GradInfershape, fail_nhwc_ksize0_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 8, 8, 3},
        {{2, 4, 4, 3}, {2, 4, 4, 3}},
        ge::FORMAT_NHWC,
        ge::DT_FLOAT,
        "NHWC",
        "CALCULATED",
        {2, 2, 2, 1},
        {1, 2, 2, 1},
        {0, 0, 0, 0},
        false, false, false, 0,
        ge::GRAPH_FAILED,
        "");
}

TEST_F(AvgPoolV2GradInfershape, fail_nhwc_strides0_not_one)
{
    ExecuteAvgPoolV2GradInfershapeTest(
        {2, 8, 8, 3},
        {{2, 4, 4, 3}, {2, 4, 4, 3}},
        ge::FORMAT_NHWC,
        ge::DT_FLOAT,
        "NHWC",
        "CALCULATED",
        {1, 2, 2, 1},
        {3, 2, 2, 1},
        {0, 0, 0, 0},
        false, false, false, 0,
        ge::GRAPH_FAILED,
        "");
}

// ======================== InferDataType ========================

TEST_F(AvgPoolV2GradInfershape, infer_dtype_basic)
{
    ge::DataType gradDtype = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(INPUT_NUM, OUTPUT_NUM)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gradDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({nullptr, &gradDtype})
                      .OutputDataTypes({&outputDtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolV2Grad")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(AvgPoolV2GradInfershape, infer_dtype_float32)
{
    ge::DataType gradDtype = ge::DT_FLOAT;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(INPUT_NUM, OUTPUT_NUM)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({nullptr, &gradDtype})
                      .OutputDataTypes({&outputDtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolV2Grad")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
