/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_avg_pool_grad_infershape.cpp
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

class AvgPoolGradInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AvgPoolGradInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AvgPoolGradInfershape TearDown" << std::endl; }
};

static void ExecuteAvgPoolGradInfershapeTest(const std::vector<int32_t>& inputShapeData,
                                             const gert::StorageShape& gradShape, ge::Format gradFormat,
                                             ge::DataType gradDtype, const std::string& dataFormat,
                                             const std::string& padding, const std::vector<int64_t>& ksize,
                                             const std::vector<int64_t>& strides, ge::graphStatus expectResult,
                                             const std::string& expectOutputStr)
{
    int64_t input0Size = static_cast<int64_t>(inputShapeData.size());
    gert::StorageShape input0Shape = {{input0Size}, {input0Size}};
    gert::StorageShape outputShape = {{}, {}};

    size_t totalSize = 0;
    auto tensorHolder = gert::Tensor::CreateFollowing(input0Size, ge::DT_INT32, totalSize);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensorHolder.get());
    tensor->MutableStorageShape().AppendDim(input0Size);
    tensor->MutableOriginShape().AppendDim(input0Size);
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    (void)memcpy_s(tensor->GetData<uint8_t>(), totalSize - sizeof(gert::Tensor), inputShapeData.data(),
                   inputShapeData.size() * sizeof(int32_t));

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("AvgPoolGrad")
                      .NodeIoNum(INPUT_NUM, OUTPUT_NUM)
                      .IrInstanceNum({1, 1})
                      .InputShapes({tensor, const_cast<gert::StorageShape*>(&gradShape)})
                      .OutputShapes({&outputShape})
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gradDtype, gradFormat, gradFormat)
                      .NodeOutputTd(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolGrad")->infer_shape;
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), expectResult);
    if (expectResult == ge::GRAPH_SUCCESS) {
        auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*output), expectOutputStr);
    }
}

// ======================== Success Cases ========================

TEST_F(AvgPoolGradInfershape, nchw_4d_same)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "SAME", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_SUCCESS, "[2, 3, 8, 8]");
}

TEST_F(AvgPoolGradInfershape, nchw_4d_valid)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "VALID", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_SUCCESS, "[2, 3, 8, 8]");
}

TEST_F(AvgPoolGradInfershape, nhwc_4d_same)
{
    ExecuteAvgPoolGradInfershapeTest({2, 8, 8, 3}, {{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::FORMAT_NHWC, ge::DT_FLOAT, "NHWC",
                                     "SAME", {1, 2, 2, 1}, {1, 2, 2, 1}, ge::GRAPH_SUCCESS, "[2, 8, 8, 3]");
}

TEST_F(AvgPoolGradInfershape, nd_format_4d)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_ND, ge::DT_FLOAT, "NCHW",
                                     "SAME", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_SUCCESS, "[2, 3, 8, 8]");
}

TEST_F(AvgPoolGradInfershape, nd_3d_input)
{
    ExecuteAvgPoolGradInfershapeTest({3, 8, 8}, {{3, 8, 8}, {3, 8, 8}}, ge::FORMAT_ND, ge::DT_FLOAT, "NCHW", "SAME",
                                     {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_SUCCESS, "[3, 8, 8]");
}

TEST_F(AvgPoolGradInfershape, nhwc_4d_valid)
{
    ExecuteAvgPoolGradInfershapeTest({2, 8, 8, 3}, {{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::FORMAT_NHWC, ge::DT_FLOAT, "NHWC",
                                     "VALID", {1, 2, 2, 1}, {1, 2, 2, 1}, ge::GRAPH_SUCCESS, "[2, 8, 8, 3]");
}

// ======================== Failure Cases: OP_LOGE_FOR_INVALID_FORMAT ========================

TEST_F(AvgPoolGradInfershape, fail_invalid_grad_format)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT,
                                     "NCHW", "SAME", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

// ======================== Failure Cases: OP_LOGE_FOR_INVALID_VALUE_WITH_REASON (padding) ========================

TEST_F(AvgPoolGradInfershape, fail_invalid_padding)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "INVALID", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

// ======================== Failure Cases: OP_LOGE_FOR_INVALID_LISTSIZE (ksize) ========================

TEST_F(AvgPoolGradInfershape, fail_invalid_ksize_length)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "SAME", {2, 2}, {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

// ======================== Failure Cases: OP_LOGE_FOR_INVALID_LISTSIZE (strides) ========================

TEST_F(AvgPoolGradInfershape, fail_invalid_strides_length)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "SAME", {1, 1, 2, 2}, {2, 2}, ge::GRAPH_FAILED, "");
}

// ======================== Failure Cases: OP_LOGE_FOR_INVALID_VALUE_WITH_REASON (NCHW ksize/strides)
// ========================

TEST_F(AvgPoolGradInfershape, fail_nchw_ksize0_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "SAME", {2, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_nchw_ksize1_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "SAME", {1, 2, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_nchw_strides0_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "SAME", {1, 1, 2, 2}, {2, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_nchw_strides1_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8}, {{2, 3, 4, 4}, {2, 3, 4, 4}}, ge::FORMAT_NCHW, ge::DT_FLOAT, "NCHW",
                                     "SAME", {1, 1, 2, 2}, {1, 2, 2, 2}, ge::GRAPH_FAILED, "");
}

// ======================== Failure Cases: OP_LOGE_FOR_INVALID_VALUE_WITH_REASON (NHWC ksize/strides)
// ========================

TEST_F(AvgPoolGradInfershape, fail_nhwc_ksize0_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 8, 8, 3}, {{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::FORMAT_NHWC, ge::DT_FLOAT, "NHWC",
                                     "SAME", {2, 2, 2, 1}, {1, 2, 2, 1}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_nhwc_ksize3_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 8, 8, 3}, {{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::FORMAT_NHWC, ge::DT_FLOAT, "NHWC",
                                     "SAME", {1, 2, 2, 2}, {1, 2, 2, 1}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_nhwc_strides0_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 8, 8, 3}, {{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::FORMAT_NHWC, ge::DT_FLOAT, "NHWC",
                                     "SAME", {1, 2, 2, 1}, {2, 2, 2, 1}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_nhwc_strides3_not_one)
{
    ExecuteAvgPoolGradInfershapeTest({2, 8, 8, 3}, {{2, 4, 4, 3}, {2, 4, 4, 3}}, ge::FORMAT_NHWC, ge::DT_FLOAT, "NHWC",
                                     "SAME", {1, 2, 2, 1}, {1, 2, 2, 2}, ge::GRAPH_FAILED, "");
}

// ======================== Failure Cases: OP_LOGE_FOR_INVALID_SHAPEDIM ========================

TEST_F(AvgPoolGradInfershape, fail_invalid_dim_count_5d)
{
    ExecuteAvgPoolGradInfershapeTest({2, 3, 8, 8, 4}, {{2, 3, 8, 8, 4}, {2, 3, 8, 8, 4}}, ge::FORMAT_ND, ge::DT_FLOAT,
                                     "NCHW", "SAME", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_invalid_dim_count_2d)
{
    ExecuteAvgPoolGradInfershapeTest({8, 8}, {{8, 8}, {8, 8}}, ge::FORMAT_ND, ge::DT_FLOAT, "NCHW", "SAME",
                                     {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

TEST_F(AvgPoolGradInfershape, fail_invalid_dim_count_1d)
{
    ExecuteAvgPoolGradInfershapeTest({8}, {{8}, {8}}, ge::FORMAT_ND, ge::DT_FLOAT, "NCHW", "SAME", {1, 1, 2, 2},
                                     {1, 1, 2, 2}, ge::GRAPH_FAILED, "");
}

// ======================== Unknown Dim Values (Dynamic Shape) ========================

TEST_F(AvgPoolGradInfershape, infer_shape_unknown_dim_3d)
{
    ExecuteAvgPoolGradInfershapeTest({-1, -1, -1}, {{-1, -1, -1}, {-1, -1, -1}}, ge::FORMAT_ND, ge::DT_FLOAT, "NCHW",
                                     "SAME", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_SUCCESS, "[-1, -1, -1]");
}

TEST_F(AvgPoolGradInfershape, infer_shape_unknown_dim_4d)
{
    ExecuteAvgPoolGradInfershapeTest({-1, -1, -1, -1}, {{-1, -1, -1, -1}, {-1, -1, -1, -1}}, ge::FORMAT_NCHW,
                                     ge::DT_FLOAT, "NCHW", "SAME", {1, 1, 2, 2}, {1, 1, 2, 2}, ge::GRAPH_SUCCESS,
                                     "[-1, -1, -1, -1]");
}

// ======================== Null Context Tests ========================

TEST_F(AvgPoolGradInfershape, infer_shape_null_context)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolGrad")->infer_shape;
    ASSERT_EQ(inferShapeFunc(nullptr), ge::GRAPH_FAILED);
}

// ======================== InferDataType ========================

TEST_F(AvgPoolGradInfershape, infer_dtype_float16)
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

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolGrad")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(AvgPoolGradInfershape, infer_dtype_float32)
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

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolGrad")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST_F(AvgPoolGradInfershape, infer_dtype_null_context)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPoolGrad")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(nullptr), ge::GRAPH_FAILED);
}