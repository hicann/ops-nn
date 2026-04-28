/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Version 2.0 (the "License").
 * Please refer to the License for details. You may not use the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_quant_max_infershape.cpp
 * @brief Unit test for QuantMax infer shape
 */

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "ut_op_util.h"
#include "ut_op_common.h"

using namespace std;
using namespace ge;

class QuantMaxInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantMaxInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantMaxInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(QuantMaxInferShapeTest, quant_max_infershape_basic)
{
    // Basic infer shape test: output y should have same shape as input x
    // output amax should have shape [1]
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    // Create infer shape context
    gert::StorageShape x_shape({1024, 512}, {1024, 512});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1024, 512}, {1024, 512});
    gert::StorageShape amax_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    // Check output shapes
    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 2);
    EXPECT_EQ(output_y_shape->GetDim(0), 1024);
    EXPECT_EQ(output_y_shape->GetDim(1), 512);

    const gert::Shape* output_amax_shape = infer_shape_context->GetOutputShape(1);
    ASSERT_NE(output_amax_shape, nullptr);
    EXPECT_EQ(output_amax_shape->GetDimNum(), 1);
    EXPECT_EQ(output_amax_shape->GetDim(0), 1);
}

TEST_F(QuantMaxInferShapeTest, quant_max_infershape_1d_input)
{
    // Infer shape test for 1D input
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({4096}, {4096});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({4096}, {4096});
    gert::StorageShape amax_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 1);
    EXPECT_EQ(output_y_shape->GetDim(0), 4096);
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_fp32_fp8)
{
    // Infer datatype test: FP32 input -> FP8_E5M2 output (dstType=35)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_FLOAT;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // FLOAT8_E5M2
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_fp16_fp8_e5m2)
{
    // Infer datatype test: FP16 input -> FP8_E5M2 output (dstType=35)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_FLOAT16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // FLOAT8_E5M2
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_fp16_fp8_e4m3fn)
{
    // Infer datatype test: FP16 input -> FP8_E4M3FN output (dstType=36)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_FLOAT8_E4M3FN;
        ge::DataType amax_dtype = ge::DT_FLOAT16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)}  // FLOAT8_E4M3FN
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_fp16_hifloat8)
{
    // Infer datatype test: FP16 input -> HIFLOAT8 output (dstType=34)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_HIFLOAT8;
        ge::DataType amax_dtype = ge::DT_FLOAT16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Round")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}  // HIFLOAT8
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_bf16_fp8_e5m2)
{
    // Infer datatype test: BF16 input -> FP8_E5M2 output (dstType=35)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_BF16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_BF16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // FLOAT8_E5M2
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_bf16_fp8_e4m3fn)
{
    // Infer datatype test: BF16 input -> FP8_E4M3FN output (dstType=36)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_BF16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_FLOAT8_E4M3FN;
        ge::DataType amax_dtype = ge::DT_BF16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)}  // FLOAT8_E4M3FN
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_bf16_hifloat8)
{
    // Infer datatype test: BF16 input -> HIFLOAT8 output (dstType=34)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_BF16;
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_HIFLOAT8;
        ge::DataType amax_dtype = ge::DT_BF16;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Round")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}  // HIFLOAT8
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_invalid_scale_dtype)
{
    // Infer datatype failure test: scale dtype != FLOAT (should be FLOAT)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_FLOAT;
        ge::DataType scale_dtype = ge::DT_FLOAT16;  // Invalid: scale should be FLOAT
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_FLOAT;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // FLOAT8_E5M2
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_inferdatatype_invalid_x_dtype)
{
    // Infer datatype failure test: x dtype not in FLOAT/FLOAT16/BF16
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    if (infer_datatype_func != nullptr) {
        ge::DataType x_dtype = ge::DT_INT32;  // Invalid: x should be FLOAT/FLOAT16/BF16
        ge::DataType scale_dtype = ge::DT_FLOAT;
        ge::DataType y_dtype = ge::DT_FLOAT8_E5M2;
        ge::DataType amax_dtype = ge::DT_INT32;

        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&x_dtype, &scale_dtype})
                                  .OutputDataTypes({&y_dtype, &amax_dtype})
                                  .NodeAttrs({
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("Rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}  // FLOAT8_E5M2
                                  })
                                  .Build();

        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(QuantMaxInferShapeTest, quant_max_infershape_multi_dim)
{
    // Infer shape test for multi-dimension input (4D)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({2, 4, 8, 16}, {2, 4, 8, 16});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({2, 4, 8, 16}, {2, 4, 8, 16});
    gert::StorageShape amax_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 4);
    EXPECT_EQ(output_y_shape->GetDim(0), 2);
    EXPECT_EQ(output_y_shape->GetDim(1), 4);
    EXPECT_EQ(output_y_shape->GetDim(2), 8);
    EXPECT_EQ(output_y_shape->GetDim(3), 16);
}

TEST_F(QuantMaxInferShapeTest, quant_max_infershape_8dim)
{
    // Infer shape test for 8-dimension input (max supported rank)
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({1, 2, 2, 2, 2, 2, 2, 2}, {1, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({1, 2, 2, 2, 2, 2, 2, 2}, {1, 2, 2, 2, 2, 2, 2, 2});
    gert::StorageShape amax_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 8);
}

TEST_F(QuantMaxInferShapeTest, quant_max_infershape_empty_tensor)
{
    // Infer shape test for empty tensor input (shape [0])
    std::string op_type("QuantMax");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape({0}, {0});
    gert::StorageShape scale_shape({1}, {1});
    gert::StorageShape y_shape({0}, {0});
    gert::StorageShape amax_shape({1}, {1});

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1}, {1, 1})
                      .InputShapes({&x_shape, &scale_shape})
                      .OutputShapes({&y_shape, &amax_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* output_y_shape = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(output_y_shape, nullptr);
    EXPECT_EQ(output_y_shape->GetDimNum(), 1);
    EXPECT_EQ(output_y_shape->GetDim(0), 0);
}