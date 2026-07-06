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
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"

class ScatterUpdateInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScatterUpdateInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScatterUpdateInferShapeTest TearDown" << std::endl; }
};

TEST_F(ScatterUpdateInferShapeTest, infer_shape_2d_float)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape var_shape = {{8, 4}, {8, 4}};
    gert::StorageShape indices_shape = {{5}, {5}};
    gert::StorageShape updates_shape = {{5, 4}, {5, 4}};
    gert::StorageShape output_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInputNum(3)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&var_shape, &indices_shape, &updates_shape})
                      .OutputShapes({&output_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 8);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 4);
}

TEST_F(ScatterUpdateInferShapeTest, infer_shape_2d_int32)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape var_shape = {{10, 6}, {10, 6}};
    gert::StorageShape indices_shape = {{3}, {3}};
    gert::StorageShape updates_shape = {{3, 6}, {3, 6}};
    gert::StorageShape output_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInputNum(3)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&var_shape, &indices_shape, &updates_shape})
                      .OutputShapes({&output_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 2);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 10);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 6);
}

TEST_F(ScatterUpdateInferShapeTest, infer_shape_1d_float16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape var_shape = {{16}, {16}};
    gert::StorageShape indices_shape = {{4}, {4}};
    gert::StorageShape updates_shape = {{4}, {4}};
    gert::StorageShape output_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInputNum(3)
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&var_shape, &indices_shape, &updates_shape})
                      .OutputShapes({&output_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 1);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 16);
}

TEST_F(ScatterUpdateInferShapeTest, infer_shape_empty_var)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape var_shape = {{0, 4}, {0, 4}};
    gert::StorageShape indices_shape = {{0}, {0}};
    gert::StorageShape updates_shape = {{0, 4}, {0, 4}};
    gert::StorageShape output_shape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInputNum(3)
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&var_shape, &indices_shape, &updates_shape})
                      .OutputShapes({&output_shape})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 4);
}

TEST_F(ScatterUpdateInferShapeTest, infer_datatype_float)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);

    ge::DataType var_type = ge::DT_FLOAT;
    ge::DataType indices_type = ge::DT_INT32;
    ge::DataType updates_type = ge::DT_FLOAT;
    ge::DataType output_type = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInputNum(3)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&var_type, &indices_type, &updates_type})
                              .OutputDataTypes({&output_type})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST_F(ScatterUpdateInferShapeTest, infer_datatype_float16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);

    ge::DataType var_type = ge::DT_FLOAT16;
    ge::DataType indices_type = ge::DT_INT64;
    ge::DataType updates_type = ge::DT_FLOAT16;
    ge::DataType output_type = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInputNum(3)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&var_type, &indices_type, &updates_type})
                              .OutputDataTypes({&output_type})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(ScatterUpdateInferShapeTest, infer_datatype_bf16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);

    ge::DataType var_type = ge::DT_BF16;
    ge::DataType indices_type = ge::DT_INT32;
    ge::DataType updates_type = ge::DT_BF16;
    ge::DataType output_type = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInputNum(3)
                              .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&var_type, &indices_type, &updates_type})
                              .OutputDataTypes({&output_type})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST_F(ScatterUpdateInferShapeTest, infer_datatype_int32)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);

    ge::DataType var_type = ge::DT_INT32;
    ge::DataType indices_type = ge::DT_INT32;
    ge::DataType updates_type = ge::DT_INT32;
    ge::DataType output_type = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInputNum(3)
                              .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&var_type, &indices_type, &updates_type})
                              .OutputDataTypes({&output_type})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST_F(ScatterUpdateInferShapeTest, infer_datatype_int64)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ScatterUpdate")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);

    ge::DataType var_type = ge::DT_INT64;
    ge::DataType indices_type = ge::DT_INT64;
    ge::DataType updates_type = ge::DT_INT64;
    ge::DataType output_type = ge::DT_UNDEFINED;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(3, 1)
                              .IrInputNum(3)
                              .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&var_type, &indices_type, &updates_type})
                              .OutputDataTypes({&output_type})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
}