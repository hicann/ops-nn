/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/dequant_situ_quant_tiling.h"
#include "../../../op_graph/dequant_situ_quant_proto.h"

using namespace std;
using namespace ge;

class DequantSituQuantInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DequantSituQuantInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DequantSituQuantInferShapeTest TearDown" << std::endl; }
};

// Test INT8 infershape: x=[4,8,64] → y=[4,8,32], scale=[4,8]
TEST_F(DequantSituQuantInferShapeTest, tiling_infershape_int8)
{
    gert::StorageShape x_shape = {{4, 8, 64}, {4, 8, 64}};
    gert::StorageShape ws_shape = {{64}, {64}};
    gert::StorageShape y_shape = {{4, 8, 32}, {4, 8, 32}};
    gert::StorageShape y_scale_shape = {{4, 8}, {4, 8}};

    std::string op_type("DequantSituQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 0, 0, 0, 0, 0}, {1, 1})
                      .InputShapes({&x_shape, &ws_shape})
                      .OutputShapes({&y_shape, &y_scale_shape})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* y_out = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(y_out, nullptr);
    ASSERT_EQ(y_out->GetDimNum(), 3);
    ASSERT_EQ(y_out->GetDim(0), 4);
    ASSERT_EQ(y_out->GetDim(1), 8);
    ASSERT_EQ(y_out->GetDim(2), 32);

    const gert::Shape* scale_out = infer_shape_context->GetOutputShape(1);
    ASSERT_NE(scale_out, nullptr);
    ASSERT_EQ(scale_out->GetDimNum(), 2);
    ASSERT_EQ(scale_out->GetDim(0), 4);
    ASSERT_EQ(scale_out->GetDim(1), 8);

    // InferDataType
    auto infer_dtype_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_datatype;
    ASSERT_NE(infer_dtype_func, nullptr);

    ge::DataType x_dtype = ge::DT_INT8;
    ge::DataType ws_dtype = ge::DT_FLOAT;

    auto dtype_holder = gert::InferDataTypeContextFaker()
                            .IrInputNum(7)
                            .NodeIoNum(7, 2)
                            .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .InputDataTypes({&x_dtype, &ws_dtype})
                            .Build();

    auto dtype_context = dtype_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(infer_dtype_func(dtype_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(dtype_context->GetOutputDataType(0), ge::DT_INT8);
    ASSERT_EQ(dtype_context->GetOutputDataType(1), ge::DT_FLOAT);
}

// Test INT32 infershape: x=[32, 6144] → y=[32, 3072], scale=[32]
TEST_F(DequantSituQuantInferShapeTest, tiling_infershape_int32)
{
    gert::StorageShape x_shape = {{32, 6144}, {32, 6144}};
    gert::StorageShape ws_shape = {{6144}, {6144}};
    gert::StorageShape act_shape = {{32}, {32}};
    gert::StorageShape y_shape = {{32, 3072}, {32, 3072}};
    gert::StorageShape y_scale_shape = {{32}, {32}};

    std::string op_type("DequantSituQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 0, 0, 0, 0}, {1, 1})
                      .InputShapes({&x_shape, &ws_shape, &act_shape})
                      .OutputShapes({&y_shape, &y_scale_shape})
                      .Build();

    auto infer_shape_context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(infer_shape_func(infer_shape_context), ge::GRAPH_SUCCESS);

    const gert::Shape* y_out = infer_shape_context->GetOutputShape(0);
    ASSERT_NE(y_out, nullptr);
    ASSERT_EQ(y_out->GetDimNum(), 2);
    ASSERT_EQ(y_out->GetDim(0), 32);
    ASSERT_EQ(y_out->GetDim(1), 3072);

    const gert::Shape* scale_out = infer_shape_context->GetOutputShape(1);
    ASSERT_NE(scale_out, nullptr);
    ASSERT_EQ(scale_out->GetDimNum(), 1);
    ASSERT_EQ(scale_out->GetDim(0), 32);
}
