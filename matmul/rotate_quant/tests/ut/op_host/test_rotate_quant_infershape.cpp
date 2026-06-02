/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_host/op_tiling/rotate_quant_tiling.h"

using namespace ge;
using namespace std;

constexpr int32_t DIMENSION_2 = 2;
constexpr int32_t CASE1_X_M = 16;
constexpr int32_t CASE1_X_N = 256;
constexpr int32_t CASE0_X_M = 4;
constexpr int32_t CASE0_X_N = 64;

class RotateQuantInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RotateQuantInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RotateQuantInferShape TearDown" << std::endl;
    }
};

TEST_F(RotateQuantInferShape, RotateQuant_infershape_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 64}, {4, 64}};
    gert::StorageShape rot_shape = {{64, 64}, {64, 64}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT8))},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), CASE0_X_M);
    EXPECT_EQ(yShape->GetDim(1), CASE0_X_N);

    EXPECT_EQ(scaleShape->GetDimNum(), 1);
    EXPECT_EQ(scaleShape->GetDim(0), CASE0_X_M);
}

TEST_F(RotateQuantInferShape, RotateQuant_infershape_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{16, 256}, {16, 256}};
    gert::StorageShape rot_shape = {{256, 256}, {256, 256}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape scale_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &rot_shape})
                      .OutputShapes({&y_shape, &scale_shape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT8))},
                                  {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto yShape = context->GetOutputShape(0);
    auto scaleShape = context->GetOutputShape(1);
    ASSERT_NE(yShape, nullptr);
    ASSERT_NE(scaleShape, nullptr);

    EXPECT_EQ(yShape->GetDimNum(), DIMENSION_2);
    EXPECT_EQ(yShape->GetDim(0), CASE1_X_M);
    EXPECT_EQ(yShape->GetDim(1), CASE1_X_N);

    EXPECT_EQ(scaleShape->GetDimNum(), 1);
    EXPECT_EQ(scaleShape->GetDim(0), CASE1_X_M);
}

// InferDataType: INT8输出
TEST_F(RotateQuantInferShape, RotateQuant_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_BF16;
        ge::DataType input_rot_ref = ge::DT_BF16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .IrInstanceNum({1, 1})
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_INT8))},
                                       {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f)}})
                                  .InputDataTypes({&input_x_ref, &input_rot_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}
