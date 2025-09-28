/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "op_proto_test_util.h" // NOLINT
#include "nn_quantize.h"        // NOLINT
#include "graph/utils/op_desc_utils.h"
#include "common/utils/ut_op_common.h"

class DynamicQuantV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicQuantV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicQuantV2 TearDown" << std::endl;
    }
};

TEST_F(DynamicQuantV2, DynamicQuantV2_infershape_case_0)
{
    ge::op::DynamicQuantV2 op;
    op.UpdateInputDesc("x", create_desc({1, 128, 1024}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scales", create_desc({1024}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto output_y = op.GetOutputDesc(0);
    auto output_scale = op.GetOutputDesc(1);
    auto output_offset = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {1, 128, 1024};
    std::vector<int64_t> expected_scale_shape = {1, 128};
    std::vector<int64_t> expected_offset_shape = {1, 128};
    EXPECT_EQ(output_y.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale.GetShape().GetDims(), expected_scale_shape);
    EXPECT_EQ(output_offset.GetShape().GetDims(), expected_offset_shape);
}

TEST_F(DynamicQuantV2, DynamicQuantV2_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_smooth_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        ge::DataType output_offset_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(2, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_x_ref, &input_smooth_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref, &output_offset_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_offset_ref);
    }
}

TEST_F(DynamicQuantV2, DynamicQuantV2_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_smooth_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        ge::DataType output_offset_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(2, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", ge::AnyValue::CreateFrom<int64_t>(2)}})
                                  .InputDataTypes({&input_x_ref, &input_smooth_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref, &output_offset_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_offset_ref);
    }
}

TEST_F(DynamicQuantV2, DynamicQuantV2_infershape_case_pertensor)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_shape;

    if (infer_shape_func != nullptr) {
        gert::StorageShape x_shape = {{1, 128, 1024}, {1, 128, 1024}};
        gert::StorageShape smooth_shape = {{1024}, {1024}};
        gert::StorageShape y_shape = {{1, 128, 1024}, {1, 128, 1024}};
        gert::StorageShape scale_shape = {{1}, {1}};

        auto context_holder = gert::InferShapeContextFaker()
                                  .NodeIoNum(2, 3)
                                  .IrInstanceNum({1, 1})
                                  .InputShapes({&x_shape, &smooth_shape})
                                  .OutputShapes({&y_shape, &scale_shape, &scale_shape})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"dst_type", ge::AnyValue::CreateFrom<int64_t>(2)},
                                       {"is_symmetrical", ge::AnyValue::CreateFrom<bool>(false)},
                                       {"quant_mode", ge::AnyValue::CreateFrom<std::string>("pertensor")}})
                                  .Build();

        auto context = context_holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        auto shape1 = context->GetOutputShape(0);
        auto shape2 = context->GetOutputShape(1);
        auto shape3 = context->GetOutputShape(2);
        ASSERT_EQ(Shape2String(*shape1), "[1, 128, 1024]");
        ASSERT_EQ(Shape2String(*shape2), "[1]");
        ASSERT_EQ(Shape2String(*shape3), "[1]");
    }
}