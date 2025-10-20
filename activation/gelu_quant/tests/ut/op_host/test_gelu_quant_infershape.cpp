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
#include "infershape_test_util.h" // NOLINT      // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/gelu_quant_proto.h"

class GeluQuant : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GeluQuant SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GeluQuant TearDown" << std::endl;
    }
};

TEST_F(GeluQuant, GeluQuant_infershape_case_0)
{
    ge::op::GeluQuant op;
    op.UpdateInputDesc("x", create_desc({1, 128, 6912}, ge::DT_FLOAT16));
    op.UpdateInputDesc("input_scale", create_desc({6912}, ge::DT_FLOAT16));
    op.UpdateInputDesc("input_offset", create_desc({6912}, ge::DT_FLOAT16));

    op.SetAttr("approximate", "none");
    op.SetAttr("quant_mode", "dynamic");
    op.SetAttr("dst_type", 2);
    op.SetAttr("quant_mode", "rint");

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto output_y = op.GetOutputDesc(0);
    auto output_scale = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {1, 128, 6912};
    std::vector<int64_t> expected_scale_shape = {1, 128};
    EXPECT_EQ(output_y.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale.GetShape().GetDims(), expected_scale_shape);
}

TEST_F(GeluQuant, GeluQuant_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GeluQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_scale_ref = ge::DT_FLOAT16;
        ge::DataType input_offset_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(3, 2)
                                  .IrInstanceNum({1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"approximate", Ops::NN::AnyValue::CreateFrom<string>("none")},
                                       {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                       {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                                  .InputDataTypes({&input_x_ref, &input_scale_ref, &input_offset_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(GeluQuant, GeluQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GeluQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GeluQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_scale_ref = ge::DT_FLOAT16;
        ge::DataType input_offset_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_HIFLOAT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(3, 2)
                                  .IrInstanceNum({1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"approximate", Ops::NN::AnyValue::CreateFrom<string>("none")},
                                       {"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                                       {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                                  .InputDataTypes({&input_x_ref, &input_scale_ref, &input_offset_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}