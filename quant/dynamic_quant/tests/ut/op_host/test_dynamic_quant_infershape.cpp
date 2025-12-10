/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/dynamic_quant_proto.h"

class DynamicQuant : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicQuant SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicQuant TearDown" << std::endl;
    }
};

TEST_F(DynamicQuant, DynamicQuant_infershape_case_0)
{
    ge::op::DynamicQuant op;
    op.UpdateInputDesc("x", create_desc({1, 128, 1024}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scales", create_desc({1024}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto output_y = op.GetOutputDesc(0);
    auto output_scale = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {1, 128, 1024};
    std::vector<int64_t> expected_scale_shape = {1, 128};
    EXPECT_EQ(output_y.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale.GetShape().GetDims(), expected_scale_shape);
}

TEST_F(DynamicQuant, DynamicQuant_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_smooth_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_x_ref})
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

TEST_F(DynamicQuant, DynamicQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_smooth_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                                  .InputDataTypes({&input_x_ref, &input_smooth_ref})
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