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
 * \file test_dynamic_block_mx_quant_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <vector>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/dynamic_block_mx_quant_proto.h"

using namespace ge;

class DynamicBlockMxQuantTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_block_mx_quant test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "dynamic_block_mx_quant test TearDown" << std::endl;
    }
};

TEST_F(DynamicBlockMxQuantTest, DynamicBlockMxQuant_InferShape_case_1) {
    ge::op::DynamicBlockMxQuant quant_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({128, 128});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);
    quant_op.UpdateInputDesc("x", XDesc);

    Runtime2TestParam param{{"round_mode", "dst_type"}};
    EXPECT_EQ(InferShapeTest(quant_op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = quant_op.GetOutputDesc(0);
    auto output_scale1_desc = quant_op.GetOutputDesc(1);
    auto output_scale2_desc = quant_op.GetOutputDesc(2);
    std::vector<int64_t> expected_y1_shape = {128, 128};
    std::vector<int64_t> expected_scale1_shape = {128, 2, 2};
    std::vector<int64_t> expected_scale2_shape = {2, 128, 2};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_scale1_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_scale2_shape);
}

TEST_F(DynamicBlockMxQuantTest, DynamicBlockMxQuant_InferShape_case_2) {
    ge::op::DynamicBlockMxQuant quant_op;
    ge::TensorDesc XDesc;
    ge::Shape xShape({32, 128, 128});
    XDesc.SetDataType(ge::DT_FLOAT16);
    XDesc.SetShape(xShape);
    XDesc.SetOriginShape(xShape);
    quant_op.UpdateInputDesc("x", XDesc);

    Runtime2TestParam param{{"round_mode", "dst_type"}};
    EXPECT_EQ(InferShapeTest(quant_op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = quant_op.GetOutputDesc(0);
    auto output_scale1_desc = quant_op.GetOutputDesc(1);
    auto output_scale2_desc = quant_op.GetOutputDesc(2);
    std::vector<int64_t> expected_y1_shape = {32, 128, 128};
    std::vector<int64_t> expected_scale1_shape = {32, 128, 2, 2};
    std::vector<int64_t> expected_scale2_shape = {32, 2, 128, 2};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_scale1_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_scale2_shape);
}

TEST_F(DynamicBlockMxQuantTest, DynamicBlockMxQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicBlockMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicBlockMxQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                                  .InputDataTypes({&input_x_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_scale_ref);
    }
}

TEST_F(DynamicBlockMxQuantTest, DynamicBlockMxQuant_InferDtype_error_case)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicBlockMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicBlockMxQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(32)}})
                                  .InputDataTypes({&input_x_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}