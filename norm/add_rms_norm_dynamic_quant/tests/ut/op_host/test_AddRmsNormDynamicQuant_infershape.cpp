/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_AddRmsNormDynamicQuant_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "ut_op_util.h"
#include "../../../op_graph/add_rms_norm_dynamic_quant_proto.h"

class AddRmsNormDynamicQuant : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddRmsNormDynamicQuant Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddRmsNormDynamicQuant Proto Test TearDown" << std::endl; }
};

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_int8)
{
    ge::op::AddRmsNormDynamicQuant op;
    op.UpdateInputDesc("x1", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale1", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale2", create_desc({64}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    std::vector<bool> out_shape = {true, true};
    op.SetAttr("output_mask", out_shape);
    op.SetAttr("dst_type", 2);
    Runtime2TestParam param{{"epsilon", "output_mask", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    auto output_scale1_desc = op.GetOutputDesc(3);
    auto output_scale2_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_scale_shape = {8};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_scale_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_scale_shape);
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_int4)
{
    ge::op::AddRmsNormDynamicQuant op;
    op.UpdateInputDesc("x1", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale1", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale2", create_desc({64}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    std::vector<bool> out_shape = {true, true};
    op.SetAttr("output_mask", out_shape);
    op.SetAttr("dst_type", 29);
    Runtime2TestParam param{{"epsilon", "output_mask", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    auto output_scale1_desc = op.GetOutputDesc(3);
    auto output_scale2_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_scale_shape = {8};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_scale_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_scale_shape);
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_hifloat8)
{
    ge::op::AddRmsNormDynamicQuant op;
    op.UpdateInputDesc("x1", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale1", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale2", create_desc({64}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    std::vector<bool> out_shape = {true, false};
    op.SetAttr("output_mask", out_shape);
    op.SetAttr("dst_type", 34);
    Runtime2TestParam param{{"epsilon", "output_mask", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    auto output_scale1_desc = op.GetOutputDesc(3);
    auto output_scale2_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_y2_shape = {1};
    std::vector<int64_t> expected_scale_shape = {8};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_scale_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_y2_shape);
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_float8_e5m2)
{
    ge::op::AddRmsNormDynamicQuant op;
    op.UpdateInputDesc("x1", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale1", create_desc({64}, ge::DT_FLOAT16));
    // op.UpdateInputDesc("smooth_scale2", create_desc({64}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    std::vector<bool> out_shape = {};
    op.SetAttr("output_mask", out_shape);
    op.SetAttr("dst_type", 35);
    Runtime2TestParam param{{"epsilon", "output_mask", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    auto output_scale1_desc = op.GetOutputDesc(3);
    auto output_scale2_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_y2_shape = {1};
    std::vector<int64_t> expected_scale_shape = {8};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y2_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_scale_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_y2_shape);
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_float8_e4m3fn)
{
    ge::op::AddRmsNormDynamicQuant op;
    op.UpdateInputDesc("x1", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale1", create_desc({64}, ge::DT_FLOAT16));
    // op.UpdateInputDesc("smooth_scale1", create_desc({64}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    std::vector<bool> out_shape = {false, true};
    op.SetAttr("output_mask", out_shape);
    op.SetAttr("dst_type", 36);
    Runtime2TestParam param{{"epsilon", "output_mask", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    auto output_scale1_desc = op.GetOutputDesc(3);
    auto output_scale2_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_y1_shape = {1};
    std::vector<int64_t> expected_scale_shape = {8};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expected_y1_shape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expected_scale_shape);
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_unknown_rank_01)
{
    ge::op::AddRmsNormDynamicQuant op;
    op.UpdateInputDesc("x1", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale1", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale2", create_desc({-2}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    std::vector<bool> out_shape = {true, true};
    op.SetAttr("output_mask", out_shape);
    op.SetAttr("dst_type", 36);
    Runtime2TestParam param{{"epsilon", "output_mask", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    auto output_scale1_desc = op.GetOutputDesc(3);
    auto output_scale2_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expectedShape = {-2};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expectedShape);
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_infershape_case_unknown_rank_02)
{
    ge::op::AddRmsNormDynamicQuant op;
    op.UpdateInputDesc("x1", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x2", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale1", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scale2", create_desc({-2}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    std::vector<bool> out_shape = {};
    op.SetAttr("output_mask", out_shape);
    op.SetAttr("dst_type", 36);
    Runtime2TestParam param{{"epsilon", "output_mask", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto output_y1_desc = op.GetOutputDesc(0);
    auto output_y2_desc = op.GetOutputDesc(1);
    auto output_x_desc = op.GetOutputDesc(2);
    auto output_scale1_desc = op.GetOutputDesc(3);
    auto output_scale2_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expectedShape = {-2};
    EXPECT_EQ(output_y1_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_y2_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_scale1_desc.GetShape().GetDims(), expectedShape);
    EXPECT_EQ(output_scale2_desc.GetShape().GetDims(), expectedShape);
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType smooth_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_INT8;
        std::vector<bool> output_mask = {};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(5)
                                  .NodeIoNum(5, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                       {"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &smooth_ref, &smooth_ref})
                                  .OutputDataTypes({&output_ref, &output_ref, &input_ref, &scale_ref, &scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(3), scale_ref);
        EXPECT_EQ(context->GetOutputDataType(4), scale_ref);
    }
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType smooth_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_HIFLOAT8;
        std::vector<bool> output_mask = {};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(5)
                                  .NodeIoNum(5, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                       {"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &smooth_ref, &smooth_ref})
                                  .OutputDataTypes({&output_ref, &output_ref, &input_ref, &scale_ref, &scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(3), scale_ref);
        EXPECT_EQ(context->GetOutputDataType(4), scale_ref);
    }
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_InferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType smooth_ref = ge::DT_FLOAT16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_INT4;
        std::vector<bool> output_mask = {};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(5)
                                  .NodeIoNum(5, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                       {"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(29)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &smooth_ref, &smooth_ref})
                                  .OutputDataTypes({&output_ref, &output_ref, &input_ref, &scale_ref, &scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(3), scale_ref);
        EXPECT_EQ(context->GetOutputDataType(4), scale_ref);
    }
}

TEST_F(AddRmsNormDynamicQuant, AddRmsNormDynamicQuant_InferDtype_case_3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType smooth_ref = ge::DT_BF16;
        ge::DataType scale_ref = ge::DT_FLOAT;
        ge::DataType output_ref = ge::DT_INT4;
        std::vector<bool> output_mask = {};
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(5)
                                  .NodeIoNum(5, 5)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs(
                                      {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                       {"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)},
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(29)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &smooth_ref, &smooth_ref})
                                  .OutputDataTypes({&output_ref, &output_ref, &input_ref, &scale_ref, &scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(3), scale_ref);
        EXPECT_EQ(context->GetOutputDataType(4), scale_ref);
    }
}
