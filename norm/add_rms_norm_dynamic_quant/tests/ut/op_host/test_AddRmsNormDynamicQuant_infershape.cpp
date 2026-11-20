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
#include <string>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../../norm_common/op_host/norm_tiling_check_common.h"
#include "../../../../norm_common/op_host/norm_error_report.h"
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

TEST_F(AddRmsNormDynamicQuant, InvalidDstTypeReportsFormalErrorOnce)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormDynamicQuant")->infer_datatype;

    ASSERT_NE(data_type_func, nullptr);
    {
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
                                       {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &smooth_ref, &smooth_ref})
                                  .OutputDataTypes({&output_ref, &output_ref, &input_ref, &scale_ref, &scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        auto& manager = ErrorManager::GetInstance();
        ASSERT_EQ(manager.Init(), 0);
        manager.GetRawErrorMessages();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
        const auto errors = manager.GetRawErrorMessages();
        ASSERT_EQ(errors.size(), 1U);
        EXPECT_EQ(errors[0].error_id, "EZ0026");
        const auto& fields = errors[0].args_map;
        ASSERT_EQ(fields.count("param_name"), 1U);
        ASSERT_EQ(fields.count("incorrect_value"), 1U);
        EXPECT_EQ(fields.at("param_name"), "dst_type");
        EXPECT_EQ(fields.at("incorrect_value"), "-1");
    }
}

TEST_F(AddRmsNormDynamicQuant, SharedShapeMismatchReportsOneFormalError)
{
    gert::StorageShape x1({2, 3}, {2, 3});
    gert::StorageShape x2({2, 4}, {2, 4});
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_FALSE(NormCheck::CheckShapeSame(&x1, &x2, "AddRmsNormDynamicQuant", "x1", "x2"));
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0010");
    EXPECT_EQ(errors[0].args_map.at("param_names"), "x1, x2");
    EXPECT_EQ(errors[0].args_map.at("incorrect_shapes"), "[2, 3], [2, 4]");
}

TEST_F(AddRmsNormDynamicQuant, SharedRankMismatchReportsOneFormalError)
{
    gert::StorageShape x1({2, 3}, {2, 3});
    gert::StorageShape gamma({2, 3, 4}, {2, 3, 4});
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_FALSE(NormCheck::CheckShapeBC(&x1, &gamma, "AddRmsNormDynamicQuant", "x1", "gamma"));
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0013");
    EXPECT_EQ(errors[0].args_map.at("param_names"), "x1, gamma");
    EXPECT_EQ(errors[0].args_map.at("incorrect_dims"), "2, 3");
}

TEST_F(AddRmsNormDynamicQuant, TilingFailurePreservesRegisteredFieldsAndSingleReport)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    Ops::NN::ErrorReport::TilingFailure("AddRmsNormDynamicQuant", "AIV core count is 0");
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("opName"), "AddRmsNormDynamicQuant");
    EXPECT_EQ(errors[0].args_map.at("reason"), "AIV core count is 0");
}

TEST_F(AddRmsNormDynamicQuant, UnsupportedCapabilityPreservesRegisteredFieldsAndSingleReport)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    Ops::NN::ErrorReport::UnsupportedCapability("AddRmsNormDynamicQuant", "Available UB 0 is below required UB 1024");
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1009");
    EXPECT_EQ(errors[0].args_map.at("opName"), "AddRmsNormDynamicQuant");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Available UB 0 is below required UB 1024");
}

TEST_F(AddRmsNormDynamicQuant, TilingFailureHandlesNullOperatorName)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    Ops::NN::ErrorReport::TilingFailure(nullptr, "Missing metadata");
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("opName"), "Norm");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing metadata");
}

TEST_F(AddRmsNormDynamicQuant, UnsupportedCapabilityHandlesNullOperatorName)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    Ops::NN::ErrorReport::UnsupportedCapability(nullptr, "Missing metadata");
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1009");
    EXPECT_EQ(errors[0].args_map.at("opName"), "Norm");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing metadata");
}

TEST_F(AddRmsNormDynamicQuant, TilingFailureFormatsRegisteredExecutionMessage)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    Ops::NN::ErrorReport::TilingFailure("AddRmsNormDynamicQuant", "SDK execution contract probe");
    const auto message = manager.GetErrorMessage();
    EXPECT_NE(message.find("EZ1008"), std::string::npos) << message;
    EXPECT_NE(message.find("Failed to execute operator AddRmsNormDynamicQuant. Reason: SDK execution contract probe"),
              std::string::npos)
        << message;
    EXPECT_EQ(message.find("%s"), std::string::npos) << message;
    manager.GetRawErrorMessages();
}

TEST_F(AddRmsNormDynamicQuant, UnsupportedCapabilityFormatsRegisteredExecutionMessage)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    Ops::NN::ErrorReport::UnsupportedCapability("AddRmsNormDynamicQuant", "SDK capability contract probe");
    const auto message = manager.GetErrorMessage();
    EXPECT_NE(message.find("EZ1009"), std::string::npos) << message;
    EXPECT_NE(message.find("Failed to execute operator AddRmsNormDynamicQuant. Reason: SDK capability contract probe"),
              std::string::npos)
        << message;
    EXPECT_EQ(message.find("%s"), std::string::npos) << message;
    manager.GetRawErrorMessages();
}
