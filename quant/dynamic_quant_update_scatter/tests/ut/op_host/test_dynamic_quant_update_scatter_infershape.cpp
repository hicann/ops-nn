/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/dynamic_quant_update_scatter_proto.h"

namespace {
class DynamicQuantUpdateScatter : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DynamicQuantUpdateScatter InferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DynamicQuantUpdateScatter InferShape TearDown" << std::endl; }
};

TEST_F(DynamicQuantUpdateScatter, DynamicQuantUpdateScatter_infershape_case_0)
{
    ge::op::DynamicQuantUpdateScatter op;
    op.UpdateInputDesc("var", create_desc({24, 1, 128}, ge::DT_INT8));
    op.UpdateInputDesc("var_scale", create_desc({24, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("indices", create_desc({24}, ge::DT_INT32));
    op.UpdateInputDesc("updates", create_desc({24, 1, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scales", create_desc({128}, ge::DT_FLOAT16));
    op.SetAttr("reduce", "update");
    op.SetAttr("axis", -2);
    Runtime2TestParam param{{"reduce", "axis"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputVar = op.GetOutputDesc(0);
    auto outputVarScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedVarShape = {24, 1, 128};
    std::vector<int64_t> expectedVarScaleShape = {24, 1, 1};
    EXPECT_EQ(outputVar.GetShape().GetDims(), expectedVarShape);
    EXPECT_EQ(outputVarScale.GetShape().GetDims(), expectedVarScaleShape);
}

TEST_F(DynamicQuantUpdateScatter, DynamicQuantUpdateScatter_infershape_case_1)
{
    ge::op::DynamicQuantUpdateScatter op;
    op.UpdateInputDesc("var", create_desc({59, 57, 256, 192}, ge::DT_INT8));
    op.UpdateInputDesc("var_scale", create_desc({59, 57, 256, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("indices", create_desc({20, 2}, ge::DT_INT32));
    op.UpdateInputDesc("updates", create_desc({20, 57, 43, 192}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scales", create_desc({192}, ge::DT_FLOAT16));
    op.SetAttr("reduce", "update");
    op.SetAttr("axis", 0);
    Runtime2TestParam param{{"reduce", "axis"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputVar = op.GetOutputDesc(0);
    auto outputVarScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedVarShape = {59, 57, 256, 192};
    std::vector<int64_t> expectedVarScaleShape = {59, 57, 256, 1};
    EXPECT_EQ(outputVar.GetShape().GetDims(), expectedVarShape);
    EXPECT_EQ(outputVarScale.GetShape().GetDims(), expectedVarScaleShape);
}

TEST_F(DynamicQuantUpdateScatter, DynamicQuantUpdateScatter_infershape_case_unknow_dim)
{
    ge::op::DynamicQuantUpdateScatter op;
    op.UpdateInputDesc("var", create_desc({-1, 1, 128}, ge::DT_INT8));
    op.UpdateInputDesc("var_scale", create_desc({-1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("indices", create_desc({24}, ge::DT_INT32));
    op.UpdateInputDesc("updates", create_desc({24, 1, 128}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scales", create_desc({128}, ge::DT_FLOAT16));
    op.SetAttr("reduce", "update");
    op.SetAttr("axis", -2);
    Runtime2TestParam param{{"reduce", "axis"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputVar = op.GetOutputDesc(0);
    auto outputVarScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedVarShape = {-1, 1, 128};
    std::vector<int64_t> expectedVarScaleShape = {-1, 1, 1};
    EXPECT_EQ(outputVar.GetShape().GetDims(), expectedVarShape);
    EXPECT_EQ(outputVarScale.GetShape().GetDims(), expectedVarScaleShape);
}

TEST_F(DynamicQuantUpdateScatter, DynamicQuantUpdateScatter_infershape_case_unknow_rank)
{
    ge::op::DynamicQuantUpdateScatter op;
    op.UpdateInputDesc("var", create_desc({-2}, ge::DT_INT8));
    op.UpdateInputDesc("var_scale", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("indices", create_desc({-2}, ge::DT_INT32));
    op.UpdateInputDesc("updates", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scales", create_desc({-2}, ge::DT_FLOAT16));
    op.SetAttr("reduce", "update");
    op.SetAttr("axis", -2);
    Runtime2TestParam param{{"reduce", "axis"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputVar = op.GetOutputDesc(0);
    auto outputVarScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedVarShape = {-2};
    std::vector<int64_t> expectedVarScaleShape = {-2};
    EXPECT_EQ(outputVar.GetShape().GetDims(), expectedVarShape);
    EXPECT_EQ(outputVarScale.GetShape().GetDims(), expectedVarScaleShape);
}

TEST_F(DynamicQuantUpdateScatter, DynamicQuantUpdateScatter_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatter"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatter")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_var = ge::DT_INT8;
        ge::DataType input_var_scale = ge::DT_FLOAT;
        ge::DataType input_indices = ge::DT_INT32;
        ge::DataType input_updates = ge::DT_FLOAT16;
        ge::DataType input_smooth_scales = ge::DT_FLOAT16;
        ge::DataType output_var = ge::DT_INT8;
        ge::DataType output_var_scale = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(5)
                                  .NodeIoNum(5, 2)
                                  .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"reduce", Ops::NN::AnyValue::CreateFrom<std::string>("update")},
                                              {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-2)}})
                                  .InputDataTypes({&input_var, &input_var_scale, &input_indices, &input_updates,
                                                   &input_smooth_scales})
                                  .OutputDataTypes({&output_var, &output_var_scale})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_var);
        EXPECT_EQ(context->GetInputDataType(1), input_var_scale);
        EXPECT_EQ(context->GetInputDataType(2), input_indices);
        EXPECT_EQ(context->GetInputDataType(3), input_updates);
        EXPECT_EQ(context->GetInputDataType(4), input_smooth_scales);

        EXPECT_EQ(context->GetOutputDataType(0), output_var);
        EXPECT_EQ(context->GetOutputDataType(1), output_var_scale);
    }
}

TEST_F(DynamicQuantUpdateScatter, DynamicQuantUpdateScatter_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatter"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantUpdateScatter")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_var = ge::DT_INT8;
        ge::DataType input_var_scale = ge::DT_FLOAT;
        ge::DataType input_indices = ge::DT_INT64;
        ge::DataType input_updates = ge::DT_BF16;
        ge::DataType input_smooth_scales = ge::DT_BF16;
        ge::DataType output_var = ge::DT_INT8;
        ge::DataType output_var_scale = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(5)
                                  .NodeIoNum(5, 2)
                                  .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"reduce", Ops::NN::AnyValue::CreateFrom<std::string>("update")},
                                              {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .InputDataTypes({&input_var, &input_var_scale, &input_indices, &input_updates,
                                                   &input_smooth_scales})
                                  .OutputDataTypes({&output_var, &output_var_scale})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_var);
        EXPECT_EQ(context->GetInputDataType(1), input_var_scale);
        EXPECT_EQ(context->GetInputDataType(2), input_indices);
        EXPECT_EQ(context->GetInputDataType(3), input_updates);
        EXPECT_EQ(context->GetInputDataType(4), input_smooth_scales);

        EXPECT_EQ(context->GetOutputDataType(0), output_var);
        EXPECT_EQ(context->GetOutputDataType(1), output_var_scale);
    }
}

} // namespace