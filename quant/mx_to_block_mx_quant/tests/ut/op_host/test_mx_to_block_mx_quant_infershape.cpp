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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/mx_to_block_mx_quant_proto.h"

namespace {
class MxToBlockMxQuantInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MxToBlockMxQuant InferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MxToBlockMxQuant InferShape TearDown" << std::endl; }
};

TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_infershape_case_0)
{
    ge::op::MxToBlockMxQuant op;
    op.UpdateInputDesc("x", create_desc({64, 64}, ge::DT_FLOAT4_E2M1));
    op.UpdateInputDesc("mxscale", create_desc({64, 1, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("dst_type", 35);
    Runtime2TestParam param{{"dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {64, 64};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);

    auto outputScale1 = op.GetOutputDesc(1);
    std::vector<int64_t> expectedScale1Shape = {64, 1, 2};
    EXPECT_EQ(outputScale1.GetShape().GetDims(), expectedScale1Shape);

    auto outputScale2 = op.GetOutputDesc(2);
    std::vector<int64_t> expectedScale2Shape = {1, 64, 2};
    EXPECT_EQ(outputScale2.GetShape().GetDims(), expectedScale2Shape);
}

TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_infershape_case_1)
{
    ge::op::MxToBlockMxQuant op;
    op.UpdateInputDesc("x", create_desc({64, 512}, ge::DT_FLOAT4_E1M2));
    op.UpdateInputDesc("mxscale", create_desc({64, 8, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("dst_type", 36);
    Runtime2TestParam param{{"dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {64, 512};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);

    auto outputScale1 = op.GetOutputDesc(1);
    std::vector<int64_t> expectedScale1Shape = {64, 8, 2};
    EXPECT_EQ(outputScale1.GetShape().GetDims(), expectedScale1Shape);

    auto outputScale2 = op.GetOutputDesc(2);
    std::vector<int64_t> expectedScale2Shape = {1, 512, 2};
    EXPECT_EQ(outputScale2.GetShape().GetDims(), expectedScale2Shape);
}

TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_infershape_case_2)
{
    ge::op::MxToBlockMxQuant op;
    op.UpdateInputDesc("x", create_desc({4, 64, 64}, ge::DT_FLOAT4_E2M1));
    op.UpdateInputDesc("mxscale", create_desc({4, 64, 1, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("dst_type", 35);
    Runtime2TestParam param{{"dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {4, 64, 64};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);

    auto outputScale1 = op.GetOutputDesc(1);
    std::vector<int64_t> expectedScale1Shape = {4, 64, 1, 2};
    EXPECT_EQ(outputScale1.GetShape().GetDims(), expectedScale1Shape);

    auto outputScale2 = op.GetOutputDesc(2);
    std::vector<int64_t> expectedScale2Shape = {4, 1, 64, 2};
    EXPECT_EQ(outputScale2.GetShape().GetDims(), expectedScale2Shape);
}

// Unknown rank case
TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_infershape_case_unknown_rank)
{
    ge::op::MxToBlockMxQuant op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT4_E2M1));
    op.UpdateInputDesc("mxscale", create_desc({-2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("dst_type", 35);
    Runtime2TestParam param{{"dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale1 = op.GetOutputDesc(1);
    auto outputScale2 = op.GetOutputDesc(2);
    std::vector<int64_t> expectedYShape = {-2};
    std::vector<int64_t> expectedScale1Shape = {-2};
    std::vector<int64_t> expectedScale2Shape = {-2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale1.GetShape().GetDims(), expectedScale1Shape);
    EXPECT_EQ(outputScale2.GetShape().GetDims(), expectedScale2Shape);
}

// Error: rank < 2 (1D input)
TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_infershape_error_rank_too_small)
{
    ge::op::MxToBlockMxQuant op;
    op.UpdateInputDesc("x", create_desc({64}, ge::DT_FLOAT4_E2M1));
    op.UpdateInputDesc("mxscale", create_desc({1, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("dst_type", 35);
    Runtime2TestParam param{{"dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// Error: rank > 3 (4D input)
TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_infershape_error_rank_too_large)
{
    ge::op::MxToBlockMxQuant op;
    op.UpdateInputDesc("x", create_desc({2, 4, 64, 64}, ge::DT_FLOAT4_E2M1));
    op.UpdateInputDesc("mxscale", create_desc({2, 4, 64, 1, 2}, ge::DT_FLOAT8_E8M0));
    op.SetAttr("dst_type", 35);
    Runtime2TestParam param{{"dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// InferDataType: dst_type=35 -> FLOAT8_E5M2
TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_InferDtype_case_e5m2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MxToBlockMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MxToBlockMxQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT4_E2M1;
        ge::DataType input_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale1_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_scale2_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                                  .InputDataTypes({&input_x_ref, &input_scale_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale1_ref, &output_scale2_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT8_E5M2);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT8_E8M0);
        EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT8_E8M0);
    }
}

// InferDataType: dst_type=36 -> FLOAT8_E4M3FN
TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_InferDtype_case_e4m3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MxToBlockMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MxToBlockMxQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT4_E1M2;
        ge::DataType input_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E4M3FN;
        ge::DataType output_scale1_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_scale2_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)}})
                                  .InputDataTypes({&input_x_ref, &input_scale_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale1_ref, &output_scale2_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT8_E4M3FN);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT8_E8M0);
        EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT8_E8M0);
    }
}

// InferDataType error: invalid dst_type=40
TEST_F(MxToBlockMxQuantInferShape, MxToBlockMxQuant_InferDtype_error_invalid_dst_type)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MxToBlockMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MxToBlockMxQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT4_E2M1;
        ge::DataType input_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale1_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType output_scale2_ref = ge::DT_FLOAT8_E8M0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)}})
                                  .InputDataTypes({&input_x_ref, &input_scale_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale1_ref, &output_scale2_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

} // namespace
