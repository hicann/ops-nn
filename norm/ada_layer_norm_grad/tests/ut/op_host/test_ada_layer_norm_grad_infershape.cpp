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
#include "log/log.h"
#include "../../../op_graph/ada_layer_norm_grad_proto.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class AdaLayerNormGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaLayerNormGradTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AdaLayerNormGradTest TearDown" << std::endl;
    }
};

// scale/shift [B, 1, H] shape - 4D input
TEST_F(AdaLayerNormGradTest, ada_layer_norm_grad_infershape_test_0)
{
    using namespace ge;
    op::AdaLayerNormGrad op;

    op.UpdateInputDesc("dy", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 4, 6, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({2, 4, 6, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({2, 4, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_scale_desc = op.GetOutputDesc(1);
    auto output_pd_shift_desc = op.GetOutputDesc(2);
    auto output_pd_gamma_desc = op.GetOutputDesc(3);
    auto output_pd_beta_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_pd_x_shape = {2, 4, 6, 4};
    std::vector<int64_t> expected_pd_scale_shape = {2, 4, 4};
    std::vector<int64_t> expected_pd_shift_shape = {2, 4, 4};  // pd_shift shape = scale shape
    std::vector<int64_t> expected_pd_gamma_shape = {4};
    std::vector<int64_t> expected_pd_beta_shape = {4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_scale_desc.GetShape().GetDims(), expected_pd_scale_shape);
    EXPECT_EQ(output_pd_shift_desc.GetShape().GetDims(), expected_pd_shift_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

// scale/shift [B, H] shape - 4D input with mean/rstd [B, S, 1, 1]
TEST_F(AdaLayerNormGradTest, ada_layer_norm_grad_infershape_test_1)
{
    using namespace ge;
    op::AdaLayerNormGrad op;

    op.UpdateInputDesc("dy", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 4, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({2, 4, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({2, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_scale_desc = op.GetOutputDesc(1);
    auto output_pd_shift_desc = op.GetOutputDesc(2);
    auto output_pd_gamma_desc = op.GetOutputDesc(3);
    auto output_pd_beta_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_pd_x_shape = {2, 4, 6, 4};
    std::vector<int64_t> expected_pd_scale_shape = {2, 4};
    std::vector<int64_t> expected_pd_shift_shape = {2, 4};  // pd_shift shape = scale shape
    std::vector<int64_t> expected_pd_gamma_shape = {4};
    std::vector<int64_t> expected_pd_beta_shape = {4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_scale_desc.GetShape().GetDims(), expected_pd_scale_shape);
    EXPECT_EQ(output_pd_shift_desc.GetShape().GetDims(), expected_pd_shift_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

// scale/shift [B, 1, H] shape with mean/rstd [B, S, 1]
TEST_F(AdaLayerNormGradTest, ada_layer_norm_grad_infershape_test_2)
{
    using namespace ge;
    op::AdaLayerNormGrad op;

    op.UpdateInputDesc("dy", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 4, 6, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({2, 4, 6, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({2, 1, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_scale_desc = op.GetOutputDesc(1);
    auto output_pd_shift_desc = op.GetOutputDesc(2);
    auto output_pd_gamma_desc = op.GetOutputDesc(3);
    auto output_pd_beta_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_pd_x_shape = {2, 4, 6, 4};
    std::vector<int64_t> expected_pd_scale_shape = {2, 1, 4};
    std::vector<int64_t> expected_pd_shift_shape = {2, 1, 4};  // pd_shift shape = scale shape
    std::vector<int64_t> expected_pd_gamma_shape = {4};
    std::vector<int64_t> expected_pd_beta_shape = {4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_scale_desc.GetShape().GetDims(), expected_pd_scale_shape);
    EXPECT_EQ(output_pd_shift_desc.GetShape().GetDims(), expected_pd_shift_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

// 3D input [B, S, H] - scale/shift [B, H]
TEST_F(AdaLayerNormGradTest, ada_layer_norm_grad_infershape_test_3)
{
    using namespace ge;
    op::AdaLayerNormGrad op;

    op.UpdateInputDesc("dy", create_desc({2, 3, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 3, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 3, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({2, 3, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({2, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_scale_desc = op.GetOutputDesc(1);
    auto output_pd_shift_desc = op.GetOutputDesc(2);
    auto output_pd_gamma_desc = op.GetOutputDesc(3);
    auto output_pd_beta_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_pd_x_shape = {2, 3, 4};
    std::vector<int64_t> expected_pd_scale_shape = {2, 4};
    std::vector<int64_t> expected_pd_shift_shape = {2, 4};  // pd_shift shape = scale shape
    std::vector<int64_t> expected_pd_gamma_shape = {4};
    std::vector<int64_t> expected_pd_beta_shape = {4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_scale_desc.GetShape().GetDims(), expected_pd_scale_shape);
    EXPECT_EQ(output_pd_shift_desc.GetShape().GetDims(), expected_pd_shift_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

// 2D input [B, H] - scale/shift [H]
TEST_F(AdaLayerNormGradTest, ada_layer_norm_grad_infershape_test_4)
{
    using namespace ge;
    op::AdaLayerNormGrad op;

    op.UpdateInputDesc("dy", create_desc({8, 9}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({8, 9}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({8, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({8, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({9}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({9}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({9}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_scale_desc = op.GetOutputDesc(1);
    auto output_pd_shift_desc = op.GetOutputDesc(2);
    auto output_pd_gamma_desc = op.GetOutputDesc(3);
    auto output_pd_beta_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_pd_x_shape = {8, 9};
    std::vector<int64_t> expected_pd_scale_shape = {9};
    std::vector<int64_t> expected_pd_shift_shape = {9};  // pd_shift shape = scale shape
    std::vector<int64_t> expected_pd_gamma_shape = {9};
    std::vector<int64_t> expected_pd_beta_shape = {9};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_scale_desc.GetShape().GetDims(), expected_pd_scale_shape);
    EXPECT_EQ(output_pd_shift_desc.GetShape().GetDims(), expected_pd_shift_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

// unknown rank (-2)
TEST_F(AdaLayerNormGradTest, ada_layer_norm_grad_infershape_test_5)
{
    using namespace ge;
    op::AdaLayerNormGrad op;

    op.UpdateInputDesc("dy", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({-2}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_scale_desc = op.GetOutputDesc(1);
    auto output_pd_shift_desc = op.GetOutputDesc(2);
    auto output_pd_gamma_desc = op.GetOutputDesc(3);
    auto output_pd_beta_desc = op.GetOutputDesc(4);
    std::vector<int64_t> expected_pd_x_shape = {-2};
    std::vector<int64_t> expected_pd_scale_shape = {-2};
    std::vector<int64_t> expected_pd_shift_shape = {-2};  // pd_shift shape = scale shape
    std::vector<int64_t> expected_pd_gamma_shape = {-2};
    std::vector<int64_t> expected_pd_beta_shape = {-2};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_scale_desc.GetShape().GetDims(), expected_pd_scale_shape);
    EXPECT_EQ(output_pd_shift_desc.GetShape().GetDims(), expected_pd_shift_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

// InferDataType test - float16
TEST_F(AdaLayerNormGradTest, AdaLayerNormGrad_InferDtype_case)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AdaLayerNormGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaLayerNormGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_dy_ref = ge::DT_FLOAT16;
        ge::DataType input_rstd_mean_ref = ge::DT_FLOAT;
        ge::DataType input_scale_gamma_beta_ref = ge::DT_FLOAT16;

        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 5)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_dy_ref, &input_dy_ref, &input_rstd_mean_ref, &input_rstd_mean_ref,
                                 &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref})
                .OutputDataTypes({&input_dy_ref, &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref,
                                  &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_dy_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_dy_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_rstd_mean_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_rstd_mean_ref);
        EXPECT_EQ(context->GetInputDataType(4), input_scale_gamma_beta_ref);
        EXPECT_EQ(context->GetInputDataType(5), input_scale_gamma_beta_ref);
        EXPECT_EQ(context->GetInputDataType(6), input_scale_gamma_beta_ref);

        EXPECT_EQ(context->GetOutputDataType(0), input_dy_ref);
        EXPECT_EQ(context->GetOutputDataType(1), input_scale_gamma_beta_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_scale_gamma_beta_ref);  // pd_shift dtype = scale dtype
        EXPECT_EQ(context->GetOutputDataType(3), input_scale_gamma_beta_ref);
        EXPECT_EQ(context->GetOutputDataType(4), input_scale_gamma_beta_ref);
    }
}

// InferDataType test - float32
TEST_F(AdaLayerNormGradTest, AdaLayerNormGrad_InferDtype_case_float32)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AdaLayerNormGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaLayerNormGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_dy_ref = ge::DT_FLOAT;
        ge::DataType input_rstd_mean_ref = ge::DT_FLOAT;
        ge::DataType input_scale_gamma_beta_ref = ge::DT_FLOAT;

        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 5)
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_dy_ref, &input_dy_ref, &input_rstd_mean_ref, &input_rstd_mean_ref,
                                 &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref})
                .OutputDataTypes({&input_dy_ref, &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref,
                                  &input_scale_gamma_beta_ref, &input_scale_gamma_beta_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), input_dy_ref);
        EXPECT_EQ(context->GetOutputDataType(1), input_scale_gamma_beta_ref);
        EXPECT_EQ(context->GetOutputDataType(2), input_scale_gamma_beta_ref);  // pd_shift dtype = scale dtype
        EXPECT_EQ(context->GetOutputDataType(3), input_scale_gamma_beta_ref);
        EXPECT_EQ(context->GetOutputDataType(4), input_scale_gamma_beta_ref);
    }
}