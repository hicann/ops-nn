/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/layer_norm_grad_v3_proto.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class LayerNormGradV3Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "LayerNormGradV3Test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "LayerNormGradV4Test TearDown" << std::endl;
    }
};

TEST_F(LayerNormGradV3Test, layer_norm_grad_v3_infershape_test_0)
{
    using namespace ge;
    // input_y_dims_not_equal_input_var

    op::LayerNormGradV3 op;

    op.UpdateInputDesc("dy", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({2, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_gamma_desc = op.GetOutputDesc(1);
    auto output_pd_beta_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_pd_x_shape = {2, 4, 6, 4};
    std::vector<int64_t> expected_pd_gamma_shape = {4};
    std::vector<int64_t> expected_pd_beta_shape = {4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

TEST_F(LayerNormGradV3Test, layer_norm_grad_v3_infershape_test_1)
{
    // params_greater_norm
    ge::op::LayerNormGradV3 op;

    op.UpdateInputDesc("dy", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 4, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({2, 4, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_gamma_desc = op.GetOutputDesc(1);
    auto output_pd_beta_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_pd_x_shape = {2, 4, 6, 4};
    std::vector<int64_t> expected_pd_gamma_shape = {4};
    std::vector<int64_t> expected_pd_beta_shape = {4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

TEST_F(LayerNormGradV3Test, layer_norm_grad_v3_infershape_test_2)
{
    // params_equal_norm
    ge::op::LayerNormGradV3 op;

    op.UpdateInputDesc("dy", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 4, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({2, 4, 1, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({6, 4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_gamma_desc = op.GetOutputDesc(1);
    auto output_pd_beta_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_pd_x_shape = {2, 4, 6, 4};
    std::vector<int64_t> expected_pd_gamma_shape = {6, 4};
    std::vector<int64_t> expected_pd_beta_shape = {6, 4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

TEST_F(LayerNormGradV3Test, layer_norm_grad_v3_infershape_test_3)
{
    // norm_greater_params
    ge::op::LayerNormGradV3 op;

    op.UpdateInputDesc("dy", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({2, 4, 6, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({2, 4, 6, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({2, 4, 6, 1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({6, 4}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_gamma_desc = op.GetOutputDesc(1);
    auto output_pd_beta_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_pd_x_shape = {2, 4, 6, 4};
    std::vector<int64_t> expected_pd_gamma_shape = {6, 4};
    std::vector<int64_t> expected_pd_beta_shape = {6, 4};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

TEST_F(LayerNormGradV3Test, layer_norm_grad_v3_infershape_test_4)
{
    // input_is_scala
    ge::op::LayerNormGradV3 op;

    op.UpdateInputDesc("dy", create_desc({}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_gamma_desc = op.GetOutputDesc(1);
    auto output_pd_beta_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_pd_x_shape = {};
    std::vector<int64_t> expected_pd_gamma_shape = {};
    std::vector<int64_t> expected_pd_beta_shape = {};
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

TEST_F(LayerNormGradV3Test, layer_norm_grad_v3_infershape_test_5)
{
    // input_is_scala
    ge::op::LayerNormGradV3 op;

    op.UpdateInputDesc(
        "dy", create_desc(
                  {
                      -2,
                  },
                  ge::DT_FLOAT16));
    op.UpdateInputDesc(
        "x", create_desc(
                 {
                     -2,
                 },
                 ge::DT_FLOAT16));
    op.UpdateInputDesc(
        "rstd", create_desc(
                    {
                        -2,
                    },
                    ge::DT_FLOAT16));
    op.UpdateInputDesc(
        "mean", create_desc(
                    {
                        -2,
                    },
                    ge::DT_FLOAT16));
    op.UpdateInputDesc(
        "gamma", create_desc(
                     {
                         -2,
                     },
                     ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_pd_x_desc = op.GetOutputDesc(0);
    auto output_pd_gamma_desc = op.GetOutputDesc(1);
    auto output_pd_beta_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_pd_x_shape = {
        -2,
    };
    std::vector<int64_t> expected_pd_gamma_shape = {
        -2,
    };
    std::vector<int64_t> expected_pd_beta_shape = {
        -2,
    };
    EXPECT_EQ(output_pd_x_desc.GetShape().GetDims(), expected_pd_x_shape);
    EXPECT_EQ(output_pd_gamma_desc.GetShape().GetDims(), expected_pd_gamma_shape);
    EXPECT_EQ(output_pd_beta_desc.GetShape().GetDims(), expected_pd_beta_shape);
}

TEST_F(LayerNormGradV3Test, LayerNormGradV3_InferDtype_case)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormGradV3"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNormGradV3")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_dy_ref = ge::DT_FLOAT16;
        ge::DataType input_var_ref = ge::DT_FLOAT;
        ge::DataType input_gamma_ref = ge::DT_FLOAT;
        ge::DataType output_x_ref = ge::DT_FLOAT16;
        ge::DataType output_gb_ref = ge::DT_FLOAT;

        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(5)
                .NodeIoNum(5, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)

                .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_dy_ref, &input_dy_ref, &input_var_ref, &input_var_ref, &input_gamma_ref})
                .OutputDataTypes({&output_x_ref, &output_gb_ref, &output_gb_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_dy_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_dy_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_var_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_var_ref);
        EXPECT_EQ(context->GetInputDataType(4), input_gamma_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_x_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_gb_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_gb_ref);
    }
}
