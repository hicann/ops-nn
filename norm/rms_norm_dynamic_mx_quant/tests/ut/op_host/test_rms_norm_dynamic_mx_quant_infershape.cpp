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
#include "../../../op_graph/rms_norm_dynamic_mx_quant_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class RmsNormDynamicMxQuantInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RmsNormDynamicMxQuantInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RmsNormDynamicMxQuantInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_case1)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({64}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_mxscale_shape = {8, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_case2)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({64}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("scale_alg", 0);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "scale_alg", "round_mode", "dst_type", "output_rstd"},{},{}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_mxscale_shape = {8, 1, 2};
    std::vector<int64_t> expected_rstd_shape = {8, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infer_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType mx_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(3)
                .NodeIoNum(3, 3)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                .InputDataTypes({&input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&y_ref, &mx_scale_ref, &rstd_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), mx_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_3d_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({4}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {4, 3, 4};
    std::vector<int64_t> expected_mxscale_shape = {4, 3, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_4d_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({2, 3, 4, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({5}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {2, 3, 4, 5};
    std::vector<int64_t> expected_mxscale_shape = {2, 3, 4, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_single_element_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {1};
    std::vector<int64_t> expected_mxscale_shape = {1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_aligned_boundary_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({256, 256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({256}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {256, 256};
    std::vector<int64_t> expected_mxscale_shape = {256, 4, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_unaligned_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({33, 17}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({17}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({17}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {33, 17};
    std::vector<int64_t> expected_mxscale_shape = {33, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_single_element_bf16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({1024}, ge::DT_BF16));
    op.UpdateInputDesc("gamma", create_desc({1024}, ge::DT_BF16));
    op.UpdateInputDesc("beta", create_desc({1024}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {1024};
    std::vector<int64_t> expected_mxscale_shape = {16, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_m_not0_n_0_true)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({176, 0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({0}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("scale_alg", 0);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "scale_alg", "round_mode", "dst_type", "output_rstd"},{},{}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {176, 0};
    std::vector<int64_t> expected_mxscale_shape = {176, 0, 2};
    std::vector<int64_t> expected_rstd_shape = {176, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_m_0_n_not0_true)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({0, 100}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({100}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({100}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("scale_alg", 0);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "scale_alg", "round_mode", "dst_type", "output_rstd"},{},{}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {0, 100};
    std::vector<int64_t> expected_mxscale_shape = {0, 2, 2};
    std::vector<int64_t> expected_rstd_shape = {0, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_m_0_n_0_true)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({0, 0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({0}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({0}, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("scale_alg", 0);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "scale_alg", "round_mode", "dst_type", "output_rstd"},{},{}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {0, 0};
    std::vector<int64_t> expected_mxscale_shape = {0, 0, 2};
    std::vector<int64_t> expected_rstd_shape = {0, 1};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expected_rstd_shape);
}