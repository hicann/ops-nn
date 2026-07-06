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
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "../../../op_graph/rms_norm_grad_quant_proto.h"

class RmsNormGradQuant : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormGradQuant Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RmsNormGradQuant Proto Test TearDown" << std::endl; }
};

TEST_F(RmsNormGradQuant, RmsNormGradQuant_infershape_case_0)
{
    ge::op::RmsNormGradQuant op;

    op.UpdateInputDesc("dy", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({4}, ge::DT_FLOAT));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_FLOAT16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_dx_desc = op.GetOutputDesc(0);
    auto output_dgamma_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_dx_shape = {4, 1, 8};
    std::vector<int64_t> expected_dgamma_shape = {8};
    EXPECT_EQ(output_dx_desc.GetShape().GetDims(), expected_dx_shape);
    EXPECT_EQ(output_dgamma_desc.GetShape().GetDims(), expected_dgamma_shape);
}

TEST_F(RmsNormGradQuant, RmsNormGradQuant_infershape_case_1_with_optional)
{
    ge::op::RmsNormGradQuant op;

    op.UpdateInputDesc("dy", create_desc({16, 8192}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({16, 8192}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("gamma", create_desc({8192}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales_x", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("offset_x", create_desc({1}, ge::DT_INT32));

    op.SetAttr("quant_mode", "static");
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", static_cast<int>(ge::DT_HIFLOAT8));
    Runtime2TestParam param{{"quant_mode", "div_mode", "dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_dx_desc = op.GetOutputDesc(0);
    auto output_dgamma_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_dx_shape = {16, 8192};
    std::vector<int64_t> expected_dgamma_shape = {8192};
    EXPECT_EQ(output_dx_desc.GetShape().GetDims(), expected_dx_shape);
    EXPECT_EQ(output_dgamma_desc.GetShape().GetDims(), expected_dgamma_shape);
}

TEST_F(RmsNormGradQuant, RmsNormGradQuant_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormGradQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormGradQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_INT8;
        ge::DataType dgamma_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref, &dgamma_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), dgamma_ref);
    }
}

// Coverage: infershape.cpp InferDataType with dst_type attr (dstTypePtr != nullptr branch)
TEST_F(RmsNormGradQuant, RmsNormGradQuant_InferDtype_with_dst_type_attr)
{
    ge::op::RmsNormGradQuant op;

    op.UpdateInputDesc("dy", create_desc({16, 8192}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({16, 8192}, ge::DT_FLOAT16));
    op.UpdateInputDesc("rstd", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("gamma", create_desc({8192}, ge::DT_FLOAT));
    op.UpdateInputDesc("scales_x", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("offset_x", create_desc({1}, ge::DT_INT32));

    op.SetAttr("quant_mode", "static");
    op.SetAttr("div_mode", true);
    op.SetAttr("dst_type", static_cast<int>(ge::DT_HIFLOAT8));
    Runtime2TestParam param{{"quant_mode", "div_mode", "dst_type"}, {}, {}};
    EXPECT_EQ(InferDataTypeTest(op, param), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuant, RmsNormGradQuant_infershape_bf16_case)
{
    ge::op::RmsNormGradQuant op;

    op.UpdateInputDesc("dy", create_desc({8, 64}, ge::DT_BF16));
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_BF16));
    op.UpdateInputDesc("rstd", create_desc({8}, ge::DT_FLOAT));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_BF16));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_dx_desc = op.GetOutputDesc(0);
    auto output_dgamma_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_dx_shape = {8, 64};
    std::vector<int64_t> expected_dgamma_shape = {64};
    EXPECT_EQ(output_dx_desc.GetShape().GetDims(), expected_dx_shape);
    EXPECT_EQ(output_dgamma_desc.GetShape().GetDims(), expected_dgamma_shape);
}
