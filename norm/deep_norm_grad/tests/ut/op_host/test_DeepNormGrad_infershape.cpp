/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "ut_op_common.h"
#include "../../../op_graph/deep_norm_grad_proto.h"

class DeepNormGrad : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DeepNormGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DeepNormGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(DeepNormGrad, DeepNormGrad_infershape_case_0)
{
    ge::op::DeepNormGrad op;
    op.UpdateInputDesc("dy", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gx", create_desc({4, 1, 8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({4, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("rstd", create_desc({4, 1, 1}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_dx_desc = op.GetOutputDesc(0);
    auto output_dgx_desc = op.GetOutputDesc(1);
    auto output_dbeta_desc = op.GetOutputDesc(2);
    auto output_dgamma_desc = op.GetOutputDesc(3);
    std::vector<int64_t> expected_dx_shape = {4, 1, 8};
    std::vector<int64_t> expected_dgx_shape = {4, 1, 8};
    std::vector<int64_t> expected_dbeta_shape = {8};
    std::vector<int64_t> expected_dgamma_shape = {8};
    EXPECT_EQ(output_dx_desc.GetShape().GetDims(), expected_dx_shape);
    EXPECT_EQ(output_dgx_desc.GetShape().GetDims(), expected_dgx_shape);
    EXPECT_EQ(output_dbeta_desc.GetShape().GetDims(), expected_dbeta_shape);
    EXPECT_EQ(output_dgamma_desc.GetShape().GetDims(), expected_dgamma_shape);
}

TEST_F(DeepNormGrad, DeepNormGrad_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DeepNormGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DeepNormGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType input_mean_ref = ge::DT_FLOAT;
        ge::DataType output_dx_ref = ge::DT_FLOAT16;
        ge::DataType output_dgx_ref = ge::DT_FLOAT16;
        ge::DataType output_dbeta_ref = ge::DT_FLOAT;
        ge::DataType output_dgamma_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(6)
                .NodeIoNum(6, 4)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_mean_ref, &input_mean_ref})
                .OutputDataTypes({&output_dx_ref, &output_dgx_ref, &output_dbeta_ref, &output_dgamma_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_dx_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_dgx_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_dbeta_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_dgamma_ref);
    }
}
