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
#include "ut_op_common.h"
#include "../../../op_graph/quantize_add_layer_norm_proto.h"

class QuantizeAddLayerNorm : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantizeAddLayerNorm infershape Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantizeAddLayerNorm infershape Test TearDown" << std::endl;
    }
};

TEST_F(QuantizeAddLayerNorm, QuantizeAddLayerNorm_infershape_case_0)
{
    ge::op::QuantizeAddLayerNorm op;
    op.UpdateInputDesc("x1", create_desc({4, 1, 8}, ge::DT_BF16));
    op.UpdateInputDesc("x2", create_desc({4, 1, 8}, ge::DT_BF16));
    op.UpdateInputDesc("gamma", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("beta", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("bias", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("scales", create_desc({8}, ge::DT_BF16));
    op.UpdateInputDesc("zero_points", create_desc({8}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_x_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {4, 1, 8};
    std::vector<int64_t> expected_x_shape = {4, 1, 8};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_x_desc.GetShape().GetDims(), expected_x_shape);
}

TEST_F(QuantizeAddLayerNorm, QuantizeAddLayerNorm_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantizeAddLayerNorm"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantizeAddLayerNorm")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType y_out_ref = ge::DT_INT8;
        ge::DataType x_out_ref = ge::DT_BF16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(7)
                .NodeIoNum(7, 2)
                .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&y_out_ref, &x_out_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), y_out_ref);
        EXPECT_EQ(context->GetOutputDataType(1), x_out_ref);
    }
}