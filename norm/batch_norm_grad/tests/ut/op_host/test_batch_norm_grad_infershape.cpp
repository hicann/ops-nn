/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_batch_norm_grad_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/batch_norm_grad_proto.h"

class BatchNormGradTestInferShapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchNormGradTest Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchNormGradTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(BatchNormGradTestInferShapeTest, BatchNormGrad_infershape_case_0)
{
    ge::op::BatchNormGrad op;
    op.UpdateInputDesc("y_backprop", create_desc({10, 20, 10, 256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("x", create_desc({10, 20, 10, 256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scale", create_desc({20}, ge::DT_FLOAT));
    op.UpdateInputDesc("reserve_space_1", create_desc({20}, ge::DT_FLOAT));
    op.UpdateInputDesc("reserve_space_2", create_desc({20}, ge::DT_FLOAT));
    op.UpdateInputDesc("reserve_space_3", create_desc({20}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape_1 = {10, 20, 10, 256};
    std::vector<int64_t> expected_shape_2 = {20};
    std::vector<int64_t> expected_shape_empty = {0};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape_1);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape_2);
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expected_shape_2);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expected_shape_empty);
    EXPECT_EQ(op.GetOutputDesc(4).GetShape().GetDims(), expected_shape_empty);
}

TEST_F(BatchNormGradTestInferShapeTest, BatchNormGrad_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref_fp16 = ge::DT_FLOAT16;
        ge::DataType input_ref_fp32 = ge::DT_FLOAT;
        ge::DataType output_ref_fp16 = ge::DT_FLOAT16;
        ge::DataType output_ref_fp32 = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(6)
                .NodeIoNum(6, 5)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes(
                    {&input_ref_fp16, &input_ref_fp16, &input_ref_fp32, &input_ref_fp32, &input_ref_fp32,
                     &input_ref_fp32})
                .OutputDataTypes(
                    {&output_ref_fp16, &output_ref_fp32, &output_ref_fp32, &output_ref_fp32, &output_ref_fp32})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(3), output_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(4), output_ref_fp32);
    }
}