/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_bn_infer_grad_infershape.cpp
 * \brief BNInferGrad InferShape / InferDataType UT
 */
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/bn_infer_grad_proto.h"

class BNInferGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BNInferGradInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BNInferGradInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(BNInferGradInferShapeTest, BNInferGrad_infershape_nchw)
{
    ge::op::BNInferGrad op;
    op.UpdateInputDesc("grads", create_desc({2, 16, 8, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("batch_variance", create_desc({16}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {2, 16, 8, 8};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
}

TEST_F(BNInferGradInferShapeTest, BNInferGrad_infershape_fp16)
{
    ge::op::BNInferGrad op;
    op.UpdateInputDesc("grads", create_desc({4, 32, 16, 16}, ge::DT_FLOAT16));
    op.UpdateInputDesc("scale", create_desc({32}, ge::DT_FLOAT));
    op.UpdateInputDesc("batch_variance", create_desc({32}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {4, 32, 16, 16};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
}

TEST_F(BNInferGradInferShapeTest, BNInferGrad_infershape_nc1hwc0)
{
    ge::op::BNInferGrad op;
    op.UpdateInputDesc("grads", create_desc({2, 1, 8, 8, 16}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({16}, ge::DT_FLOAT));
    op.UpdateInputDesc("batch_variance", create_desc({16}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {2, 1, 8, 8, 16};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
}

TEST_F(BNInferGradInferShapeTest, BNInferGrad_InferDtype_fp32)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BNInferGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BNInferGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_grads = ge::DT_FLOAT;
        ge::DataType input_scale = ge::DT_FLOAT;
        ge::DataType input_variance = ge::DT_FLOAT;
        ge::DataType output_x_backprop = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(3)
                .NodeIoNum(3, 1)
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .InputDataTypes({&input_grads, &input_scale, &input_variance})
                .OutputDataTypes({&output_x_backprop})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_grads);
        EXPECT_EQ(context->GetOutputDataType(0), output_x_backprop);
    }
}

TEST_F(BNInferGradInferShapeTest, BNInferGrad_InferDtype_fp16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BNInferGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BNInferGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_grads = ge::DT_FLOAT16;
        ge::DataType input_scale = ge::DT_FLOAT;
        ge::DataType input_variance = ge::DT_FLOAT;
        ge::DataType output_x_backprop = ge::DT_FLOAT16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(3)
                .NodeIoNum(3, 1)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .InputDataTypes({&input_grads, &input_scale, &input_variance})
                .OutputDataTypes({&output_x_backprop})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_grads);
        EXPECT_EQ(context->GetOutputDataType(0), output_x_backprop);
    }
}

TEST_F(BNInferGradInferShapeTest, BNInferGrad_InferDtype_bf16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BNInferGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BNInferGrad")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_grads = ge::DT_BF16;
        ge::DataType input_scale = ge::DT_FLOAT;
        ge::DataType input_variance = ge::DT_FLOAT;
        ge::DataType output_x_backprop = ge::DT_BF16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(3)
                .NodeIoNum(3, 1)
                .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                .InputDataTypes({&input_grads, &input_scale, &input_variance})
                .OutputDataTypes({&output_x_backprop})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_grads);
        EXPECT_EQ(context->GetOutputDataType(0), output_x_backprop);
    }
}
