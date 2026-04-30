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
 * \file test_sync_batch_norm_backward_reduce_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/sync_batch_norm_backward_reduce_proto.h"

class SyncBatchNormBackwardReduceInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SyncBatchNormBackwardReduce Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SyncBatchNormBackwardReduce Proto Test TearDown" << std::endl;
    }
};

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_infershape_case_0)
{
    ge::op::SyncBatchNormBackwardReduce op;
    op.UpdateInputDesc("sum_dy", create_desc({10, 20, 30, 40}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sum_dy_dx_pad", create_desc({10, 20, 30, 40}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({10, 20, 30, 40}, ge::DT_FLOAT16));
    op.UpdateInputDesc("invert_std", create_desc({10, 20, 30, 40}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {10, 20, 30, 40};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_infershape_case_1)
{
    ge::op::SyncBatchNormBackwardReduce op;
    op.UpdateInputDesc("sum_dy", create_desc({128, 256}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum_dy_dx_pad", create_desc({128, 256}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({128, 256}, ge::DT_FLOAT));
    op.UpdateInputDesc("invert_std", create_desc({128, 256}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {128, 256};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_infershape_case_2)
{
    ge::op::SyncBatchNormBackwardReduce op;
    op.UpdateInputDesc("sum_dy", create_desc({2, 3, 4}, ge::DT_BF16));
    op.UpdateInputDesc("sum_dy_dx_pad", create_desc({2, 3, 4}, ge::DT_BF16));
    op.UpdateInputDesc("mean", create_desc({2, 3, 4}, ge::DT_BF16));
    op.UpdateInputDesc("invert_std", create_desc({2, 3, 4}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {2, 3, 4};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_infershape_case_3)
{
    ge::op::SyncBatchNormBackwardReduce op;
    op.UpdateInputDesc("sum_dy", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sum_dy_dx_pad", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("mean", create_desc({-2}, ge::DT_FLOAT16));
    op.UpdateInputDesc("invert_std", create_desc({-2}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {-2};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_infershape_case_4)
{
    ge::op::SyncBatchNormBackwardReduce op;
    op.UpdateInputDesc("sum_dy", create_desc({1, 2, 3, 4, 5, 6, 7, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum_dy_dx_pad", create_desc({1, 2, 3, 4, 5, 6, 7, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({1, 2, 3, 4, 5, 6, 7, 8}, ge::DT_FLOAT));
    op.UpdateInputDesc("invert_std", create_desc({1, 2, 3, 4, 5, 6, 7, 8}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormBackwardReduce"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormBackwardReduce")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref_fp16 = ge::DT_FLOAT16;
        ge::DataType output_ref_fp16 = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref_fp16, &input_ref_fp16, &input_ref_fp16, &input_ref_fp16})
                                  .OutputDataTypes({&output_ref_fp16, &output_ref_fp16})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_fp16);
    }
}

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormBackwardReduce"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormBackwardReduce")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref_fp32 = ge::DT_FLOAT;
        ge::DataType output_ref_fp32 = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref_fp32, &input_ref_fp32, &input_ref_fp32, &input_ref_fp32})
                                  .OutputDataTypes({&output_ref_fp32, &output_ref_fp32})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_fp32);
    }
}

TEST_F(SyncBatchNormBackwardReduceInferShapeTest, SyncBatchNormBackwardReduce_InferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormBackwardReduce"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormBackwardReduce")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref_bf16 = ge::DT_BF16;
        ge::DataType output_ref_bf16 = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(4)
                                  .NodeIoNum(4, 2)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref_bf16, &input_ref_bf16, &input_ref_bf16, &input_ref_bf16})
                                  .OutputDataTypes({&output_ref_bf16, &output_ref_bf16})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref_bf16);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref_bf16);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_bf16);
    }
}