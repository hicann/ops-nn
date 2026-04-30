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
 * \file test_sync_batch_norm_gather_stats_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/sync_batch_norm_gather_stats_proto.h"

class SyncBatchNormGatherStatsInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SyncBatchNormGatherStats Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SyncBatchNormGatherStats Proto Test TearDown" << std::endl;
    }
};

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_infershape_case_0)
{
    ge::op::SyncBatchNormGatherStats op;
    op.UpdateInputDesc("total_sum", create_desc({8, 256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("total_square_sum", create_desc({8, 256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sample_count", create_desc({8}, ge::DT_INT32));
    op.UpdateInputDesc("mean", create_desc({256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("variance", create_desc({256}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {256};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_infershape_case_1)
{
    ge::op::SyncBatchNormGatherStats op;
    op.UpdateInputDesc("total_sum", create_desc({4, 128}, ge::DT_FLOAT));
    op.UpdateInputDesc("total_square_sum", create_desc({4, 128}, ge::DT_FLOAT));
    op.UpdateInputDesc("sample_count", create_desc({4}, ge::DT_INT32));
    op.UpdateInputDesc("mean", create_desc({128}, ge::DT_FLOAT));
    op.UpdateInputDesc("variance", create_desc({128}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {128};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_infershape_case_2)
{
    ge::op::SyncBatchNormGatherStats op;
    op.UpdateInputDesc("total_sum", create_desc({16, 512}, ge::DT_BF16));
    op.UpdateInputDesc("total_square_sum", create_desc({16, 512}, ge::DT_BF16));
    op.UpdateInputDesc("sample_count", create_desc({16}, ge::DT_INT32));
    op.UpdateInputDesc("mean", create_desc({512}, ge::DT_BF16));
    op.UpdateInputDesc("variance", create_desc({512}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {512};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_infershape_case_3)
{
    ge::op::SyncBatchNormGatherStats op;
    op.UpdateInputDesc("total_sum", create_desc({2, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("total_square_sum", create_desc({2, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sample_count", create_desc({2}, ge::DT_INT32));
    op.UpdateInputDesc("mean", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("variance", create_desc({64}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {64};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_infershape_case_4)
{
    ge::op::SyncBatchNormGatherStats op;
    op.UpdateInputDesc("total_sum", create_desc({32, 1024}, ge::DT_FLOAT));
    op.UpdateInputDesc("total_square_sum", create_desc({32, 1024}, ge::DT_FLOAT));
    op.UpdateInputDesc("sample_count", create_desc({32}, ge::DT_INT32));
    op.UpdateInputDesc("mean", create_desc({1024}, ge::DT_FLOAT));
    op.UpdateInputDesc("variance", create_desc({1024}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected_shape = {1024};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), expected_shape);
    EXPECT_EQ(op.GetOutputDesc(3).GetShape().GetDims(), expected_shape);
}

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormGatherStats"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormGatherStats")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref_fp16 = ge::DT_FLOAT16;
        ge::DataType input_ref_int32 = ge::DT_INT32;
        ge::DataType output_ref_fp16 = ge::DT_FLOAT16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(5)
                .NodeIoNum(5, 4)
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref_fp16, &input_ref_fp16, &input_ref_int32, &input_ref_fp16, &input_ref_fp16})
                .OutputDataTypes({&output_ref_fp16, &output_ref_fp16, &output_ref_fp16, &output_ref_fp16})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref_fp16);
        EXPECT_EQ(context->GetOutputDataType(3), output_ref_fp16);
    }
}

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormGatherStats"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormGatherStats")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref_fp32 = ge::DT_FLOAT;
        ge::DataType input_ref_int32 = ge::DT_INT32;
        ge::DataType output_ref_fp32 = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(5)
                .NodeIoNum(5, 4)
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref_fp32, &input_ref_fp32, &input_ref_int32, &input_ref_fp32, &input_ref_fp32})
                .OutputDataTypes({&output_ref_fp32, &output_ref_fp32, &output_ref_fp32, &output_ref_fp32})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref_fp32);
        EXPECT_EQ(context->GetOutputDataType(3), output_ref_fp32);
    }
}

TEST_F(SyncBatchNormGatherStatsInferShapeTest, SyncBatchNormGatherStats_InferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormGatherStats"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SyncBatchNormGatherStats")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref_bf16 = ge::DT_BF16;
        ge::DataType input_ref_int32 = ge::DT_INT32;
        ge::DataType output_ref_bf16 = ge::DT_BF16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .IrInputNum(5)
                .NodeIoNum(5, 4)
                .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref_bf16, &input_ref_bf16, &input_ref_int32, &input_ref_bf16, &input_ref_bf16})
                .OutputDataTypes({&output_ref_bf16, &output_ref_bf16, &output_ref_bf16, &output_ref_bf16})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref_bf16);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref_bf16);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref_bf16);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref_bf16);
        EXPECT_EQ(context->GetOutputDataType(3), output_ref_bf16);
    }
}