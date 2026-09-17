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

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "exe_graph/runtime/runtime_tensor.h"

namespace ops {
ge::graphStatus InferShapeForINTrainingReduceV2(gert::InferShapeContext* context);
ge::graphStatus InferDataType4INTrainingReduceV2(gert::InferDataTypeContext* context);
} // namespace ops

TEST(INTrainingReduceV2GraphInferTest, graph_infer_nd_channel_first_001)
{
    const gert::StorageShape inputShape({2, 3, 5, 7}, {2, 3, 5, 7});
    const gert::StorageFormat inputFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor inputTensor(inputShape, inputFormat, ge::DT_FLOAT);

    gert::OpInferShapeContextBuilder builder;
    builder.OpType("INTrainingReduceV2").OpName("INTrainingReduceV2");
    builder.IONum(1, 2);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensors({&inputTensor});
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferShapeForINTrainingReduceV2(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    ASSERT_NE(context->GetOutputShape(1), nullptr);
    EXPECT_EQ(*context->GetOutputShape(0), gert::Shape({2, 3, 1, 1}));
    EXPECT_EQ(*context->GetOutputShape(1), gert::Shape({2, 3, 1, 1}));
}

TEST(INTrainingReduceV2GraphInferTest, graph_infer_dtype_fp32_002)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("INTrainingReduceV2").OpName("INTrainingReduceV2");
    builder.IONum(1, 2);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataType4INTrainingReduceV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}

TEST(INTrainingReduceV2GraphInferTest, graph_infer_rejects_null_context_003)
{
    EXPECT_EQ(ops::InferShapeForINTrainingReduceV2(nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(ops::InferDataType4INTrainingReduceV2(nullptr), ge::GRAPH_FAILED);
}

TEST(INTrainingReduceV2GraphInferTest, graph_infer_rejects_invalid_nd_rank_004)
{
    const gert::StorageShape inputShape({8}, {8});
    const gert::StorageFormat inputFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    gert::Tensor inputTensor(inputShape, inputFormat, ge::DT_FLOAT);

    gert::OpInferShapeContextBuilder builder;
    builder.OpType("INTrainingReduceV2").OpName("INTrainingReduceV2");
    builder.IONum(1, 2);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensors({&inputTensor});
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    EXPECT_EQ(ops::InferShapeForINTrainingReduceV2(holder.GetContext()), ge::GRAPH_FAILED);
}
