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

namespace ops {
ge::graphStatus InferDataTypeForINTrainingUpdateGradGammaBeta(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
gert::ContextHolder<gert::InferDataTypeContext> BuildContext(ge::DataType gammaType, ge::DataType betaType)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("INTrainingUpdateGradGammaBeta").OpName("INTrainingUpdateGradGammaBeta");
    builder.IONum(2, 2);
    builder.InputTensorDesc(0, gammaType, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorDesc(1, betaType, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.OutputTensorDesc(1, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    return builder.Build();
}
} // namespace

TEST(INTrainingUpdateGradGammaBetaGraphInferTest, infer_float_outputs)
{
    auto holder = BuildContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(ops::InferDataTypeForINTrainingUpdateGradGammaBeta(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}

TEST(INTrainingUpdateGradGammaBetaGraphInferTest, reject_mismatched_input_types)
{
    auto holder = BuildContext(ge::DT_FLOAT, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeForINTrainingUpdateGradGammaBeta(context), ge::GRAPH_FAILED);
}

TEST(INTrainingUpdateGradGammaBetaGraphInferTest, reject_declared_output_type_mismatch)
{
    auto holder = BuildContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ops::InferDataTypeForINTrainingUpdateGradGammaBeta(context), ge::GRAPH_FAILED);
}

TEST(INTrainingUpdateGradGammaBetaGraphInferTest, reject_null_context)
{
    EXPECT_EQ(ops::InferDataTypeForINTrainingUpdateGradGammaBeta(nullptr), ge::GRAPH_FAILED);
}
