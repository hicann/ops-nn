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
 * \file test_conv2d_infer_pass.cpp
 * \brief Conv2D InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForConv2D(gert::InferDataTypeContext* context);
}

namespace {
gert::ContextHolder<gert::InferDataTypeContext> BuildConv2DDtypeContext(ge::DataType xDtype, ge::DataType wDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Conv2D").OpName("Conv2D");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, xDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorDesc(1, wDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    return builder.Build();
}
} // namespace

TEST(Conv2DGraphInfer, InferDataTypeInt8ToInt32)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_INT8, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(Conv2DGraphInfer, InferDataTypeInt4ToInt32)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_INT4, ge::DT_INT4);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(Conv2DGraphInfer, InferDataTypeFp16WeightInt8ToInt32)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_FLOAT16, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(Conv2DGraphInfer, InferDataTypeHifloat8Keep)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_HIFLOAT8, ge::DT_HIFLOAT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_HIFLOAT8);
}

TEST(Conv2DGraphInfer, RejectsNullContext) { EXPECT_EQ(ops::InferDataTypeForConv2D(nullptr), ge::GRAPH_FAILED); }

TEST(Conv2DGraphInfer, InferDataTypeInt16ToInt32)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_INT16, ge::DT_INT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(Conv2DGraphInfer, InferDataTypeFloatWeightInt8ToInt32)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_FLOAT, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(Conv2DGraphInfer, InferDataTypeFloatKeep)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(Conv2DGraphInfer, InferDataTypeBf16Keep)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(Conv2DGraphInfer, InferDataTypeFloat8E4m3fnKeep)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT8_E4M3FN);
}

TEST(Conv2DGraphInfer, InferDataTypeInt8WeightFloatStillInt32)
{
    auto holder = BuildConv2DDtypeContext(ge::DT_INT8, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}
