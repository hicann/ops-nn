/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_depthwise_conv2d_infer_datatype.cpp
 * \brief DepthwiseConv2D InferDataType UT (op_graph)
 *
 * x 为 INT8/INT4/INT16（量化）时 y 为 INT32，其余与 x 一致。
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForDepthwiseConv2D(gert::InferDataTypeContext* context);
}

namespace {
gert::ContextHolder<gert::InferDataTypeContext> BuildDwDtypeContext(ge::DataType xDtype, ge::DataType wDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("DepthwiseConv2D").OpName("DepthwiseConv2D");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, xDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorDesc(1, wDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    return builder.Build();
}
} // namespace

TEST(DepthwiseConv2DGraphInfer, InferDataTypeInt8ToInt32)
{
    auto holder = BuildDwDtypeContext(ge::DT_INT8, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDepthwiseConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(DepthwiseConv2DGraphInfer, InferDataTypeInt4ToInt32)
{
    auto holder = BuildDwDtypeContext(ge::DT_INT4, ge::DT_INT4);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDepthwiseConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

// 混合精度：INT16 fmap + INT8 weight，y 为 INT32
TEST(DepthwiseConv2DGraphInfer, InferDataTypeInt16FmapInt8WeightToInt32)
{
    auto holder = BuildDwDtypeContext(ge::DT_INT16, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDepthwiseConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(DepthwiseConv2DGraphInfer, InferDataTypeInt16ToInt32)
{
    auto holder = BuildDwDtypeContext(ge::DT_INT16, ge::DT_INT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDepthwiseConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(DepthwiseConv2DGraphInfer, InferDataTypeFp16Passthrough)
{
    auto holder = BuildDwDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDepthwiseConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(DepthwiseConv2DGraphInfer, InferDataTypeFp32Passthrough)
{
    auto holder = BuildDwDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDepthwiseConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(DepthwiseConv2DGraphInfer, InferDataTypeNullContextFail)
{
    EXPECT_EQ(ops::InferDataTypeForDepthwiseConv2D(nullptr), ge::GRAPH_FAILED);
}
