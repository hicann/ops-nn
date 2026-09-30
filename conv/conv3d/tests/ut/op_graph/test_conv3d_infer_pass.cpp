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
 * \file test_conv3d_infer_pass.cpp
 * \brief Conv3D 算子 RT2.0 InferDataType UT（迁移自 canndev RT1.0 SetOutDtype）
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeConv3D(gert::InferDataTypeContext* context);
}

TEST(Conv3DGraphInfer, InferDataTypeFP16)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Conv3D").OpName("Conv3D");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataTypeConv3D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(Conv3DGraphInfer, InferDataTypeFP32)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Conv3D").OpName("Conv3D");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataTypeConv3D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(Conv3DGraphInfer, InferDataTypeBF16)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Conv3D").OpName("Conv3D");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.InputTensorDesc(1, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataTypeConv3D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(Conv3DGraphInfer, InferDataTypeHiFloat8)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Conv3D").OpName("Conv3D");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_HIFLOAT8, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.InputTensorDesc(1, ge::DT_HIFLOAT8, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataTypeConv3D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_HIFLOAT8);
}

// int8 输入推导 int32 输出（对齐 RT1.0 SetOutDtype）
TEST(Conv3DGraphInfer, InferDataTypeInt8ToInt32)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Conv3D").OpName("Conv3D");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_INT8, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.InputTensorDesc(1, ge::DT_INT8, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    ASSERT_EQ(ops::InferDataTypeConv3D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(Conv3DGraphInfer, RejectsNullContext) { EXPECT_EQ(ops::InferDataTypeConv3D(nullptr), ge::GRAPH_FAILED); }
