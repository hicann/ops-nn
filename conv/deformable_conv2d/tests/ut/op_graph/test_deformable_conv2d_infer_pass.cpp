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
 * \file test_deformable_conv2d_infer_pass.cpp
 * \brief 覆盖DeformableConv2D InferDataType推导：y dtype跟随x，量化输入映射INT32。
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForDeformableConv2D(gert::InferDataTypeContext* context);
}

namespace {
gert::ContextHolder<gert::InferDataTypeContext> BuildDfmDtypeContext(ge::DataType xDtype, ge::DataType wDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("DeformableConv2D").OpName("DeformableConv2D");
    builder.IONum(3, 1);
    builder.InputTensorDesc(0, xDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorDesc(1, wDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.InputTensorDesc(2, xDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    return builder.Build();
}
} // namespace

TEST(DeformableConv2DGraphInfer, InferDataTypeFp16Kept)
{
    auto holder = BuildDfmDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDeformableConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(DeformableConv2DGraphInfer, InferDataTypeFp32Kept)
{
    auto holder = BuildDfmDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDeformableConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(DeformableConv2DGraphInfer, InferDataTypeBf16Kept)
{
    auto holder = BuildDfmDtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDeformableConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(DeformableConv2DGraphInfer, InferDataTypeInt8ToInt32)
{
    auto holder = BuildDfmDtypeContext(ge::DT_INT8, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForDeformableConv2D(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(DeformableConv2DGraphInfer, InferDataTypeNullContextFail)
{
    ASSERT_EQ(ops::InferDataTypeForDeformableConv2D(nullptr), ge::GRAPH_FAILED);
}
