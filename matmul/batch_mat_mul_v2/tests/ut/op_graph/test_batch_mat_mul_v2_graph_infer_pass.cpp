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
 * \file test_batch_mat_mul_v2_graph_infer_pass.cpp
 * \brief BatchMatMulV2 InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeBatchMatMulV2(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kRequiredInputNum = 2;
constexpr size_t kInputNum = 3; // x1/x2/bias
constexpr size_t kOutputNum = 1;

gert::ContextHolder<gert::InferDataTypeContext> BuildBatchMatMulDtypeContext(ge::DataType x1Dtype, ge::DataType x2Dtype,
                                                                             bool withBias = false,
                                                                             ge::DataType biasDtype = ge::DT_UNDEFINED)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BatchMatMulV2").OpName("BatchMatMulV2");
    builder.IONum(withBias ? kInputNum : kRequiredInputNum, kOutputNum);
    builder.InputTensorDesc(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    if (withBias) {
        builder.InputTensorDesc(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

TEST(BatchMatMulV2GraphInfer, InferDataTypeFp32OutFp32)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeFp16OutFp16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeBf16OutBf16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeInt32OutInt32)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT32, ge::DT_INT32);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeHiFloat8OutHiFloat8)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_HIFLOAT8, ge::DT_HIFLOAT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_HIFLOAT8), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_HIFLOAT8);
}

// 对齐canndev RT1:不校验x1与x2一致,y跟随x1(伪量化场景x1与x2 dtype可以不同)
TEST(BatchMatMulV2GraphInfer, MixedFp16Bf16InputYFollowsX1)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 950/mc62组合表无16进32出列, fp16声明的fp32被覆盖为fp16(canndev的split-k分支依赖NZ格式条件, RT2.0无法评估故不保留)
TEST(BatchMatMulV2GraphInfer, DeclaredFp32OutOverriddenToFp16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_FLOAT), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 对齐canndev RT1.0的SetMatMulOutputDtype: hifp8输入尊重声明的y(量化融合图需fp16/bf16/fp32输出)
TEST(BatchMatMulV2GraphInfer, DeclaredYRespectedForHifp8Input)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_HIFLOAT8, ge::DT_HIFLOAT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 对齐canndev传播语义: hifp8输入时y无条件取声明的值, 未声明(UNDEFINED)也原样传播
TEST(BatchMatMulV2GraphInfer, Hifp8PropagatesDeclaredY)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_HIFLOAT8, ge::DT_HIFLOAT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_UNDEFINED);
}

// 输出槽未初始化(builder残留DT_MAX)时视同未声明y,y跟随x1,不因残留值推导失败
TEST(BatchMatMulV2GraphInfer, InferDataTypeUninitializedOutputSlot)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BatchMatMulV2").OpName("BatchMatMulV2");
    builder.IONum(kRequiredInputNum, kOutputNum);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetOutputDataType(0), ge::DT_MAX);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// v2支持int8/int4输入,整数累加y为int32
TEST(BatchMatMulV2GraphInfer, InferDataTypeInt8InInt32Out)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT8, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeInt4InInt32Out)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT4, ge::DT_INT4);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeFp16InWithFp16Bias)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 对齐canndev RT1:不校验bias dtype
TEST(BatchMatMulV2GraphInfer, Int4BiasWithFp16Input)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_INT4);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(BatchMatMulV2GraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeBatchMatMulV2(nullptr), ge::GRAPH_FAILED);
}
